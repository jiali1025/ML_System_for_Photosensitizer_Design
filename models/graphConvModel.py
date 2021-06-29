import collections
import numpy as np
import tensorflow as tf

from typing import List, Union, Tuple, Iterable, Dict
from deepchem.feat.mol_graphs import ConvMol
from deepchem.models import KerasModel, layers
from deepchem.models.losses import L2Loss
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
"""
Replicate GraphConvModel from DeepChem 2.3.0 and
apply to regression in our application specifically

Class GraphConvModel(): it inheritance from the KerasModel from deepchem.models 
so it could use its attributes and methods

Edited GraphConvModel from deepchem github source 
- Included tunability for multiple dense layers instead of just 1 by default
- However this dense layer is more like a re-parameterization method before the graph gather layer, so it is a node-wise
dense layer, it is shared for all nodes.
- the graph-level prediction is simply one node after the neural-fingerprint has been created, so it may not be that good
since it is just a weighted sum.
- Added modified dense layer after the representation. 

"""

class _GraphConvKerasModel(tf.keras.Model):
    def __init__(self, 
                n_tasks, 
                graph_conv_layers = [128, 128], 
                dense_layers = [64, 64], # tunable dense layers
                dropout = 0.01, 
                number_atom_features = 75, 
                uncertainty = True,
                batch_size = 8,
                batch_normalize = True,
                mode = "regression", # or predict
                **kwargs):

        super(_GraphConvKerasModel, self).__init__()
        self.uncertainty = uncertainty

        if not isinstance(dropout, collections.Sequence):
            dropout = [dropout] * (len(graph_conv_layers) + len(dense_layers))
        if len(dropout) != len(graph_conv_layers) + len(dense_layers):
            raise ValueError("Wrong number of dropout probabilities provided")
        if uncertainty and any(d == 0.0 for d in dropout):
            raise ValueError("Dropout must be included in every layer to predict uncertainty!")
        
        self.graph_convs = [
            layers.GraphConv(layer_size, activation_fn = tf.nn.relu)
            for layer_size in graph_conv_layers
        ]
        self.graph_pools = [layers.GraphPool() for x in graph_conv_layers] 

        # multiple dense layers; change batch norm and dropout to consider more than 1 dense layers too 
        self.dense_layers = [
            Dense(layer_size, activation = tf.nn.relu)
            for layer_size in dense_layers
        ]
        self.batch_norms = [
            BatchNormalization(fused = False) if batch_normalize else None
            for x in range(len(graph_conv_layers) + len(dense_layers))
        ]
        self.dropouts = [
            Dropout(rate = dropout_rate) if dropout_rate > 0.0 else None for dropout_rate in dropout
        ]

        self.graph_gather = layers.GraphGather(batch_size = batch_size, activation_fn = tf.nn.tanh)
        self.trim = TrimGraphOutput()

        self.regression_dense = Dense(n_tasks)
        if self.uncertainty:
            self.uncertainty_dense = Dense(n_tasks)
            self.uncertainty_trim = TrimGraphOutput()
            self.uncertainty_activation = Activation(tf.exp)

    def call(self, inputs, training = False):
        atom_features = inputs[0]
        degree_slice = tf.cast(inputs[1], dtype=tf.int32)
        membership = tf.cast(inputs[2], dtype=tf.int32)
        n_samples = tf.cast(inputs[3], dtype=tf.int32)
        deg_adjs = [tf.cast(deg_adj, dtype=tf.int32) for deg_adj in inputs[4:]]

        in_layer = atom_features
        # construct the graphconv layers first
        for i in range(len(self.graph_convs)):
            graphConv_in = [in_layer, degree_slice, membership] + deg_adjs
            graphConv = self.graph_convs[i](graphConv_in)
            if self.batch_norms[i] is not None:
                graphConv = self.batch_norms[i](graphConv, training = training)
            if training and self.dropouts[i] is not None:
                graphConv = self.dropouts[i](graphConv, training = training)
            graphPool_in = [graphConv, degree_slice, membership] + deg_adjs
            in_layer = self.graph_pools[i](graphPool_in) 

        # feed to dense layers
        startnum = len(self.graph_convs)
        endnum = len(self.dense_layers) + len(self.graph_convs)
        for i in range(startnum, endnum): # cont index for batchnorms and dropouts
            dense = self.dense_layers[i-startnum](in_layer)
            if self.batch_norms[i] is not None:
                dense = self.batch_norms[i](dense, training = training)
            if training and self.dropouts[i] is not None:
                dense = self.dropouts[i](dense, training = training)
            in_layer = dense

        # construct nfp
        neural_fingerprint = self.graph_gather(
            [in_layer, degree_slice, membership] + deg_adjs
            )
        # final regression layer and trim output
        output = self.regression_dense(neural_fingerprint)
        output = self.trim([output, n_samples]) 

        # get outputs 
        if self.uncertainty:
            log_var = self.uncertainty_dense(neural_fingerprint)
            log_var = self.uncertainty_trim([log_var, n_samples])
            var = self.uncertainty_activation(log_var)
            outputs = [output, var, output, log_var, neural_fingerprint]
        else:
            outputs = [output, neural_fingerprint]

        return outputs


class GraphConvModel(KerasModel):
    def __init__(self,
                n_tasks: int,
                graph_conv_layers: List[int] = [128, 128],
                dense_layers: List[int] = [64, 64], 
                dropout: float = 0.0,
                number_atom_features: int = 75,
                batch_size: int = 100,
                batch_normalize: bool = True,
                uncertainty: bool = False,
                **kwargs):

        self.n_tasks = n_tasks
        self.batch_size = batch_size
        self.uncertainty = uncertainty
        
        # get GraphConvModel
        model = _GraphConvKerasModel(
            n_tasks,
            graph_conv_layers = graph_conv_layers,
            dense_layers = dense_layers,
            dropout = dropout,
            number_atom_features = number_atom_features,
            batch_normalize = batch_normalize,
            uncertainty = uncertainty,
            batch_size = batch_size
        )

        # get outputs and loss function
        if self.uncertainty:
            output_types = ['prediction', 'variance', 'loss', 'loss', 'embedding']
            def loss(outputs, labels, weights):
                diff = labels[0] - outputs[0]
                return tf.reduce_mean(diff * diff / tf.exp(outputs[1]) + outputs[1])
        else:
            output_types = ['prediction', 'embedding']
            loss = L2Loss()

        # keras base model         
        super(GraphConvModel, self).__init__(
            model, loss, output_types = output_types, batch_size = batch_size, **kwargs
        )

    def default_generator(self, dataset, epochs=1, mode='fit', deterministic=True, pad_batches=True):
        for epoch in range(epochs): # for all epochs 
            for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
                batch_size=self.batch_size,
                deterministic=deterministic,
                pad_batches=pad_batches):

                multiConvMol = ConvMol.agglomerate_mols(X_b)
                n_samples = np.array(X_b.shape[0])

                inputs = [
                    multiConvMol.get_atom_features(), multiConvMol.deg_slice,
                    np.array(multiConvMol.membership), n_samples
                ]

                for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
                    inputs.append(multiConvMol.get_deg_adjacency_lists()[i] )
                
                yield (inputs, [y_b], [w_b])


class TrimGraphOutput(tf.keras.layers.Layer):
    """
    Trim the output to the correct number of samples.
    Since GraphGather always outputs the fixed size batches, this layer trims the output to
    the number of samples that were in the actual input tensors. 
    """
    def __init__(self, **kwargs):
        super(TrimGraphOutput, self).__init__(**kwargs)

    def call(self, inputs):
        n_samples = tf.squeeze(inputs[1])
        return inputs[0][0: n_samples]
