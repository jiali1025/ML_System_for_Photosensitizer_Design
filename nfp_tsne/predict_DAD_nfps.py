import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import tensorflow as tf 
import deepchem as dc 
from graphConvModel import GraphConvModel

# ignore the future DeepChem deprecation error 
import warnings
warnings.filterwarnings('ignore')
import logging
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

# set random seed
tf.random.set_seed(42)
np.random.seed(42)

from rdkit import Chem

graph_featurizer = dc.feat.graph_features.ConvMolFeaturizer()

# Gather all DAD
DAD_init_df = pd.read_excel("Compiled Labelled DAD PS 20210510.xlsx", "DAD0")
all_DAD_dfs = [DAD_init_df]
for i in range(1, 11):
    AL_df =  pd.read_excel("Compiled Labelled DAD PS 20210510.xlsx", sheet_name = f"DAD_AL{i}")
    all_DAD_dfs.append(AL_df)
final_unlabelled_DAD_space = pd.read_csv("Unlabelled_randomDAD_DADn_DADx_Space_115455.csv") # 0313: Update scale down 200k to 100k

# Restore model # get a test molecule DADx_1098
DADx_1098_mol = Chem.MolFromSmiles("N#CC(C(C#N)=N1)=NC2=C1C(C=CC(C3=CC=C(N(C4=CC=CC=C4)C5=CC=CC=C5)C=C3)=C6)=C6C7=C2C=CC(C8=CC=C(N(C9=CC=CC=C9)C%10=CC=CC=C%10)C=C8)=C7")
DADx_1098_graph = graph_featurizer.featurize([DADx_1098_mol])
DADx_1098_data = dc.data.NumpyDataset(DADx_1098_graph, ids = ["DADx_1098"])

current_tasks = ["S1T1 Gap", "Bandgap"]
model = GraphConvModel(n_tasks = len(current_tasks),
                       graph_conv_layers = [512, 512, 512, 512], 
                       dense_layers = [128, 128, 128],
                       dropout = 0.01,
                       learning_rate = 0.001,
                       batch_size = 10,
                       uncertainty = True,
                       model_dir = "Trained_Model")
model.restore(model.get_checkpoints()[-1])
model.predict_uncertainty(DADx_1098_data)
model.restore(model.get_checkpoints()[-1]) # real restore 
DADx_1098_pred = model.predict(DADx_1098_data)
print("Bef: [0.19907482, ~0.09498724] <--> Now: ", DADx_1098_pred)

### 1. Predict embedding for final unlabelled space ###
print()
print("1. Running embedding predictions for unlabelled dataset...")
unlabelled_names = list(final_unlabelled_DAD_space["Generated Molecule Name"].values)
all_smiles = list(final_unlabelled_DAD_space["Generated Molecule SMILES"].values)
print("Done smiles for unlabelled")
mols = [Chem.MolFromSmiles(smi) for smi in all_smiles]
print("Done mols for unlabelled")
mols1 = mols[:25000]
mols2 = mols[25000:50000]
mols3 = mols[50000:75000]
mols4 = mols[75000:]
all_graphs = []
for current_mols in [mols1,mols2,mols3,mols4]:
    graph_list = graph_featurizer.featurize(current_mols)
    print("Done 1 part of graphs for unlabelled")
    all_graphs.append(graph_list)
unlabelled_data1 = dc.data.NumpyDataset(all_graphs[0], ids = unlabelled_names[:25000])
unlabelled_data2 = dc.data.NumpyDataset(all_graphs[1], ids = unlabelled_names[25000:50000])
unlabelled_data3 = dc.data.NumpyDataset(all_graphs[2], ids = unlabelled_names[50000:75000])
unlabelled_data4 = dc.data.NumpyDataset(all_graphs[3], ids = unlabelled_names[75000:])

for ind, current_unlabelled_data in enumerate([unlabelled_data1, unlabelled_data2, unlabelled_data3, unlabelled_data4]):
    print(f"Currently running {ind} unlabelled data")
    unlabelled_nfps = model.predict_embedding(current_unlabelled_data)
    with open(f"DAD_unlabelled_nfps_{ind}.npy", "wb") as f:
        np.save(f, unlabelled_nfps)

### 2. Predict embedding for initial labelled DAD ### 
print()
print("2. Running embedding predictions for initial labelled dataset...")
DAD_init_df["Mols"] = [Chem.MolFromSmiles(smi) for smi in DAD_init_df["SMILES"]]
initial_mol_list = list(DAD_init_df["Mols"].values)
initial_graph_list = graph_featurizer.featurize(initial_mol_list)
DAD_initial_set = dc.data.NumpyDataset(X = initial_graph_list, 
                               y = np.vstack((
                                   DAD_init_df["S1T1 Gap"].values, 
                                   DAD_init_df["Bandgap"].values)).T,
                               ids = DAD_init_df["Molname"]
                              )
initial_DAD_nfps = model.predict_embedding(DAD_initial_set)
with open("initial_DAD_labelled_nfps.npy", "wb") as f:
    np.save(f, initial_DAD_nfps)


### 3. Predict embedding for each DAD AL cycle ### 
print()
print("3. Running embedding predictions for all DAD AL...")
all_DAD_AL_datasets = []
all_DAD_AL_nfps = []
for df in all_DAD_dfs:
    df["Mols"] = [Chem.MolFromSmiles(smi) for smi in df["SMILES"]]
    current_mol_list = list(df["Mols"].values)
    current_graph_list = graph_featurizer.featurize(current_mol_list)
    current_data_set = dc.data.NumpyDataset(X = current_graph_list, 
                                           y = np.vstack((
                                               df["S1T1 Gap"].values, 
                                               df["Bandgap"].values)).T,
                                           ids = df["Molname"]
                                          )
    all_DAD_AL_datasets.append(current_data_set)
for dataset in all_DAD_AL_datasets:
    current_nfp = model.predict_embedding(dataset)
    all_DAD_AL_nfps.append(current_nfp)
for ind, nfp in enumerate(all_DAD_AL_nfps):
    with open(f"DAD_AL_nfp_{ind}.npy", "wb") as f:
        np.save(f, nfp)

### 4. Predict embedding for final selected DAD ### 
print("4. Running embedding predictions for selected DADx1098...")
DADx_1098_nfp = model.predict_embedding(DADx_1098_data)
with open("DADx_1098.npy", "wb") as f:
    np.save(f, DADx_1098_nfp)
print()
print("-------------- ALL DONE --------------")