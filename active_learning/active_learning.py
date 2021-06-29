"""
Active Learning framework on top of DeepChem GraphConv model 
- Datasets flow using dataset_functions from deepchem 
- Bayesian Optimization: Determine molecules to label next as per max expected improvement (Max EI)
The Bayesian optimization in this project used our graph conv base_model as the surrogate model to approximate the
actual function that can describe the relationship between the molecules' topology, atom features and the target
property. From this we can get the uncertainty which is the std in prediction of unlabelled space from the deepchem
dropout models. Since dropout creates several number of models. The traditional BO used gaussian process, but here our
model substitute the GP as the surrogate. Then is acquisition function is chosen. We used Max EI.

"""
import numpy as np 
import pandas as pd 
import deepchem as dc
from rdkit import Chem
from scipy.special import ndtr
from scipy.stats import norm 

def recommend_smiles_MEI(unlabelled_csv = "Unlabelled_DA_Syn_17611.csv",
                            tradeoff = 0.01, 
                            labelled_train_set = None,
                            model = None):

    df_unlabelled = pd.read_csv(unlabelled_csv)
    smiles_list = list(df_unlabelled["Generated Molecule SMILES"])
    '''
    Utlized the rdkit to convert to mols object then use deepchem featurizer convert to graphs then make it
    into deepchem numpy dataset form.
    '''
    mol_list = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    graph_featurizer = dc.feat.graph_features.ConvMolFeaturizer()
    graph_list = graph_featurizer.featurize(mol_list)
    unlabelled_space = dc.data.NumpyDataset(graph_list)

    all_EI_desc, all_idx_desc, pred, std = predict_and_select_MEI(model = model,
                                                                    train_set = labelled_train_set,
                                                                    unlabelled_space = unlabelled_space,
                                                                    tradeoff = tradeoff) 

    all_mols = np.asarray(df_unlabelled)[all_idx_desc]
    final_df = pd.DataFrame({'Name': all_mols[:, 1], 'SMILES': all_mols[:, 3], 'EI_Index': all_idx_desc, "EI": all_EI_desc})
    
    if len(labelled_train_set.y[0]) > 1:
        pred0 = np.asarray([pred[i][0] for i in range(len(pred))])
        pred1 = np.asarray([pred[i][1] for i in range(len(pred))])
        final_df["Predicted ST Gap"] = pred0[all_idx_desc]
        final_df["Predicted Band Gap"] = pred1[all_idx_desc]
        std0 = np.asarray([std[i][0] for i in range(len(std))])
        std1 = np.asarray([std[i][1] for i in range(len(std))])
        final_df["Uncertainty for ST Gap"] = std0[all_idx_desc]
        final_df["Uncertainty for Band Gap"] = std1[all_idx_desc]

    else:
        final_df["Predicted ST Gap"] = pred[all_idx_desc]
        final_df["Uncertainty for ST Gap"] = std[all_idx_desc]

    return final_df, all_idx_desc, pred, std

# Bayesian Optimization acquisition function via max expected improvement
def predict_and_select_MEI(model, train_set, unlabelled_space, tradeoff):
    # first get mean prediction and uncertainty 
    pred, std = model.predict_uncertainty(unlabelled_space)
    
    if len(train_set.y[0]) > 1:
        print("")
        print("Detected more than 1 y value in multitask model, but will only optimize the first value...")
        print("")
        pred0 = np.asarray([pred[i][0] for i in range(len(pred))])
        pred1 = np.asarray([pred[i][1] for i in range(len(pred))])
        std0 = np.asarray([std[i][0] for i in range(len(std))])
        std1 = np.asarray([std[i][1] for i in range(len(std))])
        train0 = np.asarray([train_set.y[i][0] for i in range(len(train_set.y))]) 
        y_min = np.min(train0) 
    else:
        y_min = np.min(train_set.y) # y_min is the smallest observed so far!
        pred0 = pred

    imp = y_min - pred0 - tradeoff 
    z = imp/std0
    EI = imp * ndtr(z) + std0 * norm.pdf(z) # CDF (cumulative distribution function) and PDF (probability density function) 
    EI[std0 == 0.0] = 0.0
    all_EI_desc = np.sort(EI)[::-1] 
    all_idx_desc = np.argsort(EI)[::-1] # position for EI in descending order, will be used to sort all pred std and EI values!

    return all_EI_desc, all_idx_desc, pred, std