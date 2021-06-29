import numpy as np
import pandas as pd

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

def create_mol_data(MOLNAMES = ["DAn_694"]):
    final_unlabelled_DA_space = pd.read_csv("Final_Unlabelled_DA_DAn_Space_142452.csv")
    full_list = list(final_unlabelled_DA_space["Generated Molecule Name"].values)
    all_smiles = []
    final_molnames = []
    for molname in MOLNAMES:
        if molname in full_list:
            final_molnames.append(molname)
            smi = final_unlabelled_DA_space[final_unlabelled_DA_space["Generated Molecule Name"] == molname]["Generated Molecule SMILES"].values[0]
            all_smiles.append(smi)
    mols = [Chem.MolFromSmiles(smi) for smi in all_smiles]
    graph_list = graph_featurizer.featurize(mols)
    data = dc.data.NumpyDataset(graph_list, ids = final_molnames)
    return data

# Gather all DA
DA_init_df = pd.read_excel("Compiled Labelled DA PS 20210624.xlsx", "DA0")
all_DA_dfs = [DA_init_df]
for i in range(1, 9):
    AL_df =  pd.read_excel("Compiled Labelled DA PS 20210624.xlsx", sheet_name = f"DA_AL{i}")
    all_DA_dfs.append(AL_df)
# DA_final_df = pd.read_excel("Compiled Labelled DA PS 20210624.xlsx", sheet_name = "DA5")
final_unlabelled_DA_space = pd.read_csv("Final_Unlabelled_DA_DAn_Space_142452.csv")

# Restore model
DAn_694_data = create_mol_data(["DAn_694"]) # get a test molecule DAn_694
current_tasks = ["S1T1 Gap", "Bandgap"]
model = GraphConvModel(n_tasks = len(current_tasks),
                       graph_conv_layers = [295, 295, 295, 295, 295, 295],
                       dense_layers = [382, 382, 382, 382],
                       dropout = 0.00874,
                       learning_rate = 0.0001,
                       batch_size = 10,
                       uncertainty = True,
                       model_dir = "Trained_Model")
model.restore(model.get_checkpoints()[-1])
model.predict_uncertainty(DAn_694_data)
model.restore(model.get_checkpoints()[-1]) # real restore 
DAn_694_pred = model.predict(DAn_694_data)
print("Bef: [0.1648583, ~0.06783755] <--> Now: ", DAn_694_pred)

### 1. Predict embedding for final unlabelled space ###
print()
print("1. Running embedding predictions for unlabelled dataset...")
unlabelled_names = list(final_unlabelled_DA_space["Generated Molecule Name"].values)
unlabelled_dataset = create_mol_data(unlabelled_names)
unlabelled_nfps = model.predict_embedding(unlabelled_dataset)
with open("DA_unlabelled_nfps.npy", "wb") as f:
    np.save(f, unlabelled_nfps)

### 2. Predict embedding for initial labelled DA ### 
print()
print("2. Running embedding predictions for initial labelled dataset...")
DA_init_df["Mols"] = [Chem.MolFromSmiles(smi) for smi in DA_init_df["SMILES"]]
initial_mol_list = list(DA_init_df["Mols"].values)
initial_graph_list = graph_featurizer.featurize(initial_mol_list)
DA_initial_set = dc.data.NumpyDataset(X = initial_graph_list, 
                               y = np.vstack((
                                   DA_init_df["S1T1 Gap"].values, 
                                   DA_init_df["Bandgap"].values)).T,
                               ids = DA_init_df["Molname"]
                              )
initial_DA_nfps = model.predict_embedding(DA_initial_set)
with open("initial_DA_labelled_nfps.npy", "wb") as f:
    np.save(f, initial_DA_nfps)


### 3.  Predict embedding for each DA AL cycle ### 
print()
print("3. Running embedding predictions for all DA AL...")
all_DA_AL_datasets = []
all_DA_AL_nfps = []
for df in all_DA_dfs:
    df["Mols"] = [Chem.MolFromSmiles(smi) for smi in df["SMILES"]]
    current_mol_list = list(df["Mols"].values)
    current_graph_list = graph_featurizer.featurize(current_mol_list)
    current_data_set = dc.data.NumpyDataset(X = current_graph_list, 
                                           y = np.vstack((
                                               df["S1T1 Gap"].values, 
                                               df["Bandgap"].values)).T,
                                           ids = df["Molname"]
                                          )
    all_DA_AL_datasets.append(current_data_set)
for dataset in all_DA_AL_datasets:
    current_nfp = model.predict_embedding(dataset)
    all_DA_AL_nfps.append(current_nfp)
for ind, nfp in enumerate(all_DA_AL_nfps):
    with open(f"DA_AL_nfp_{ind}.npy", "wb") as f:
        np.save(f, nfp)

### 4. Predict embedding for final selected 3 DA ### 
print("4. Running embedding predictions for selected 3 DA...")
selected_3_names = ["DAn_694", "DAn_4493", "DAn_1003"]
selected_3_set = create_mol_data(selected_3_names)
selected_3_nfps = model.predict_embedding(selected_3_set)
with open("DA_final_3_nfps.npy", "wb") as f:
    np.save(f, selected_3_nfps)
print()
print("-------------- ALL DONE --------------")