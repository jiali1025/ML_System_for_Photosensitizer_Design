# To run AL iterations experiments

from argparse import ArgumentParser 
import tensorflow as tf 
import numpy as np
import pandas as pd

# set random seed
tf.random.set_seed(123)
np.random.seed(123)

import deepchem as dc 

# edited graphConvModel
from graphConvModel import GraphConvModel
from active_learning import recommend_smiles_MEI

def parse_arguments():
    parser = ArgumentParser(description = "Model training and active learning iteration")
    parser.add_argument('--labelled', '-l', type=str,
                        default='DA_Cycle8_Cummulative_Labelled_DA8051.csv',
                        help = "CSV containing labelled data")
    parser.add_argument('--unlabelled', '-u', type=str,
                        default= 'Unlabelled_DA_Cycle9_Syn_24382.csv',
                        help = "CSV containing all unlabelled data")
    parser.add_argument('--tasks', '-t', nargs = '+',
                        default = ['S1T1 Gap', 'Bandgap'])
    parser.add_argument('--save', '-s', type=str,
                        default='Trained_Model') 
    parser.add_argument('--out1', '-o1', type=str,
                        default='train_results.csv') 
    parser.add_argument('--out2', '-o2', type=str,
                        default='EI_pred_results.csv') 
    return parser.parse_args()


def main():
    args = parse_arguments()
    labelled_csv = args.labelled
    unlabelled_csv = args.unlabelled
    tasks = args.tasks
    save_model = args.save
    train_results_csv = args.out1
    EI_pred_csv = args.out2

    graph_featurizer = dc.feat.graph_features.ConvMolFeaturizer()
    loader = dc.data.data_loader.CSVLoader(tasks = tasks, 
                                            smiles_field = "Generated Molecule SMILES", 
                                            id_field = "Generated Molecule Name", 
                                            featurizer=graph_featurizer)

    # train full dataset 
    data_dir = "./Data"
    dataset = loader.create_dataset(labelled_csv, data_dir)

    # get the hyperparams
    with open("default_params.txt", 'r') as f:
        lines = f.readlines()
    gc_values = lines[0].strip().split(":")[1].split(",")
    gc_layers = []
    for gc in gc_values:
        gc_layers.append(int(gc))
    dense_values = lines[1].strip().split(":")[1].split(",")
    dense_layers = []
    for dense in dense_values:
        dense_layers.append(int(dense))
    learning_rate = float(lines[2].strip().split(":")[1])
    dropout = float(lines[3].strip().split(":")[1])
    if dropout == 0.0:
        dropout = 0.01 # default for uncertainty
    batch_size = int(lines[4].strip().split(":")[1])
    num_epochs = int(lines[5].strip().split(":")[1])

    metric_list = [
        dc.metrics.Metric( dc.metrics.pearson_r2_score), 
        dc.metrics.Metric( dc.metrics.rms_score), 
        dc.metrics.Metric( dc.metrics.mae_score)
        ]

    model_dir = "./" + save_model
    model = GraphConvModel(n_tasks = len(tasks),
                            graph_conv_layers = gc_layers,
                            dense_layers = dense_layers,
                            dropout = dropout, 
                            mode = 'regression',
                            learning_rate = learning_rate,
                            batch_size = batch_size,
                            uncertainty = True, # uncertainty must be true 
                            model_dir = model_dir)
    logdict = {
        "Epoch": [],
        "Train MAE ST": [],
        "Train R2 ST": [],
        "Train RMSE ST": [],
        "Train MAE BG": [],
        "Train R2 BG": [],
        "Train RMSE BG": []
    }

    for i in range(num_epochs):
        model.fit(dataset, nb_epoch = 1)
        train_scores = model.evaluate(dataset, metric_list, [], per_task_metrics = True)[1]
        print(f"Epoch {i} - ", train_scores) # no transformers
        logdict["Epoch"].append(i+1)
        logdict["Train MAE ST"].append(train_scores['mae_score'][0])
        logdict["Train R2 ST"].append(train_scores['pearson_r2_score'][0])
        logdict["Train RMSE ST"].append(train_scores['rms_score'][0])
        logdict["Train MAE BG"].append(train_scores['mae_score'][1])
        logdict["Train R2 BG"].append(train_scores['pearson_r2_score'][1])
        logdict["Train RMSE BG"].append(train_scores['rms_score'][1])

    # A1. Save model and model weights
    if save_model != "None": # default
        model_dir = "./" + save_model
        model.save_checkpoint(model_dir = model_dir)

    # A2. Save model train performance
    train_results_df = pd.DataFrame.from_dict(logdict)
    train_results_df.to_csv(train_results_csv)

    # B1. Get predicted ST and BG, their uncertainties, expected improvement of ST
    final_df, all_idx_desc, pred, std = recommend_smiles_MEI(unlabelled_csv = unlabelled_csv,
                                                            tradeoff = 0.01, 
                                                            labelled_train_set = dataset,
                                                            model = model)
    # then save
    final_df.to_csv(EI_pred_csv)


if __name__ == '__main__':
    main()