import sys
import tensorflow as tf 
import numpy as np
import pandas as pd

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

current_tasks = ["S1T1 Gap", "Bandgap"]
graph_featurizer = dc.feat.graph_features.ConvMolFeaturizer()
loader = dc.data.data_loader.CSVLoader(tasks = current_tasks, 
                                       smiles_field = "SMILES", 
                                       id_field = "MOLNAME", 
                                       featurizer=graph_featurizer)

STRUCTURE_TYPE = str(sys.argv[1])
IF_MULTIPLE = str(sys.argv[2])

if STRUCTURE_TYPE == "DA":
    ALL_TRAIN_CSVS = ["DA_Train_6386_0", 
                    "DA_Train_6495_1", 
                    "DA_Train_6600_2", 
                    "DA_Train_6712_3",
                    "DA_Train_6813_4",
                    "DA_Train_6925_5",
                    "DA_Train_7031_6",
                    "DA_Train_7135_7",
                    "DA_Train_7245_8"]
    TEST_CSV = "DA_Test_806.csv"
    MODEL_PARAMS = {
        "GC_LAYERS": [295, 295, 295, 295, 295, 295],
        "DENSE_LAYERS": [382, 382, 382, 382],
        "DROPOUT": 0.00874,
        "LEARNING_RATE": 0.0001,
        "BATCH_SIZE": 10,
        "MODEL_DIR": "DA_Models/"
    }
    FINAL_CSV_STORE_NAME = "DA_Final_Model_Performances.csv"
    TRAIN_DIR = "DA_DATA/Train/"
    TEST_DIR = "DA_DATA/Test"

elif STRUCTURE_TYPE == "DADAD10":
    ALL_TRAIN_CSVS = ["DADAD_Train_12112_0", 
                    "DADAD_Train_12217_1", 
                    "DADAD_Train_12319_2", 
                    "DADAD_Train_12430_3",
                    "DADAD_Train_12537_4",
                    "DADAD_Train_12643_5",
                    "DADAD_Train_12754_6",
                    "DADAD_Train_12865_7",
                    "DADAD_Train_12975_8",
                    "DADAD_Train_13082_9",
                    "DADAD_Train_13192_10"]

    TEST_CSV = "DAD_Test_612.csv"
    MODEL_PARAMS = {
        "GC_LAYERS": [512, 512, 512, 512], 
        "DENSE_LAYERS": [128, 128, 128],
        "DROPOUT": 0.01,
        "LEARNING_RATE": 0.001,
        "BATCH_SIZE": 10,
        "MODEL_DIR": "DADAD10_Models/"
    }
    FINAL_CSV_STORE_NAME = "DADAD10_Final_Model_Performances_Test_DAD.csv"
    TRAIN_DIR = "DADAD10_DATA/Train/"
    TEST_DIR = "DADAD10_DATA/Test"

# fixed test set
test_set = loader.create_dataset(TEST_CSV, TEST_DIR)
print(f"Currently featurizing: {TEST_CSV}")

# average of 5 
final_results_data_store = {
    "st_train_mae" : [],
    "st_train_r2": [],
    "st_train_rmse" : [],
    "bg_train_mae" : [],
    "bg_train_r2": [],
    "bg_train_rmse" : [],
    "st_test_mae" : [],
    "st_test_r2": [],
    "st_test_rmse" : [],
    "bg_test_mae" : [],
    "bg_test_r2": [],
    "bg_test_rmse" : []
}


MAE_metric = [dc.metrics.Metric(dc.metrics.mae_score)]
R2_metric = [dc.metrics.Metric(dc.metrics.pearson_r2_score)]
RMSE_metric = [dc.metrics.Metric(dc.metrics.rms_score)]

NUM_EPOCHS = 200

if IF_MULTIPLE == "1":
    NUM_REPEATS = 5
    print("RUNNING 5 ROUNDS AND AVERAGE OF 5 FOR EVERY MODEL")
else:
    NUM_REPEATS = 1

for train_i in range(len(ALL_TRAIN_CSVS)):
    current_csv = ALL_TRAIN_CSVS[train_i]
    print(f"Currently featurizing: {current_csv}.csv")
    CURRENT_TRAIN_DIR = TRAIN_DIR + f"{train_i}"
    train_set = loader.create_dataset(f"{current_csv}.csv", CURRENT_TRAIN_DIR)
    print("")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Current train test size:", len(train_set), len(test_set))

    model_results_data_store = {
        "st_train_mae" : [],
        "st_train_r2": [],
        "st_train_rmse" : [],
        "bg_train_mae" : [],
        "bg_train_r2": [],
        "bg_train_rmse" : [],
        "st_test_mae" : [],
        "st_test_r2": [],
        "st_test_rmse" : [],
        "bg_test_mae" : [],
        "bg_test_r2": [],
        "bg_test_rmse" : []
    }

    for repeat_i in range(NUM_REPEATS):
        print(f"Running iteration {repeat_i}")
    
        results_data_store = {
            "st_train_mae" : [],
            "st_train_r2": [],
            "st_train_rmse" : [],
            "bg_train_mae" : [],
            "bg_train_r2": [],
            "bg_train_rmse" : [],
            "st_test_mae" : [],
            "st_test_r2": [],
            "st_test_rmse" : [],
            "bg_test_mae" : [],
            "bg_test_r2": [],
            "bg_test_rmse" : []
        }
        predictions_data_store = {
            "predicted_st": [],
            "actual_st": [],
            "predicted_bg": [],
            "actual_bg": []
        }

        CURRENT_MODEL_DIR = MODEL_PARAMS["MODEL_DIR"] + f"{train_i}_{repeat_i}" 
        model = GraphConvModel(n_tasks = len(current_tasks),
                                graph_conv_layers = MODEL_PARAMS["GC_LAYERS"],
                                dense_layers = MODEL_PARAMS["DENSE_LAYERS"],
                                dropout = MODEL_PARAMS["DROPOUT"],
                                learning_rate = MODEL_PARAMS["LEARNING_RATE"],
                                batch_size = MODEL_PARAMS["BATCH_SIZE"],
                                uncertainty = False,
                                mode = 'regression',
                                model_dir = CURRENT_MODEL_DIR)
    
        for i in range(NUM_EPOCHS):
            model.fit(train_set, nb_epoch = 1)
            test_mae = model.evaluate(test_set, MAE_metric, [], per_task_metrics = True)[1]["mae_score"]
            test_r2 = model.evaluate(test_set, R2_metric, [], per_task_metrics = True)[1]["pearson_r2_score"]
            test_rmse = model.evaluate(test_set, RMSE_metric, [], per_task_metrics = True)[1]["rms_score"]
            train_mae = model.evaluate(train_set, MAE_metric, [], per_task_metrics = True)[1]["mae_score"]
            train_r2 = model.evaluate(train_set, R2_metric, [], per_task_metrics = True)[1]["pearson_r2_score"]
            train_rmse = model.evaluate(train_set, RMSE_metric, [], per_task_metrics = True)[1]["rms_score"]
            print(f"Epoch {i} with MAE: ", test_mae, " / R2: ", test_r2, " / RMSE: ", test_rmse)
            results_data_store["st_train_mae"].append(train_mae[0])
            results_data_store["st_train_r2"].append(train_r2[0])
            results_data_store["st_train_rmse"].append(train_rmse[0])
            results_data_store["bg_train_mae"].append(train_mae[1])
            results_data_store["bg_train_r2"].append(train_r2[1])
            results_data_store["bg_train_rmse"].append(train_rmse[1])
            results_data_store["st_test_mae"].append(test_mae[0])
            results_data_store["st_test_r2"].append(test_r2[0])
            results_data_store["st_test_rmse"].append(test_rmse[0])
            results_data_store["bg_test_mae"].append(test_mae[1])
            results_data_store["bg_test_r2"].append(test_r2[1])
            results_data_store["bg_test_rmse"].append(test_rmse[1])
    
        # store the final model performance metrics 
        test_mae = model.evaluate(test_set, MAE_metric, [], per_task_metrics = True)[1]["mae_score"]
        test_r2 = model.evaluate(test_set, R2_metric, [], per_task_metrics = True)[1]["pearson_r2_score"]
        test_rmse = model.evaluate(test_set, RMSE_metric, [], per_task_metrics = True)[1]["rms_score"]
        train_mae = model.evaluate(train_set, MAE_metric, [], per_task_metrics = True)[1]["mae_score"]
        train_r2 = model.evaluate(train_set, R2_metric, [], per_task_metrics = True)[1]["pearson_r2_score"]
        train_rmse = model.evaluate(train_set, RMSE_metric, [], per_task_metrics = True)[1]["rms_score"]
        print("")
        print(f"Final MAE: ", test_mae, " / R2: ", test_r2, " / RMSE: ", test_rmse)
        model_results_data_store["st_train_mae"].append(train_mae[0])
        model_results_data_store["st_train_r2"].append(train_r2[0])
        model_results_data_store["st_train_rmse"].append(train_rmse[0])
        model_results_data_store["bg_train_mae"].append(train_mae[1])
        model_results_data_store["bg_train_r2"].append(train_r2[1])
        model_results_data_store["bg_train_rmse"].append(train_rmse[1])
        model_results_data_store["st_test_mae"].append(test_mae[0])
        model_results_data_store["st_test_r2"].append(test_r2[0])
        model_results_data_store["st_test_rmse"].append(test_rmse[0])
        model_results_data_store["bg_test_mae"].append(test_mae[1])
        model_results_data_store["bg_test_r2"].append(test_r2[1])
        model_results_data_store["bg_test_rmse"].append(test_rmse[1])

        # make predictions on test set and compare with actual 
        pred = model.predict(test_set)
        st_pred = np.asarray([pred[xi][0] for xi in range(len(pred))])
        bg_pred = np.asarray([pred[xi][1] for xi in range(len(pred))])
        st_actual = np.asarray([test_set.y[xi][0] for xi in range(len(test_set.y))]) 
        bg_actual = np.asarray([test_set.y[xi][1] for xi in range(len(test_set.y))]) 

        predictions_data_store["predicted_st"] = st_pred
        predictions_data_store["actual_st"] = st_actual
        predictions_data_store["predicted_bg"] = bg_pred
        predictions_data_store["actual_bg"] = bg_actual

        # save the 2 model store csvs 
        if STRUCTURE_TYPE == "DA":
            pd.DataFrame.from_dict(results_data_store).to_csv(f"DA_Model_{train_i}_Performances_{repeat_i}.csv")
            pd.DataFrame.from_dict(predictions_data_store).to_csv(f"DA_Model_{train_i}_Predictions_{repeat_i}.csv")
        elif STRUCTURE_TYPE == "DADAD10":
            pd.DataFrame.from_dict(results_data_store).to_csv(f"DADAD10_Model_{train_i}_Performances_{repeat_i}.csv")
            pd.DataFrame.from_dict(predictions_data_store).to_csv(f"DADAD10_Model_{train_i}_Predictions_{repeat_i}.csv")

        del model
        print("")
        print("-----------------------------------")
        print(f"MODEL {train_i}_{repeat_i} DONE!")   
        print("-----------------------------------")
        print("")

    del train_set

    REPEATED_MODEL_CSV_STORE_NAME = FINAL_CSV_STORE_NAME + f"_{train_i}"
    pd.DataFrame.from_dict(model_results_data_store).to_csv(REPEATED_MODEL_CSV_STORE_NAME + ".csv")

    # save the mean of 5 scores to final_results_data_store
    for key in model_results_data_store.keys():
        assert len(model_results_data_store[key]) == NUM_REPEATS
        final_results_data_store[key].append(np.mean(model_results_data_store[key]))

    print("")
    print("#######################################")
    print(f"MODEL {train_i} ALL REPEATS DONE!")   
    print("#######################################")
    print("")

FINAL_CSV_STORE_NAME = FINAL_CSV_STORE_NAME + "_MEAN"
pd.DataFrame.from_dict(final_results_data_store).to_csv(FINAL_CSV_STORE_NAME + ".csv")
print("")
print("#######################################")
print("ALL MODELS DONE!")
