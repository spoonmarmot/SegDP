import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold

# generate 
N_FOLDS = 10
FOLD_FN_FMT = "CV"
UCI_CLF_FOLDER = '../../ori_data/rw_clf/'
MAX_SAMPLES = np.inf

try:
    os.mkdir("./data/")
except Exception as e:
    print(e)

for i in range(N_FOLDS):
    try:
        os.mkdir(f"./data/{FOLD_FN_FMT}_{i+1}/")
    except Exception as e:
        print(e)


def generate_IPD_command(X:pd.DataFrame, dfn:str, folder):
    LOG_OUTPUT = folder + '/' + "train_dsc_" + dfn.replace(".csv", ".log")
    FILE_INPUT = folder + '/' + "train_" + dfn
    FILE_CP_OUTPUT = folder + '/' + "train_cp_" + dfn
    FILE_RUNTIME_OUTPUT = folder + '/' + "train_rt_" + dfn
    FILE_DATA_OUTPUT = folder + '/' + "train_dsc_" + dfn
    NUM_ROWS = X.shape[0]  
    NUM_MEASURE_COLS = X.iloc[:, 0:-1].columns.tolist().count(1) # !!!! cause X include the class label column in the end
    NUM_CAT_CONTEXT_COLS = X.iloc[:, 0:-1].columns.tolist().count(0)
    MAX_VAL = X.iloc[:, 0:X.shape[1]-1].max(axis=None)
    METHOD = 0
 
    command = f"echo {folder}/{dfn} \njava -jar ./bin/ipd.jar -FILE_INPUT {FILE_INPUT} -FILE_CP_OUTPUT {FILE_CP_OUTPUT} -FILE_RUNTIME_OUTPUT {FILE_RUNTIME_OUTPUT} -FILE_DATA_OUTPUT {FILE_DATA_OUTPUT} -NUM_ROWS {NUM_ROWS} -NUM_MEASURE_COLS {NUM_MEASURE_COLS} -NUM_CAT_CONTEXT_COLS {NUM_CAT_CONTEXT_COLS} -MAX_VAL {MAX_VAL} -METHOD {METHOD} >> {LOG_OUTPUT} &\n"
    
    return command
    
def check_catefea(data:pd.DataFrame):
    columns_to_keep = []
    for c in range(data.shape[1]):
        t = data.columns[c]
        if t == 1:
            columns_to_keep.append(c)
            continue
        unique_vals = np.unique(data.iloc[:, c])
        if unique_vals.shape[0] > 1 and unique_vals.shape[0] < data.shape[0] / 2:
            columns_to_keep.append(c) 
            val_to_code = {val: code for code, val in enumerate(unique_vals)}
            for i in range(data.shape[0]):
                data.iloc[i, c] = val_to_code[data.iloc[i, c]]
            
    return data.iloc[:, columns_to_keep]
    

def check_numfea(data:pd.DataFrame):
    new_columns = []
    for c in range(data.shape[1]):
        t = data.columns[c]
        if t == 0:
            new_columns.append(t)
            continue
        unique_vals = np.unique(data.iloc[:, c])
        if unique_vals.shape[0] <= 10:
            # replace the categorical feature with integer codes
            val_to_code = {val: code for code, val in enumerate(unique_vals)}
            for i in range(data.shape[0]):
                data.iloc[i, c] = val_to_code[data.iloc[i, c]]
            new_columns.append(0)
        else: 
            new_columns.append(1)
    data.columns = new_columns 
    return data 


data_summary = []
ipd_commands = []

for root, dirs, files in os.walk(UCI_CLF_FOLDER):
    for file in files:
        try: 
            if file.endswith("_clf.csv"):
                data = pd.read_csv("/".join([UCI_CLF_FOLDER, file]), index_col=0, header=0).T
                label = pd.read_csv("/".join([UCI_CLF_FOLDER, file.replace("_clf.csv", "_clf_labels.csv")]), index_col=[0, 1, 2], header=0) 
                if label.shape[0] > 1:
                    raise Exception("Multiple-labels for one sample")
                if np.unique(label, return_counts=True)[1].min() < N_FOLDS:
                    raise Exception("too few samples in some classes")
                if data.shape[0] >= MAX_SAMPLES:
                    raise Exception("too many samples")
                
                data = check_numfea(data)
                data = check_catefea(data)
                
                if np.sum(data.columns==1) == 0:
                    raise Exception("no numerical features left after checking")
                
                kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=20201010)
                kf.get_n_splits(data)
                for i, (train_index, test_index) in enumerate(kf.split(data, label.T.values[:, 0])):
                    CV_FOLDER = f"./data/{FOLD_FN_FMT}_{i+1}"
                    X_train = data.iloc[train_index, :] 
                    if 0 in X_train.columns:
                        X_train = pd.concat(
                            (
                                X_train[1],
                                X_train[0], 
                            ), 
                            axis=1
                        )
                    
                    X_train = pd.concat(
                        (
                            X_train,
                            pd.DataFrame(np.zeros((X_train.shape[0], 1)), index=X_train.index)
                        ), 
                        axis=1
                    )
                    X_train.to_csv('/'.join([CV_FOLDER, "train_" + file]), sep=';', index=False, header=False)
                    ipd_commands.append(generate_IPD_command(X_train, file, CV_FOLDER))

                    Y_train = label.iloc[:, train_index]
                    Y_train.to_csv('/'.join([CV_FOLDER, "train_" + file.replace("_clf.csv", "_clf_labels.csv")]), sep=';', index=False, header=False)

                    X_test = data.iloc[test_index, :]
                    if 0 in X_test.columns:
                        X_test = pd.concat(
                            (
                                X_test[1],
                                X_test[0], 
                            ), 
                            axis=1
                        )
                    X_test.to_csv('/'.join([CV_FOLDER, "test_" + file]), sep=';', index=False, header=False)

                    Y_test = label.iloc[:, test_index]
                    Y_test.to_csv('/'.join([CV_FOLDER, "test_" + file.replace("_clf.csv", "_clf_labels.csv")]), sep=';', index=False, header=False)

                data_summary.append([file, data.shape[0], data.shape[1]]) 
                # ipd_commands.append("wait\n")
                print(f"{file} done")
        except Exception as e:
            print(f"skip {file} because {str(e)}")

pd.DataFrame(data_summary, columns=['file', 'samples', 'features']).to_csv("UCI_data_summary.csv")

with open("discretize_UCI_by_IPD.sh", "a") as fout:
    for ic, c in enumerate(ipd_commands):
        if ic > 0 and ic%2==0:
            fout.write("wait \n")
        fout.write(c)
        fout.write("\n")
