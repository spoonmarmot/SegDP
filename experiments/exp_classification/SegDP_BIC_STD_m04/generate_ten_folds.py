import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold, KFold

# generate 
N_FOLDS = 10
FOLD_FN_FMT = "CV"
UCI_CLF_FOLDER = '../../ori_data/rw_clf/'
# N_MAX_GROUPS = 20
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

def generate_SegDP_command(X:pd.DataFrame, dfn:str, folder:str, thd:str, whiten:int, prefix:str):
    
    FEATURE = folder + "/train_" + dfn
    # MIN_LEN = int(np.floor(X.shape[1] / N_MAX_GROUPS))
    MAX_K = int(np.floor(np.sqrt(X.shape[1]))) if X.shape[1] < 6000 else 70
    MIN_LEN = int(np.floor(X.shape[1] / MAX_K))
    TARGET_K = -1
    THD = thd
    OUTPUT_FILE = folder +  "/" + f"train_dsc_{THD}_w{whiten}_" + file
    SCALING = "standard"
    A_WS = 1
    
    command = f"{prefix}echo {folder}/{dfn} \n{prefix}../../../stratification_bydp/build/segdp -f {FEATURE} -g {MIN_LEN} -k {MAX_K} -t {TARGET_K} -m {THD} -s {SCALING} -a {A_WS} -o {OUTPUT_FILE} >> {folder}/{dfn.replace('.csv', '.log')} &\n"
    
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
segdp_commands = []
for root, dirs, files in os.walk(UCI_CLF_FOLDER):
    for file in files:
        try: 
            if file.endswith("_clf.csv"):
                
                # if "uci_075" in file:
                #     cmd_prefix = "# "
                # else:
                #     cmd_prefix = ""
                
                cmd_prefix = ""
                              
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
                # kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=20201010)
                kf.get_n_splits(data)
                for i, (train_index, test_index) in enumerate(kf.split(data, label.T.values[:,0])):
                    CV_FOLDER = f"./data/{FOLD_FN_FMT}_{i+1}"
                    X_train = data.iloc[train_index, :].T # !!!SegDP: row=>features col=>samples
                    X_train.to_csv('/'.join([CV_FOLDER, "train_" + file]))
                    # segdp_commands.append(generate_SegDP_command(X_train, file, CV_FOLDER, "CH", 0, cmd_prefix))
                    # segdp_commands.append(generate_SegDP_command(X_train, file, CV_FOLDER, "BICg", 0, cmd_prefix))
                    segdp_commands.append(generate_SegDP_command(X_train, file, CV_FOLDER, "0.4", 0, cmd_prefix))

                    Y_train = label.iloc[:, train_index]
                    Y_train.to_csv('/'.join([CV_FOLDER, "train_" + file.replace("_clf.csv", "_clf_labels.csv")]))

                    X_test = data.iloc[test_index, :].T # !!!SegDP: row=>features col=>samples
                    X_test.to_csv('/'.join([CV_FOLDER, "test_" + file]))

                    Y_test = label.iloc[:, test_index]
                    Y_test.to_csv('/'.join([CV_FOLDER, "test_" + file.replace("_clf.csv", "_clf_labels.csv")]))
                data_summary.append([file, data.shape[0], data.shape[1], data.columns.tolist().count(1), data.columns.tolist().count(0)])  
                print(f"{file} done")
            
        except Exception as e:
            print(f"skip {file} because {str(e)}")
            
pd.DataFrame(data_summary, columns=['file', 'samples', 'features', 'num_features', 'cate_features']).to_csv("UCI_data_summary.csv")

with open("discretize_UCI_by_SegDP.sh", "a") as fout:
    for ic, c in enumerate(segdp_commands):
        if ic>0: 
            if ("uci_075" not in c) and ic%5==0:
                fout.write("wait\n")
            elif ("uci_075" in c) and ic%10==0:
                fout.write("wait\n")
            else:
                pass
                 
        fout.write(c)
        fout.write("\n")
    
    fout.write("wait\n")

