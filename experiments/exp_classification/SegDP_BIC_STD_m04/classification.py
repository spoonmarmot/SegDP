import numpy as np
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import precision_score, recall_score, f1_score
import json

UCI_LIST_FN = "./UCI_data_summary.csv"
N_FOLDS = 10
AVERAGE = "macro"
DFN_PREFIX = "train_dsc_0.4_w0_"

# UCI dataset list
dfn_list = pd.read_csv(UCI_LIST_FN, index_col=0, header=0)["file"].values.tolist()

def obtain_segheads(sh_fn):
    sh = []
    with open(sh_fn) as fin:
        while True:
            line = fin.readline()
            if line[0:len("Total running time:")] == "Total running time:":
                break
            elif line[0:len("------")] == "------":
                line = fin.readline()
                line = fin.readline()
                line = fin.readline()
                res = re.findall("segment_heads: (.*?)$", line)
                sh.append([int(v) for v in res[0].strip().split(' ')])
            else:
                continue
    return sh

def discretize_train(X:pd.DataFrame, sh:list):
    values = X.values
    bins = []
    if values.shape[1] != len(sh):
        raise Exception("dimension unmatched")
    valuesD = np.zeros_like(values)
    
    for c in range(values.shape[1]):
        srt_values_c = np.sort(values[:, c], kind="stable")
        tmp_cp_real = np.sort(np.unique([ srt_values_c[i] for i in sh[c][0:]]))
        tmp_bin = []
        for iv, v in enumerate(tmp_cp_real):
            if iv == 0:
                tmp_bin.append([-np.inf, v])
            else:
                tmp_bin.append([tmp_cp_real[iv-1], v])
            tmp_bin.append([tmp_cp_real[-1], np.inf])
            
        
        bins.append(tmp_bin)
        for ix in range(values.shape[0]):
            x = values[ix][c]
            for iv, v in enumerate(tmp_bin):
                if (v[0]!= v[1] and x>=v[0] and x<v[1]) or (v[0]==v[1] and x==v[0]):
                    valuesD[ix][c] = iv
                    break 
    XD = pd.DataFrame(valuesD, index=X.index, columns=X.columns)
    
    return XD, bins

def _NN_index(subset:np.ndarray, target:np.ndarray):
    NN_dist = np.inf
    NN_index = -1
    for row in range(subset.shape[0]):
        v = subset[row]
        dist = np.linalg.norm(target - v)
        if dist < NN_dist:
            NN_dist = dist
            NN_index = row
    return NN_index 
    
def discretize_test(Xtest:pd.DataFrame, Xtrain:pd.DataFrame, XtrainD:pd.DataFrame):
    '''
        Xtest is the test data to be discretized
        Xtrain is the training data
        XtrainD is the discretized Xtrain
        each row is a sample and each column is a feature
        
        for each feature in Xtest
            for each value v in Xtest
                if have the same feature value in Xtrain
                    find the NN in Xtrain samples whose has the same feature value
                else
                    find v1 v2 in Xtrain that v1<v<v2 
                    assign the d of the closer one between v1 and v2 to this value
    '''
    
    test = Xtest.values
    train = Xtrain.values
    trainD = XtrainD.values
    
    XtestD = np.zeros_like(test)
    XtestD[:,:] = -100
    
    for col in range(test.shape[1]): 
        for row in range(test.shape[0]):
            v = test[row][col]
            if Xtest.columns[col] == 0:
                XtestD[row][col] = v
                continue
            
            flag = train[:, col] == v
            if np.sum(flag) > 0:
                train_subset = np.delete(train[flag, :], col, axis=1)
                test_target = np.delete(test[row], col)
                index_NN = _NN_index(train_subset, test_target)
                d = (trainD[flag, col])[index_NN]
            else:
                train_argsrt = np.argsort(train[:, col])
                train_sorted = train[train_argsrt, col]
                trainD_sorted = trainD[train_argsrt, col]
                
                d = -100
                for i, u in enumerate(train_sorted):
                    if v < u:
                        if i>0 and trainD_sorted[i-1] != trainD_sorted[i]:
                            d = trainD_sorted[i-1] if (v-train_sorted[i-1]) < (u-v) else trainD_sorted[i]
                        else:
                            d = trainD_sorted[i]
                        break
                if d == -100:
                    d = trainD_sorted[-1] 
            XtestD[row][col] = d     
    return XtestD

def discretize_test_bins(X:pd.DataFrame, bins:list):
    values = X.values
    if values.shape[1] != len(bins):
        raise Exception("dimension unmatched")
    valuesD = np.zeros_like(values)
    for c in range(values.shape[1]):
        for ix in range(values.shape[0]):
            x = values[ix][c]
            for iv, v in enumerate(bins[c]):
                if (v[0]!= v[1] and x>=v[0] and x<v[1]) or (v[0]==v[1] and x==v[0]):
                    valuesD[ix][c] = iv
                    break
    XD = pd.DataFrame(valuesD, index=X.index, columns=X.columns) 
    return XD

classifiers = {
    'Ada': AdaBoostClassifier(random_state=20201010),
    'RF': RandomForestClassifier(n_jobs=5, random_state=20201010),
    'LinearSVC': LinearSVC(random_state=0, tol=1e-4, max_iter=2000, dual='auto'),
    'SVC': SVC(random_state=0, tol=1e-4, max_iter=2000),
    'LDA': LinearDiscriminantAnalysis() 
}  

precision = {}
recall = {}
f1 = {}
# for each data
for dfn in dfn_list:
     
    precision[dfn] = []
    recall[dfn] = []
    f1[dfn] = []
    # for each fold
    for cv_i in range(N_FOLDS):
        try: 
            # obtain segment head
            sh = obtain_segheads(f"./data/CV_{cv_i+1}/" + DFN_PREFIX + dfn + ".dscinfo")
            
            # read train data and get bins
            Xtrain = pd.read_csv(f"./data/CV_{cv_i+1}/train_" + dfn, sep=",", header=0, index_col=0).T
            if 0 in Xtrain.columns:
                Xtrain = pd.concat((Xtrain[[1]], Xtrain[[0]]), axis=1)
            # XtrainD, bins = discretize_train(Xtrain, sh)
            XtrainD = pd.read_csv(f"./data/CV_{cv_i+1}/{DFN_PREFIX}" + dfn, sep=",", header=None, index_col=None).T

        
            # read train labels
            Ytrain = pd.read_csv(f"./data/CV_{cv_i+1}/train_" + dfn.replace("_clf.csv", "_clf_labels.csv"), 
                                 sep=",", header=0, index_col=[0, 1, 2])
            if Ytrain.shape[0] > 1:
                print(f"skip {dfn} due to not classification") 
            
            Ytrain = Ytrain.iloc[0,:].values.T

             
            # read test data
            Xtest = pd.read_csv(f"./data/CV_{cv_i+1}/test_" + dfn, sep=",", header=0, index_col=0).T
            if 0 in Xtest.columns:
                Xtest = pd.concat((Xtest[[1]], Xtest[[0]]), axis=1)
            XtestD = discretize_test(Xtest, Xtrain, XtrainD)
                
            # XtestD = discretize_test_bins(Xtest, bins)
            
            # read test labels
            Ytest = pd.read_csv(f"./data/CV_{cv_i+1}/test_" + dfn.replace("_clf.csv", "_clf_labels.csv"), 
                                 sep=",", header=0, index_col=[0, 1, 2])
            Ytest = Ytest.iloc[0,:].values.T 
            


        
            clf_prec = {}
            clf_recall = {}
            clf_f1 = {}
            for clf_name, clf in classifiers.items():
                try:
                    clf.fit(XtrainD, Ytrain)
                    Ypred = clf.predict(XtestD)

                    p = precision_score(Ytest, Ypred, average="macro", zero_division=0)
                    r = recall_score(Ytest, Ypred, average="macro", zero_division=0)
                    f = f1_score(Ytest, Ypred, average="macro", zero_division=0)
                    clf_prec[clf_name] = p
                    clf_recall[clf_name] = r
                    clf_f1[clf_name] = f
                    print(f"{dfn}, fold {cv_i+1}, classifier={clf_name}, prec={p}, recall={r}, f1={f}")
                except Exception as e:
                    print(f"Error for {dfn}, fold {cv_i+1}, classifier={clf_name}: {e}")
                    clf_prec[clf_name] = -np.inf
                    clf_recall[clf_name] = -np.inf
                    clf_f1[clf_name] = -np.inf            
            precision[dfn].append(clf_prec)
            recall[dfn].append(clf_recall)
            f1[dfn].append(clf_f1)
            
        except Exception as e:
            print(f"skip {dfn}, fold {cv_i}, due to:")
            raise e
            continue
        

 

# for dfn in dfn_list:
#     try:
#         if len(precision[dfn])<10 or len(recall[dfn])<10 or len(f1[dfn])<10:
#             continue
#         else:
#                 print(f"f1={np.mean(f1[dfn]):.3f}/+-{np.std(f1[dfn]):.3f}\tprec={np.mean(precision[dfn]):.3f}/+-{np.std(precision[dfn]):.3f}\trecall={np.mean(recall[dfn]):.3f}/+-{np.std(recall[dfn]):.3f}\t{dfn}")
#     except Exception as e:
#         continue
 
with open("SegDP_BICl_classification_results.json", "w") as fout:
    json.dump({"f1": f1, "precision": precision, "recall": recall}, fout) 
