import numpy as np
import pandas as pd
import re
import json
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import precision_score, recall_score, f1_score

# UCI dataset list
UCI_LIST_FN = "./UCI_data_summary.csv"
dfn_list = pd.read_csv(UCI_LIST_FN, index_col=0, header=0)["file"].values.tolist()

def obtain_cutpoints(cp_fn):
    cp = []
    with open(cp_fn) as fin:
        while True:
            line = fin.readline()
            if line == '':
                break
            elif line[0:4] == "----":
                continue
            else:
                res = re.findall("dimension (.*?) \((.*?) bins\)", line)
                n_bins = int(res[0][1])
                cp_d = []
                for r in range(n_bins):
                    cp_d.append(eval(fin.readline().strip()))
                cp.append(cp_d)
    return cp

def discretize(X:pd.DataFrame, cp:list):
    values = X.values
    valuesF = values[:, 0:len(cp)]
    valuesC = values[:, len(cp):]
    
    valuesD = np.zeros_like(valuesF)
    for c in range(valuesF.shape[1]):
        bins = []
        for iv, v in enumerate(cp[c]):
            if iv == 0:
                bins.append([-np.inf, v])
            else:
                bins.append([cp[c][iv-1], v])
        bins.append([cp[c][-1], np.inf])
        for ix in range(valuesF.shape[0]):
            x = valuesF[ix][c]
            for iv, v in enumerate(bins):
                if x>v[0] and x<=v[1]:
                    valuesD[ix][c] = iv
                    break
    valuesD = np.concatenate((valuesD, valuesC), axis=1)
    XD = pd.DataFrame(valuesD, index=X.index, columns=X.columns)
    
    
    return XD


classifiers = { 
    'Ada': AdaBoostClassifier(random_state=20201010),
    'RF': RandomForestClassifier(n_jobs=5, random_state=20201010),
    'LinearSVC': LinearSVC(random_state=0, tol=1e-4, max_iter=2000, dual='auto'),
    'SVC': SVC(random_state=0, tol=1e-4, max_iter=2000),
    'LDA': LinearDiscriminantAnalysis(solver='svd') 
}

N_FOLDS = 10
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
        # read train data
        Xtrain = pd.read_csv(f"./data/CV_{cv_i+1}/train_" + dfn, sep=";", header=None, index_col=None)
        Xtrain = Xtrain.iloc[:, 0:Xtrain.shape[1]-1]
        # obtain cut points
        cp = obtain_cutpoints(f"./data/CV_{cv_i+1}/train_cp_" + dfn)
        XtrainD = discretize(Xtrain, cp)
        # read train labels
        Ytrain = pd.read_csv(f"./data/CV_{cv_i+1}/train_" + dfn.replace("_clf.csv", "_clf_labels.csv"), 
                             sep=";", header=None, index_col=None)
        if Ytrain.shape[0] > 1:
            print(f"skip {dfn} due to not classification")
        Ytrain = Ytrain.iloc[0,:].values.T
        Yunique, Ytrain = np.unique(Ytrain, return_inverse=True)
        
        
        # read test data
        Xtest = pd.read_csv(f"./data/CV_{cv_i+1}/test_" + dfn, sep=";", header=None, index_col=None)
        XtestD = discretize(Xtest, cp)
        # read test labels
        Ytest = pd.read_csv(f"./data/CV_{cv_i+1}/test_" + dfn.replace("_clf.csv", "_clf_labels.csv"), 
                             sep=";", header=None, index_col=None)
        Ytest = Ytest.iloc[0,:].values.T 
        Ytest = [ Yunique.tolist().index(v)  for v in Ytest]
        
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
        
# for dfn in dfn_list:
#     if len(precision[dfn])<10 or len(recall[dfn])<10 or len(f1[dfn])<10:
#         continue
#     else:
#         print(f"f1={np.mean(f1[dfn]):.3f}/+-{np.std(f1[dfn]):.3f}\tprec={np.mean(precision[dfn]):.3f}/+-{np.std(precision[dfn]):.3f}\trecall={np.mean(recall[dfn]):.3f}/+-{np.std(recall[dfn]):.3f}\t{dfn}")

with open("IPD_classification_results.json", "w") as fout:
    json.dump({"f1": f1, "precision": precision, "recall": recall}, fout) 


