#
# Generate sh commands to discretize UCI datasets using equal-freq and 
# equal-width. The list of datasets is the same with SegDP. For each
# dataset, traverse number of bins from 3 to max(20, sqrt(#samples)) and 
# keep the one with the best classification performance
# #
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
import json
import re
import warnings
warnings.filterwarnings('ignore')


SEGDP_FOLDER = "./"
# UCI dataset list
UCI_LIST_FN = f"{SEGDP_FOLDER}/UCI_data_summary.csv"
# folds
N_FOLDS = 10
# SegDP results folder


def _discretize_train_1D(X, n_bins, method):
    if method == "ef":
        Xsort = np.sort(np.unique(X))
        cp_list = [ Xsort[i * int(np.floor(len(Xsort)/n_bins))-1] for i in range(1, n_bins)]
    elif method == "km":
        X2d = X.reshape(X.shape[0], 1)
        km = KMeans(n_clusters=n_bins, random_state=20201010)
        labels = km.fit_predict(X2d)
        cp_list = [np.max(X[labels==l]) for l in np.unique(labels)][0:-1] 
    else:
        cp_list = np.linspace(np.min(X), np.max(X), n_bins, endpoint=False)[1:]
    
    cp_list = np.sort(np.unique(cp_list))
    
    Xd = []
    for i in range(len(X)):
        v = X[i]
        d = np.nan
        for j_cp, cp in enumerate(cp_list):
            if v <= cp: # cp_list is ascending 
                d = j_cp
                break
        if d is np.nan:
            d = len(cp_list)
        Xd.append(d)
    return Xd, cp_list

def discretize_train(X, n_bins, method):
    Xf = X[[1]].values
    Xc = X[[0]].values if 0 in X.columns else None
    
    Xd = []
    cp_list = []
    M = Xf.shape[1]
    for d in range(M):
        xd, cp = _discretize_train_1D(Xf[:, d], n_bins[d], method)
        Xd.append(xd)
        cp_list.append(cp)
    Xd = np.array(Xd).T
    if Xc is not None:
        Xd = np.concatenate((Xd, Xc), axis=1)
    return Xd, cp_list

def _discretize_test_1D(X, cp_list):    
    Xd = []
    for i in range(len(X)):
        v = X[i]
        d = np.nan
        for j_cp, cp in enumerate(cp_list):
            if v < cp: # cp_list is ascending 
                d = j_cp
                break
        if d is np.nan:
            d = len(cp_list)
        Xd.append(d)
    return Xd
    
def discretize_test(X, cp_list):
    Xf = X[[1]].values
    Xc = X[[0]].values if 0 in X.columns else None
    
    Xd = []
    M = Xf.shape[1]
    for d in range(M):
        xd= _discretize_test_1D(Xf[:, d], cp_list[d])
        Xd.append(xd)
    Xd = np.array(Xd).T
    if Xc is not None:
        Xd = np.concatenate((Xd, Xc), axis=1)
    return Xd


def get_SegDP_bins(dfn): 
    n_bins = []
    with open(dfn) as fin:
        while True:
            line = fin.readline()
            if line[0:len("Total running time:")] == "Total running time:":
                break
            elif line[0:len("------")] == "------":
                line = fin.readline()
                line = fin.readline()
                res = re.findall("score=(.*?), \|segments\|=(.*?)$", line)
                n_bins.append(int(eval(res[0][1])))
                line = fin.readline()
            else:
                continue
    return n_bins 
                
    
classifiers = {
    'Ada': AdaBoostClassifier(random_state=20201010),
    'RF': RandomForestClassifier(n_jobs=5, random_state=20201010),
    'LinearSVC': LinearSVC(random_state=0, tol=1e-4, max_iter=2000, dual='auto'),
    'SVC': SVC(random_state=0, tol=1e-4, max_iter=2000),
    'LDA': LinearDiscriminantAnalysis() 
}    

dfn_list = pd.read_csv(UCI_LIST_FN, index_col=0, header=0)["file"].values.tolist()
ew_precision = {}
ew_recall = {}
ew_f1 = {}
ef_precision = {}
ef_recall = {}
ef_f1 = {}
km_precision = {}
km_recall = {}
km_f1 = {}
for dfn in dfn_list: 
    print(dfn)

    for strategy in ["ew", 'ef']: 
        
        f1 = []
        prec = []
        recall = []

        for i in range(N_FOLDS):
            nb = get_SegDP_bins(f"{SEGDP_FOLDER}/data/CV_{i+1}/train_dsc_0.4_w0_{dfn}.dscinfo")

            # read train data and get bins
            Xtrain = pd.read_csv(f"{SEGDP_FOLDER}/data/CV_{i+1}/train_" + dfn, sep=",", header=0, index_col=0).T
            # read train labels
            Ytrain = pd.read_csv(f"{SEGDP_FOLDER}/data/CV_{i+1}/train_" + dfn.replace("_clf.csv", "_clf_labels.csv"), sep=",", header=0, index_col=[0, 1, 2])
            Ytrain = Ytrain.iloc[0,:].values.T
            # read test data
            Xtest = pd.read_csv(f"{SEGDP_FOLDER}/data/CV_{i+1}/test_" + dfn, sep=",", header=0, index_col=0).T
            # read test label
            Ytest = pd.read_csv(f"{SEGDP_FOLDER}/data/CV_{i+1}/test_" + dfn.replace("_clf.csv", "_clf_labels.csv"), sep=",", header=0, index_col=[0, 1, 2])
            Ytest = Ytest.iloc[0,:].values.T 

            XtrainD, cp_list = discretize_train(Xtrain, nb, strategy)
            XtestD = discretize_test(Xtest, cp_list)

 
            
            
            # # output to ../analysis_tmp
            # # Xtrain
            # pd.DataFrame(Xtrain).to_csv(f"../analysis_tmp/{strategy}_{dfn.replace('.csv', '')}_CV{i+1}_Xtrain.csv", header=None, index=None)            
            # # XtrainD
            # pd.DataFrame(XtrainD).to_csv(f"../analysis_tmp/{strategy}_{dfn.replace('.csv', '')}_CV{i+1}_XtrainD.csv", header=None, index=None)            
            # # Xtest
            # pd.DataFrame(Xtest).to_csv(f"../analysis_tmp/{strategy}_{dfn.replace('.csv', '')}_CV{i+1}_Xtest.csv", header=None, index=None)            
            # # XtestD
            # pd.DataFrame(XtestD).to_csv(f"../analysis_tmp/{strategy}_{dfn.replace('.csv', '')}_CV{i+1}_XtestD.csv", header=None, index=None) 
            # # Ytrain
            # pd.DataFrame(Ytrain).to_csv(f"../analysis_tmp/{strategy}_{dfn.replace('.csv', '')}_CV{i+1}_Ytrain.csv", header=None, index=None) 
            # # Ytest
            # pd.DataFrame(Ytest).to_csv(f"../analysis_tmp/{strategy}_{dfn.replace('.csv', '')}_CV{i+1}_Ytest.csv", header=None, index=None)             
            # # Ypred
            # pd.DataFrame(Ypred).to_csv(f"../analysis_tmp/{strategy}_{dfn.replace('.csv', '')}_CV{i+1}_Ypred.csv", header=None, index=None) 
            
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
                    print(f"{dfn}, fold {i+1}, classifier={clf_name}, prec={p}, recall={r}, f1={f}")
                except Exception as e:
                    print(f"Error for {dfn}, fold {i+1}, classifier={clf_name}: {e}")
                    clf_prec[clf_name] = -np.inf
                    clf_recall[clf_name] = -np.inf
                    clf_f1[clf_name] = -np.inf            
            prec.append(clf_prec)
            recall.append(clf_recall)
            f1.append(clf_f1)
         
        if strategy == "ew":
            ew_f1[dfn] = f1
            ew_recall[dfn] = recall
            ew_precision[dfn] = prec 
        elif strategy == "ef":
            ef_f1[dfn] = f1
            ef_recall[dfn] = recall
            ef_precision[dfn] = prec 
        else:
            km_f1[dfn] = f1
            km_recall[dfn] = recall
            km_precision[dfn] = prec 
            
        

with open("EW_classification_results.json", "w") as fout:
    json.dump({"f1": ew_f1, "precision": ew_precision, "recall": ew_recall}, fout) 
                 
with open("EF_classification_results.json", "w") as fout:
    json.dump({"f1": ef_f1, "precision": ef_precision, "recall": ef_recall}, fout) 

with open("KM_classification_results.json", "w") as fout:
    json.dump({"f1": km_f1, "precision": km_precision, "recall": km_recall}, fout) 
print()