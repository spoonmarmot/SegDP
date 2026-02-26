import numpy as np
import json
import pandas as pd
from step0_const import *

SEGDP_FOLDER = "./SegDP_BIC_STD_m04"
EFEW_FOLDER = SEGDP_FOLDER
IPD_FOLDER = "./IPD"

# SegDP
SEGDP_RES_FN = f"./{SEGDP_FOLDER}/SegDP_BICl_classification_results.json"
with open(SEGDP_RES_FN) as fin:
    segdp_res = json.load(fin)

# EW 
EW_RES_FN = f"./{EFEW_FOLDER}/EW_classification_results.json"
with open(EW_RES_FN) as fin:
    ew_res = json.load(fin)

# EF 
EF_RES_FN = f"./{EFEW_FOLDER}/EF_classification_results.json"
with open(EF_RES_FN) as fin:
    ef_res = json.load(fin)
# KM 
KM_RES_FN = f"./{EFEW_FOLDER}/KM_classification_results.json"
with open(KM_RES_FN) as fin:
    km_res = json.load(fin)

# IPD 
IPD_RES_FN = f"./{IPD_FOLDER}/IPD_classification_results.json"
with open(IPD_RES_FN) as fin:
    ipd_res = json.load(fin)


##########################################################

classifiers = [
    'Ada',
    'RF',
    'SVC',

]

method_col = [
    "SegDP", 
    "EW", 
    "EF",
    # "KM", 
    "IPD"
]

for clf in classifiers:

    f1_segdp_avg = {}
    for fn in segdp_res["f1"]:
        f1_segdp_avg[fn] = np.mean([f[clf] for f in segdp_res["f1"][fn]])


    f1_ew_avg = {}
    for fn in ew_res["f1"]:
        f1_ew_avg[fn] = np.mean([f[clf] for f in ew_res["f1"][fn]])

    f1_ef_avg = {}
    for fn in ef_res["f1"]:
        f1_ef_avg[fn] = np.mean([f[clf] for f in ef_res["f1"][fn]])

    f1_km_avg = {}
    for fn in km_res["f1"]:
        f1_km_avg[fn] = np.mean([f[clf] for f in km_res["f1"][fn]])

    f1_ipd_avg = {}
    for fn in ipd_res["f1"]:
        f1_ipd_avg[fn] = np.mean([f[clf] for f in ipd_res["f1"][fn]])

    fn_index = []
    val = []
    for fn in f1_segdp_avg:
        try:
            val.append([
                f1_segdp_avg[fn], 
                f1_ew_avg[fn], 
                f1_ef_avg[fn], 
                # f1_km_avg[fn], 
                f1_ipd_avg[fn]
                ])    
            fn_index.append(fn)
        except Exception as e:
            print(fn, e)
            continue 

    f1_score = pd.DataFrame(val, index=fn_index, columns=method_col)
    f1_score.index = index
    f1_score.to_csv(f"{SEGDP_FOLDER}/f1_table_{clf}.csv")
    
    f1_score = f1_score.replace(-np.inf, 0)
    f1_rank = f1_score.rank(axis=1, ascending=False, method="min")
    f1_score_meanrank = f1_rank.mean()
    f1_score_moderank = f1_rank.mode(axis=0)
    f1_score_bestcnt = (f1_rank == 1).sum()

    
    print(f"==================\n{clf}")
    
    print("------------------\nMean F1:")
    print(f1_score.mean())
    
    print("------------------\nBest count:")
    print(f1_score_bestcnt)

    print("------------------\nMean rank:")
    print(f1_score_meanrank)


    print("------------------\nMode rank:")
    print(f1_score_moderank)