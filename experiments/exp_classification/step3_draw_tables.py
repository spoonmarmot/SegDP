import numpy as np
import json
import pandas as pd
import re
from step0_const import *
from matplotlib import pyplot as plt
import matplotlib
font = {'size'   : 27}
matplotlib.rc('font', **font)



SEGDP_FOLDER = "./SegDP_BIC_STD_m04"

classifiers = ["RF", "Ada", "SVC"]

f1_list = []
for clf in classifiers:
    f1 = pd.read_csv(f"{SEGDP_FOLDER}/f1_table_{clf}.csv", index_col = 0)
    f1.sort_index(inplace=True)
    new_col = []
    for col in f1.columns:
        new_col.append([clf, col])
    f1.columns = pd.MultiIndex.from_tuples(new_col)
    f1.index = index
    
    
    f1_score = f1.replace(-np.inf, 0)
    f1_rank = f1_score.rank(axis=1, ascending=False, method="min")
    f1_score_meanrank = f1_rank.mean()
    f1_score_moderank = f1_rank.mode(axis=0)
    if len(f1_score_moderank.shape) > 1:
        f1_score_moderank = f1_score_moderank.iloc[0] 
    f1_score_bestcnt = (f1_rank == 1).sum()
    
    stat = pd.concat([f1_score_bestcnt, f1_score_meanrank, f1_score_moderank], axis=1).T
    stat.index = ["Best cnt.", "Mean Rank", "Mode Rank"]
    
    f1 = pd.concat([f1, stat], axis=0)
    f1_list.append(f1)
    print()

f1 = pd.concat(f1_list, axis=1)

float_cols = [c for c in f1.columns if pd.api.types.is_float_dtype(f1[c])]
sty = f1.style
for g in f1.columns.get_level_values(0).unique():
    sty = sty.highlight_max(

        axis=1,
        subset=(slice(None), pd.IndexSlice[g, :]),
        props="textbf:--rwrap;"   # LaTeX-style props (no convert_css needed)
    )
sty = sty.format("{:.3f}", subset=float_cols)
sty.to_latex('./clf_performance.tex', hrules=True)

print()

