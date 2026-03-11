import numpy as np
import pandas as pd
from const import *
from matplotlib import pyplot as plt
import json
import matplotlib
font = {'size'   : 20}
matplotlib.rc('font', **font)
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

DFN_PREFIX = "train_dsc_"

LOCAL_UCI_LIST = [ 
    "uci_030",
    "uci_078",
    "uci_096",
    "uci_107",
    "uci_143",
    "uci_151",
    "uci_277",
    "uci_519",
    "uci_529",
    "uci_545",
    "uci_759",                  
]

LOCAL_UCI_LABEL = [
    "030_contraceptive",
    "078_page",
    "096_spectf",
    "107_waveform",
    "143_statlog1",
    "151_connectionist",
    "277_surgery",
    "292_customers",
    "519_heart",
    "529_diabetes",
    "545_rice",
    "759_glioma",
    
]


styles = [
    {'linestyle': '-',                   'marker': 'o'},
    {'linestyle': '--',                  'marker': 's'},
    {'linestyle': '-.',                  'marker': '^'},
    {'linestyle': ':',                   'marker': 'D'},
    {'linestyle': (0, (1, 1)),           'marker': 'v'},
    {'linestyle': (0, (5, 1)),           'marker': 'P'},
    {'linestyle': (0, (3, 1, 1, 1)),     'marker': 'X'},
    {'linestyle': (0, (5, 2, 1, 2)),     'marker': '*'},
    {'linestyle': (0, (2, 2)),           'marker': '<'},
    {'linestyle': (0, (7, 2)),           'marker': '>'},
    {'linestyle': (0, (4, 1, 1, 1, 1, 1)),'marker': 'h'},
    {'linestyle': (0, (6, 2, 2, 2)),     'marker': 'd'},
]

prefix_W = [ DFN_PREFIX + f"w{w}e0.4kgNone_" for w in W_LIST] 
prefix_E = [ DFN_PREFIX + f"w1e{e}kgNone_" for e in E_LIST]  
prefix_L = [ DFN_PREFIX + f"w1e0.4kg{kgi}_" for kgi in range(len(KG_LIST))]

with open("./SegDP_BICl_classification_results.json") as fin:
    clf_perf = json.load(fin)

# W
W = []
for dn in LOCAL_UCI_LIST:
    W_d = []
    for prefix in prefix_W:
        f1_all = clf_perf["f1"][prefix + dn]
        f1_mean = pd.DataFrame(f1_all).mean(axis=0)
        W_d.append(f1_mean)
    W_d = pd.concat(W_d, axis=1).T
    W.append(W_d)

# E
E = []
for dn in LOCAL_UCI_LIST:
    E_d = []
    for prefix in prefix_E:
        f1_all = clf_perf["f1"][prefix + dn]
        f1_mean = pd.DataFrame(f1_all).mean(axis=0)
        E_d.append(f1_mean)
    E_d = pd.concat(E_d, axis=1).T
    E.append(E_d)   

# L
L = []
for dn in LOCAL_UCI_LIST:
    L_d = []
    for prefix in prefix_L:
        f1_all = clf_perf["f1"][prefix + dn]
        f1_mean = pd.DataFrame(f1_all).mean(axis=0)
        L_d.append(f1_mean)
    L_d = pd.concat(L_d, axis=1).T
    L.append(L_d)   

CLF_LIST = ["RF", "Ada", "SVC"]

fig, axes = plt.subplots(nrows=3, ncols=3)
fig.set_size_inches(14,10)
for ic, clf in enumerate(CLF_LIST):
    
    
    # W
    x = [0, 1, 2, 3]
    for i, d in enumerate(LOCAL_UCI_LIST):
        y = W[i][clf]
        axes[0][ic].plot(x, y, label=LOCAL_UCI_LABEL[i], markevery=1, markersize=8, linewidth=3, **styles[i])
        print()
    axes[0][ic].set_ylim((0.25, 1.05))
    axes[0][ic].set_xticks(x)
    axes[0][ic].set_xticklabels(W_LIST)
    axes[0][ic].set_ylabel("F1" , fontsize = 16)
    axes[0][ic].set_xlabel("$w$", fontsize = 16)
    
        
    # E
    x = [0, 1, 2, 3, 4]
    for i, d in enumerate(LOCAL_UCI_LIST):
        y = E[i][clf]
        axes[1][ic].plot(x, y, label=LOCAL_UCI_LABEL[i], markevery=1, markersize=8, linewidth=3, **styles[i])
        print()
    axes[1][ic].set_ylim((0.42, 1.05))
    axes[1][ic].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axes[1][ic].set_xticks(x)
    axes[1][ic].set_xticklabels(E_LIST)
    axes[1][ic].set_ylabel("F1"        , fontsize = 16)
    axes[1][ic].set_xlabel("$\epsilon$", fontsize = 16)
    
    # L
    x = [0, 1, 2, 3, 4]
    for i, d in enumerate(LOCAL_UCI_LIST):
        y = L[i][clf]
        axes[2][ic].plot(x, y, label=LOCAL_UCI_LABEL[i], markevery=1, markersize=8, linewidth=3, **styles[i])
        print()
    axes[2][ic].set_ylim((0.42, 1.05))
    axes[2][ic].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axes[2][ic].set_xticks(x)
    axes[2][ic].set_xticklabels(KG_LABELS)
    axes[2][ic].set_ylabel("F1"       , fontsize = 16)
    axes[2][ic].set_xlabel("$\lambda$", fontsize = 16)
        
    if ic == 0:
        handles, labels = axes[2][ic].get_legend_handles_labels()
        fig.legend(handles, labels,  loc="center", fontsize=18,
                bbox_to_anchor=(0.0, -0.1, 1, 0.18), 
                ncol=4,
                columnspacing=1.2,
                handletextpad=1,
                handlelength=1.4,
                borderpad=0.2
        )

fig.text(0.13, 0.98, "Random Forest")    
fig.text(0.48, 0.98, "AdaBoost")    
fig.text(0.83, 0.98, "SVM")    
    
fig.tight_layout(rect=(0.02, 0.02, 1, 0.99))
# plt.show()
fig.savefig(f"../../manuscript/figure/sensitivity/sensitivity.eps" , bbox_inches="tight", pad_inches=0)
print()