import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
font = {'size'   : 27}
matplotlib.rc('font', **font)

K = 8
ORI_DATA_FOLDER = "./data/"
DSC_DATA_FOLDER = "./output/"

markers = [
    'x', '+', 'v', '^', '1', 's', 'o', '.', 'd', '*'
]

colors = [
    'indigo', 'seagreen', 'black', 'yellow', 'pink'
]


# case 1-4

num_titles = [
    "Linear",
    "Quadratic",
    "Liner and Quadratic",
    "Simpson's Paradox"
]
for idx in range(4):
    fig, axes = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches((7, 7))
    # read orignal data
    ori_data = pd.read_csv(f"{ORI_DATA_FOLDER}/case{idx+1}.csv", index_col=0, header=0)
    # read discretized data
    dsc_data = pd.read_csv(f"{DSC_DATA_FOLDER}/case{idx+1}_dsc.csv", index_col=None, header=None)
    dsc_data.index = ori_data.index
    dsc_data.columns = ori_data.columns
    
    # find vertical lines
    ori_fea = ori_data.iloc[0, :].values
    dsc_fea = dsc_data.iloc[0, :].values
    uniq_vals = np.unique(dsc_fea)
    vlines = []
    for v in uniq_vals:
        vlines.append(np.max(ori_fea[dsc_fea == v]))
    vlines.remove(np.max(vlines))
    
    # draw plots
    x = ori_data.iloc[0, :]
    for i in range(1, ori_data.shape[0]):
        axes.scatter(x, ori_data.iloc[i, :], 25, marker=markers[i-1], color = colors[i-1], linewidth=2, label = f"assistant feature #{i}")
    axes.vlines(vlines, ymin = ori_data.iloc[1:, :].min(axis=None), ymax = ori_data.iloc[1:, :].max(axis=None), 
                colors='red', linestyle='dashed', zorder = 0, linewidth = 4)
    
    plt.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        left=True,         # ticks along the top edge are off
        labelbottom=False,   # labels along the bottom edge are off
        labelleft=False   # labels along the bottom edge are off
    ) 
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False) 
    
    axes.set_xlabel("target feature")
    axes.set_ylabel("assistant feature")
    
    axes.set_title(num_titles[idx], fontsize=30)
     
    if ori_data.shape[0] > 2:
        leg = axes.legend(markerscale=5., fontsize=22)
        for h in leg.legend_handles:
            if hasattr(h, "set_linewidth"):
                h.set_linewidth(3.5)
            if hasattr(h, "set_markeredgewidth"):
                h.set_markeredgewidth(3.5)    
    fig.tight_layout()
    fig.savefig(f"./synthetic_case_{idx+1}.eps", bbox_inches="tight", pad_inches=0)


cate_titles = [
    "One Categorical Asso. Fea.",
    "Two Categorical Asso. Fea.",
    "Categorical and Linear",
    "Categorical and Quadratic"
] 
for idx in range(4, 8):
    fig, axes = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches((7, 7))
    # read orignal data
    ori_data = pd.read_csv(f"{ORI_DATA_FOLDER}/case{idx+1}.csv", index_col=0, header=0)
    # read discretized data
    dsc_data = pd.read_csv(f"{DSC_DATA_FOLDER}/case{idx+1}_dsc.csv", index_col=None, header=None)
    dsc_data.index = ori_data.index
    dsc_data.columns = ori_data.columns
    
    # find vertical lines
    ori_fea = ori_data.iloc[0, :].values
    dsc_fea = dsc_data.iloc[0, :].values
    uniq_vals = np.unique(dsc_fea)
    vlines = []
    for v in uniq_vals:
        vlines.append(np.max(ori_fea[dsc_fea == v]))
    vlines.remove(np.max(vlines))
    
    # draw plots
    x = ori_data.iloc[0, :]
    
    if idx == 4:
        for c in [0, 1, 2]:
            xc = x[ori_data.iloc[1, :] == c]
            yc = [c for i in range(xc.shape[0])] 
            axes.scatter(xc, yc, 25, marker="|", color = "blue", linewidth=1, label = f"cate.-fea.={c}")
        axes.vlines(vlines, ymin = -0.15, ymax = ori_data.iloc[1:, :].max(axis=None)+0.05, 
                    colors='red', linestyle='dashed', zorder = 0, linewidth = 4)
        axes.set_ylim((-0.25, +2.25))    
        axes.set_xlabel("target feature")
        axes.set_ylabel("categorical feature")
        
    elif idx == 5:
        for i in range(1, ori_data.shape[0]):
            if i == 2:
                y = - ori_data.iloc[i, :] + 2
            else:
                y = ori_data.iloc[i, :] + 1
            y *= 0.5
            axes.scatter(x, y, 20, marker="|", color = colors[i-1], linewidth=1, label = f"assistant feature #{i}")
        axes.vlines(vlines, ymin = -1.5, ymax = ori_data.iloc[1:, :].max(axis=None), 
                    colors='red', linestyle='dashed', zorder = 0, linewidth = 4)
        # axes.hlines([0], xmin=x.min()-3, xmax=x.max()+3, colors='k', zorder = -1)
        axes.text(x.min()-4, 0.5, 'cate. fea. #1', rotation=90)
        axes.text(x.min()-4, -1.5, 'cate. fea. #2', rotation=90)
        axes.text(x.min(), -0.20, 'target feature')
        axes.spines["bottom"].set_position(("data", 0))
        axes.set_ylim((-1.5, +2))
    elif idx >= 6:
        axes.scatter(x, ori_data.iloc[1, :], 20, color = colors[1], linewidth=2, marker=markers[0], label = "numerical")
        axes.set_ylabel("numerical assistant feature") 
        axes.vlines(vlines, ymin = ori_data.iloc[1:, :].min(axis=None), ymax = ori_data.iloc[1:, :].max(axis=None),
                    colors='red', linestyle='dashed', zorder = 0, linewidth = 4)
        axes.set_xlabel("target feature") 
        
        twin_axes = axes.twinx()
        twin_axes.scatter(x, ori_data.iloc[2, :], 15, color = colors[0], linewidth=1, marker="|", label = "categorical") 
        twin_axes.set_ylabel("categorical assistant feature") 
        twin_axes.tick_params(
            axis='both',       # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=True,      # ticks along the bottom edge are off
            left=False,         # ticks along the top edge are off
            labelbottom=False,   # labels along the bottom edge are off
            labelleft=False,   # labels along the bottom edge are off
            labelright=False   # labels along the bottom edge are off
        ) 
        twin_axes.spines['top'].set_visible(False)
        twin_axes.spines['left'].set_visible(False)
        twin_axes.set_ylim((-0.25, 2.25))
        if idx == 6:
            leg = fig.legend(markerscale=4, fontsize=21, loc="center left", bbox_to_anchor=(0.45, 0.3))
        else:
            leg = fig.legend(markerscale=4, fontsize=21, loc="center left", bbox_to_anchor=(0.40, 0.3))
        for h in leg.legend_handles:
            if hasattr(h, "set_linewidth"):
                h.set_linewidth(3.5)
            if hasattr(h, "set_markeredgewidth"):
                h.set_markeredgewidth(3.5) 
            
        
        
     
    
    axes.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        left=True,         # ticks along the top edge are off
        labelbottom=False,   # labels along the bottom edge are off
        labelleft=False   # labels along the bottom edge are off
    ) 
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False) 


    
    axes.set_title(cate_titles[idx-4], fontsize=30)
     
    # if ori_data.shape[0] > 2:
    #     axes.legend(markerscale=5., )
        
    fig.tight_layout()
    fig.savefig(f"./synthetic_case_{idx+1}.eps" , bbox_inches="tight", pad_inches=0)


