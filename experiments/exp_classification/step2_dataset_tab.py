import numpy as np
import json
import pandas as pd
from step0_const import *

SEGDP_FOLDER = "./SegDP_BIC_STD_m04"

data_list = pd.read_csv(SEGDP_FOLDER + '/UCI_data_summary.csv', index_col=[0], header=0)   
data_list = data_list.sort_values("file")

data_list["file"] = index
data_list.columns = header

list_1 = data_list.iloc[0:17, :]
list_2 = data_list.iloc[17:, :]
empty_row = pd.DataFrame([["","","","", ""]], columns=list_2.columns)
list_2 = pd.concat([list_2, empty_row], ignore_index=True)

list_1.index = [i for i in range(list_1.shape[0])]
list_2.index = [i for i in range(list_2.shape[0])]

full_list = pd.concat([list_1, list_2], axis=1)
full_list.to_latex("../../manuscript/table/data_summary.tex", index=False)
print()