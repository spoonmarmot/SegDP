import numpy as np
import pandas as pd
import os
import re

index = [
    "017\_breast",
    "030\_contraceptive",
    "050\_image",
    "052\_ionosphere",
    "075\_musk",
    "078\_page",
    "081\_pen",
    "096\_spectf",
    "107\_waveform",
    "143\_statlog1",
    "145\_statlog2",
    "146\_statlog3",
    "147\_statlog4",
    "151\_connectionist",
    "159\_telescope",
    "176\_blood",
    "212\_vertebral",
    "257\_knowledge",
    "264\_eeg",
    "267\_banknote",
    "277\_surgery",
    "292\_customers",
    "372\_htru2",
    "468\_shoppers",
    "503\_hepatitis",
    "519\_heart",
    "529\_diabetes",
    "544\_obesity",
    "545\_rice",
    "563\_churn",
    "602\_drybean",
    "759\_glioma",
    "763\_mines"
]

dt = {}
for root, dirs, files in os.walk("./data/"):
    for fn in files:
        if ".dscinfo" not in fn:
            continue
        
        with open("./data/" + fn) as fin:
            line = fin.readline()
            while "True":
                if "Total running time:" in line:
                    break
                elif line == "":
                    break
                else:
                    line = fin.readline()
            time = re.findall("Total running time: (.*?) seconds$", line)[0]
            dt[fn[fn.index("uci_"):fn.index(".dscinfo")]] = eval(time)

data_summary = pd.read_csv("./UCI_data_summary.csv")

time_table = []
for idx in data_summary.index:
    
    f = data_summary.loc[idx, "file"]
    n = data_summary.loc[idx, "samples"]
    m = data_summary.loc[idx, "features"]     
    m1 = data_summary.loc[idx, "num_features"]     
    time_table.append([f, n, f"{m} ({m1})", dt[f]])

time_table = sorted(time_table, key=lambda x:x[0])
for idx in range(len(time_table)):
    time_table[idx][0] = index[idx]

# tables = [ 
#     time_table[0:11],
#     time_table[11:22],
#     time_table[22:],
# ]
tables = [ 
    time_table[0:17],
    time_table[17:] + [["", "", "", ""]],
]

df_tables = [
    pd.DataFrame(t, columns=["data", "$n$", "$m (m_1)$", "sec"])
    for t in tables 
]

pd.concat(df_tables, axis=1).to_latex("../../manuscript/table/rt.tex", index=False)


print()
