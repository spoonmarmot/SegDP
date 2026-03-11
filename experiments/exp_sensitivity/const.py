import numpy as np

W_LIST = ["0", "0.5", "1", "2"]
E_LIST = ["0", "0.2", "0.4", "0.6", "1"]
KG_LIST = [
    (None, lambda X:5),
    (None, lambda X:10),
    (None, lambda X:int(np.floor(np.sqrt(X.shape[1])) / 2)),
    (None, None),
    (lambda X:int(np.floor(np.sqrt(X.shape[1]) / 1.5)), lambda X:int(np.floor(np.sqrt(X.shape[1]) * 1.5)))
]
KG_LABELS = [
    r"5",
    r"10",
    r"$0.5\sqrt{n}$",
    r"$\sqrt{n}$",
    r"$1.5\sqrt{n}$",
    
    # "$\lambda=5$, $K_{max}=\sqrt{n}$",
    # "$\lambda=10$, $K_{max}=\sqrt{n}$",
    # "$\lambda=\sqrt{n}/2$, $K_{max}=\sqrt{n}$",
    # "$\lambda=\sqrt{n}$, $K_{max}=\sqrt{n}$",
    # "$\lambda=1.5\sqrt{n}$, $K_{max}=\sqrt{n}/1.5$",
]
UCI_LIST = [
    "uci_030",
    "uci_050",
    "uci_078",
    "uci_096",
    "uci_107",
    "uci_143",
    "uci_151",
    "uci_277",
    "uci_292",
    "uci_519",
    "uci_529",
    "uci_545",
    "uci_759",
]

UCI_LABEL = [
    "030_contraceptive",
    "050_image",
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

