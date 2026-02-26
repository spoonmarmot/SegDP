import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

DATA_FOLDER = "./data/"
STD = 1.0
CONFUSION_P = 0.05
# N: number of samples
N = 1000
# x: main feature to be discretized
# y1...yd: assistant features

def generate_X(d1, d2, d3, s1, s2, s3, n1=500, n2=200, n3=300):
    
    # return 18 * (np.random.rand(N) - 0.5)
    
    X1 = np.random.randn(n1) * s1 + d1
    X2 = np.random.randn(n2) * s2 + d2
    X3 = np.random.randn(n3) * s3 + d3
    return np.concatenate([X1, X2, X3])
    

# {DATA_FOLDER}/case 1: one assistant features, i.e., d=1
#   y1 =  2 * x + 6
# ##
# X = generate_X(-8, 1, 8, 3, 1, 2)
X = generate_X(-0, 0, 0, 4, 0, 0, 1000, 0, 0)
y1 = 1/2 * X + 6 + np.random.randn(*X.shape) * STD
pd.DataFrame([X, y1],
             index=[1, 1],
             columns=[f"sample#{c}" for c in range(N)]
             ).to_csv(f"{DATA_FOLDER}/case1.csv", float_format="%.15g")

# {DATA_FOLDER}/case 2: one assistant features, i.e., d=1
#   y1 =  -x^2 + 9
# ##
# X = generate_X(-8, 0, 6, 3, 1, 2)
X = generate_X(-0, 0, 0, 4, 0, 0, 1000, 0, 0)
y1 = - 0.05 * X * X + 9 + np.random.randn(*X.shape) * STD
pd.DataFrame([X, y1],
             index=[1, 1],
             columns=[f"sample#{c}" for c in range(N)]
             ).to_csv(f"{DATA_FOLDER}/case2.csv", float_format="%.15g")



# {DATA_FOLDER}/case 3: two assistant features, i.e., d=2
#   y1 =  2 * x + 6
#   y2 = - x * x + 9
# 
# ##
# X = generate_X(-8, 2, 8, 3, 2, 2)
X = generate_X(-6, 5, 0, 2, 3, 0, 500, 500, 0)
y1 = 1/4 * X + 6 + np.random.randn(*X.shape) * STD
y2 = - 0.05 * X * X + 10 + np.random.randn(*X.shape) * STD 
pd.DataFrame([X, y1, y2],
             index=[1, 1, 1],
             columns=[f"sample#{c}" for c in range(N)]
             ).to_csv(f"{DATA_FOLDER}/case3.csv", float_format="%.15g")


# {DATA_FOLDER}/case 4: one assistant features, i.e., d=1
#   y1 = x + 9 if -9 <= x < -3 
#      = x     if -3 <= x <  3 
#      = x - 9 if  3 <= x <  9 
# 
# ##
# X = generate_X(-8, 0, 6, 3, 1, 2)
X = generate_X(0, 0, 0, 4, 0, 0, 1000, 0, 0)
y1 = np.ones(X.shape)
y1[X<-3] = X[X<-3] + 12 
y1[(X>=-3) & (X<3)] = X[(X>=-3) & (X<3)]
y1[X>=3] = X[X>=3] - 12
y1 += np.random.randn(*X.shape) * STD 
pd.DataFrame([X, y1],
             index=[1, 1],
             columns=[f"sample#{c}" for c in range(N)]
             ).to_csv(f"{DATA_FOLDER}/case4.csv", float_format="%.15g")


# {DATA_FOLDER}/case 5: one assistant feature, i.e., d=1
#   y1 = 0  if x<=-3
#        1  if x>-3 and x<3
#        2  if x>=3
# ##
# X = generate_X(-10, 1, 8, 3, 1, 2)
X = generate_X(0, 0, 0, 4, 0, 0, 1000, 0, 0)
y1 = np.ones(X.shape)
for i in range(y1.shape[0]):
    r = np.random.rand()
    if X[i] < -3:
        if r < CONFUSION_P:
            y1[i] = 1
        elif r > 1-CONFUSION_P:
            y1[i] = 2
        else:
            y1[i] = 0
    elif X[i] > 3:
        if r < CONFUSION_P:
            y1[i] = 1
        elif r > 1-CONFUSION_P:
            y1[i] = 0
        else:
            y1[i] = 2
    else:
        if r < CONFUSION_P:
            y1[i] = 0
        elif r > 1-CONFUSION_P:
            y1[i] = 2
        else:
            y1[i] = 1
pd.DataFrame([X, y1],
            index=[1, 0],
            columns=[f"sample#{c}" for c in range(N)]
            ).to_csv(f"{DATA_FOLDER}/case5.csv", float_format="%.15g")           
            

# {DATA_FOLDER}/case 6: two assistant feature, i.e., d=2
#   y1 = 0  if x<=-3
#        1  if x>-3 and x<3
#        2  if x>=3
#
#   y2 = 0  if x<=0
#      = 1  if x> 0        
# ##
# X = generate_X(-10, 1, 8, 3, 1, 2)
X = generate_X(0, 0, 0, 4, 0, 0, 1000, 0, 0)
y1 = np.ones(X.shape)
for i in range(y1.shape[0]):
    r = np.random.rand()
    if X[i] < -3:
        if r < CONFUSION_P:
            y1[i] = 1
        elif r > 1-CONFUSION_P:
            y1[i] = 2
        else:
            y1[i] = 0
    elif X[i] > 3:
        if r < CONFUSION_P:
            y1[i] = 1
        elif r > 1-CONFUSION_P:
            y1[i] = 0
        else:
            y1[i] = 2
    else:
        if r < CONFUSION_P:
            y1[i] = 0
        elif r > 1-CONFUSION_P:
            y1[i] = 2
        else:
            y1[i] = 1

y2 = np.ones(X.shape)
for i in range(y2.shape[0]):
    r = np.random.rand()
    if X[i] < -3:
        if r < 2 * CONFUSION_P:
            y2[i] = 3
        else:
            y2[i] = 4
    else:
        if r < 2 * CONFUSION_P:
            y2[i] = 4
        else:
            y2[i] = 3

pd.DataFrame([X, y1, y2],
             index=[1, 0, 0],
             columns=[f"sample#{c}" for c in range(N)]
             ).to_csv(f"{DATA_FOLDER}/case6.csv", float_format="%.15g")


# {DATA_FOLDER}/case 7: two assistant feature, i.e., d=2
#   y1 = 2 * x + 6
#   y2 = 0  if x<=-3
#        1  if x>-3 and x<3
#        2  if x>=3
#
# ##
# X = generate_X(-10, 1, 8, 3, 1, 2)
X = generate_X(0, 0, 0, 4, 0, 0, 1000, 0, 0)
y1 = 1/2 * X + 6 + np.random.randn(*X.shape) * STD
y2 = np.ones(X.shape)
for i in range(y2.shape[0]):
    r = np.random.rand()
    if X[i] < -3:
        if r < CONFUSION_P:
            y2[i] = 2
        elif r > 1-CONFUSION_P:
            y2[i] = 1
        else:
            y2[i] = 0
    elif X[i] > 3:
        if r < CONFUSION_P:
            y2[i] = 0
        elif r > 1-CONFUSION_P:
            y2[i] = 2
        else:
            y2[i] = 1
    else:
        if r < CONFUSION_P:
            y2[i] = 0
        elif r > 1-CONFUSION_P:
            y2[i] = 1
        else:
            y2[i] = 2
pd.DataFrame([X, y1, y2],
             index=[1, 1, 0],
             columns=[f"sample#{c}" for c in range(N)]
             ).to_csv(f"{DATA_FOLDER}/case7.csv", float_format="%.15g")

# {DATA_FOLDER}/case 8: two assistant feature, i.e., d=2
#   y1 =  -x^2 + 9
#   y2 = 0  if x<=-3
#        1  if x>-3 and x<3
#        2  if x>=3
#
# ##

# X = generate_X(-10, 1, 8, 3, 1, 2)
X = generate_X(0, 0, 0, 4, 0, 0, 1000, 0, 0)
y1 = - 0.05 * X * X + 9 + np.random.randn(*X.shape) * STD 
y2 = np.ones(X.shape)
for i in range(y2.shape[0]):
    r = np.random.rand()
    if X[i] < -3:
        if r < 2 * CONFUSION_P:
            y2[i] = 1
        else:
            y2[i] = 2
    else:
        if r < 2 * CONFUSION_P:
            y2[i] = 2
        else:
            y2[i] = 1
    
    # if X[i] < -3:
    #     if r < CONFUSION_P:
    #         y2[i] = 2
    #     elif r > 1-CONFUSION_P:
    #         y2[i] = 1
    #     else:
    #         y2[i] = 0
    # elif X[i] > 1.5:
    #     if r < CONFUSION_P:
    #         y2[i] = 0
    #     elif r > 1-CONFUSION_P:
    #         y2[i] = 2
    #     else:
    #         y2[i] = 1
    # else:
    #     if r < CONFUSION_P:
    #         y2[i] = 0
    #     elif r > 1-CONFUSION_P:
    #         y2[i] = 1
    #     else:
    #         y2[i] = 2

pd.DataFrame([X, y1, y2],
             index=[1, 1, 0],
             columns=[f"sample#{c}" for c in range(N)]
             ).to_csv(f"{DATA_FOLDER}/case8.csv", float_format="%.15g")
