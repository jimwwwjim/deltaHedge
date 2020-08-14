

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.family'] = 'serif'
#
# Parameters
#
S0 = 8500.0  # initial stock value
K = 8500.0  # strike price
T = 1.0  # time to maturity
r = 0.03  # risk-less short rate
sigma = 0.28  # volatility of stock value
M = 300  # number of time steps
I = 50000  # number of paths

D = 5  # number of regression functions

def DBA_put_value(S0,M,I):
    rand = np.random.standard_normal((M+1,I))
    dt = T / M
    df = math.exp(-r*dt)
    S = np.zeros((M+1,I),dtype = np.float)
    S[0] = S0
    for t in range(1,M+1,1):
        S[t] = S[t-1]*(np.exp((r-sigma**2/2)*dt+sigma+math.sqrt(dt)*rand[t]))
    S_ = S
    for i in range(len(S)):
        S_[i] = np.where(S_[i]>S0,S0,S_[i])
    h = np.maximum(K-S_,0) #inner values
    V = np.maximum(K-S_,0) #value matrix
    C  = np.zeros((M+1,I),dtype=np.float)
    rg = np.zeros((M+1,D+1),dtype=np.float)
    for t in range(M-1,0,-1):
        rg[t]=np.polyfit(S[t],V[t+1]*df,D)
        C[t]=np.polyval(rg[t],S[t])
        C[t]=np.where(C[t]<0,0.,C[t])
        V