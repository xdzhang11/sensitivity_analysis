#%% all variables
#%reset -f
import joblib
from joblib import Parallel, delayed
num_cores = joblib.cpu_count()
#from cmath import pi
from scipy.stats import norm, rayleigh, weibull_min
from matplotlib import style

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from numpy.random import multivariate_normal, rand
from itertools import permutations
from numpy.linalg import inv
import time

#y_pred = gbrm.predict(X_test)

gbrm = joblib.load("models/randomforests.joblib")
def cost(x):
    return gbrm.predict(x)


def trans(M):
    X_v = pd.DataFrame(data = np.zeros((len(M),d)), columns=['wsp', 'ti', 'cl', 'bladeIx', 'towerIx'])
    v_ave = 8.5
    r_scale = np.sqrt(2/np.pi)*v_ave
    X_v.wsp = rayleigh.ppf(M[:,0], loc=0, scale=r_scale)
    X_v.ti  = weibull_min.ppf(M[:,1], 2.5)
    X_v.cl = norm.ppf(M[:,2], loc=1, scale=0.05)
    X_v.bladeIx = norm.ppf(M[:,3], loc=1, scale=0.05)
    X_v.towerIx = norm.ppf(M[:,4], loc=1, scale=0.05)
    return X_v

#%% Shapley effects
d = 5       # dimension of inputs
Nv = 10000  # MC sample size to estimate var(Y)
Ni = 30      # sample size for inner loop
No = 30      # sample size for outer loop
m = 1000


t = time.time()

#%%

A = rand(Nv,d)
X_A = trans(A)
y =  cost(X_A)

EY = np.mean(y)
VarY = np.var(y)


# %%
rho = 0.5

def f_shapley(p):
    Sh = np.zeros(d)
    Sh2 = np.zeros(d)
    pi = np.random.permutation(d)
    s_index = np.argsort(pi)       # sorted 
            
    prevC = 0
    
    for j in range(d): # loop through dimension
        X_B = np.zeros((No*Ni,d))
        if j == d-1:
            pass
        else: 
            Sj = pi[:j+1]         # set of the 1st-jth elements in pi 
            Sjc = pi[j+1:]        # set of the (j+1)th-kth elements in pi
            for l in range(No): # loop through outer loop
                if (0 in Sjc) & (1 in Sjc):
                    loc_0 = np.where(Sjc == 0)[0][0]
                    loc_1 = np.where(Sjc == 1)[0][0]
                    a = multivariate_normal([0, 0], [[1, rho], [rho, 1]] , 1).flatten()
                    xn = norm.cdf(a)
                    xjc = rand(1,len(Sjc)).flatten()
                    xjc[loc_0] = xn[0]
                    xjc[loc_1] = xn[1]
                elif 0 in Sjc:
                    loc_0 = np.where(Sjc == 0)[0][0]
                    a = norm.rvs()
                    xn = norm.cdf(a)
                    xjc = rand(1,len(Sjc)).flatten()
                    xjc[loc_0] = xn
                elif 1 in Sjc:
                    loc_1 = np.where(Sjc == 1)[0][0]
                    a = norm.rvs()
                    xn = norm.cdf(a)
                    xjc = rand(1,len(Sjc)).flatten()
                    xjc[loc_1] = xn
                else:
                    xjc = rand(1,len(Sjc)).flatten()
          
                X_t = np.zeros((Ni,d))
                for h in range(Ni):     # loop through inner loop

                    if (0 in Sjc) & (1 in Sjc):
                        xj = rand(1,len(Sj)).flatten()
                    elif 0 in Sjc:
                        loc_1 = np.where(Sj == 1)[0][0]
                        b = norm.rvs(rho*a, np.sqrt(1-rho**2))
                        xn2 = norm.cdf(b)
                        xj = rand(1,len(Sj)).flatten()
                        xj[loc_1] = xn2

                    elif 1 in Sjc:
                        loc_0 = np.where(Sj == 0)[0][0]
                        b = norm.rvs(rho*a, np.sqrt(1-rho**2))
                        xn2 = norm.cdf(b)
                        xj = rand(1,len(Sj)).flatten()
                        xj[loc_0] = xn2
                
                    else:
                        loc_0 = np.where(Sj == 0)[0][0]
                        loc_1 = np.where(Sj == 1)[0][0]
                        a = multivariate_normal([0, 0], [[1, rho], [rho, 1]] , 1).flatten()
                        xn2 = norm.cdf(a)
                        xj = rand(1,len(Sj)).flatten()
                        xj[loc_0] = xn2[0]
                        xj[loc_1] = xn2[1]
                    
                    x = np.concatenate([xj,xjc])
                    X_t[h,:] = x[s_index]
                X_B[l*Ni:(l+1)*Ni,:] = X_t
            
            X_all = trans(X_B)
            y_all = cost(X_all)
            

        if j == d-1:
            c_hat = VarY
            delta = c_hat-prevC
        else:
            cVar = []           # conditional variance    
            for l in range(No): # loop through outer loop              
                c_Y = y_all[l*Ni:(l+1)*Ni] 
                cVar.append(np.var(c_Y))

            c_hat = np.mean(cVar)
            delta = c_hat-prevC
        
        Sh[pi[j]] = Sh[pi[j]] + delta
        Sh2[pi[j]] = Sh2[pi[j]] + delta**2
        prevC = c_hat       
    return Sh, Sh2


res = Parallel(n_jobs=num_cores)(delayed(f_shapley)(i) for i in range(m))

r=[item[0] for item in res]
r2=[item[1] for item in res]

Sh = np.sum(r, axis =0)
Sh2 = np.sum(r2, axis =0)

Sh = Sh/m/VarY
Sh2 = Sh2/m
Shapley = Sh
SEShapley = np.sqrt((Sh2 - Sh**2)/m)
VarY = VarY
EY = EY

elapsed = time.time() - t
elapsed


# %%
