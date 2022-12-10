from scipy.stats import qmc
import numpy as np
from numpy.random import rand
from scipy.stats import norm, halfnorm, rayleigh, weibull_min
import pandas as pd
import copy

def trans(M, n, d):
    v_ave = 8.5
    r_scale = np.sqrt(2 / np.pi) * v_ave
    X_v = pd.DataFrame(data=np.zeros((n - 1, d)), columns=['wsp', 'ti', 'cl', 'bladeIx', 'towerIx'])

    X_v.wsp = rayleigh.ppf(M[:, 0], loc=0, scale=r_scale)
    wb_shape = 0.27 * X_v.wsp + 1.4
    Iref = 0.12
    wb_scale = Iref * (0.75 * X_v.wsp + 3.3)
    sigma = weibull_min.ppf(M[:, 1], wb_shape, 0, wb_scale)
    ti = sigma / X_v.wsp
    X_v.ti = ti
    # #Fixed win paramters
    # X_v.wsp = 11.4
    # X_v.ti = 0.14

    # #half normal distribution parameters
    # mu_hn = 1
    # scale_hn = 0.05/np.sqrt(1-2/pi)
    # X_v.cl = 2-halfnorm.ppf(M[:,2], loc=mu_hn, scale=scale_hn)

    X_v.cl = norm.ppf(M[:, 2], loc=1, scale=0.05)
    X_v.bladeIx = norm.ppf(M[:, 3], loc=1, scale=0.05)
    X_v.towerIx = norm.ppf(M[:, 4], loc=1, scale=0.05)
    return X_v

def sobol(cost, d, n):
    sampler = qmc.Sobol(d=2 * d, scramble=False)
    AB = sampler.random_base2(m=int(np.log2(n)))

    AB = AB[1:n, :]
    # A = AB[:,0:d]
    # B = AB[:,d:2*d]
    A = rand(n - 1, d)
    B = rand(n - 1, d)
    R = rand(n - 1, d)

    X_A = trans(A, n, d)
    X_B = trans(B, n, d)
    X_R = trans(R, n, d)
    index = ((X_A.wsp > 3) & (X_A.wsp < 27)) & ((X_B.wsp > 3) & (X_B.wsp < 27)) & ((X_R.wsp > 3) & (X_R.wsp < 27))
    X_A = X_A[index].reset_index(drop=True)
    X_B = X_B[index].reset_index(drop=True)
    X_R = X_R[index].reset_index(drop=True)
    n = len(X_A)

    y_A = cost(X_A).reshape((n, 1))
    y_B = cost(X_B).reshape((1, n))
    Y = cost(X_R)
    varY = np.var(Y)

    S_i = np.zeros(d)
    S_Ti = np.zeros(d)
    for i in range(d):
        # for i in range(2,d): #for three parameters
        X_A_Bi = copy.deepcopy(X_A)
        X_A_Bi.iloc[:, i] = X_B.iloc[:, i]
        y_A_Bi = cost(X_A_Bi).reshape((n, 1))
        S_i[i] = np.dot(y_B, y_A_Bi - y_A) / n / varY
        S_Ti[i] = np.sum((y_A - y_A_Bi) ** 2) / 2 / n / varY

    return S_i, S_Ti