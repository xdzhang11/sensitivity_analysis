from scipy.stats import norm, rayleigh, weibull_min
from numpy.random import multivariate_normal, rand
import pandas as pd
import numpy as np
from numpy.linalg import multi_dot, inv


#%% wind turbine example

def trans_wt(M, d):
    X_v = pd.DataFrame(data=np.zeros((len(M), d)), columns=['wsp', 'ti', 'cl', 'bladeIx', 'towerIx'])
    v_ave = 8.5
    r_scale = np.sqrt(2 / np.pi) * v_ave
    X_v.wsp = rayleigh.ppf(M[:, 0], loc=0, scale=r_scale)
    X_v.ti = weibull_min.ppf(M[:, 1], 2.5)
    X_v.cl = norm.ppf(M[:, 2], loc=1, scale=0.05)
    X_v.bladeIx = norm.ppf(M[:, 3], loc=1, scale=0.05)
    X_v.towerIx = norm.ppf(M[:, 4], loc=1, scale=0.05)
    return X_v


def X_j_wt(Nv, d, rho):
    A = rand(Nv, d)
    A[:, 0:2] = norm.cdf(multivariate_normal([0, 0], [[1, rho], [rho, 1]], Nv))
    x = trans_wt(A, d)
    return x


def X_dep_wt(pi, s_index, No, Ni, d, j, rho):
    x_t = np.zeros((No * Ni, d))
    Sj = pi[:j + 1]  # set of the 1st-jth elements in pi
    Sjc = pi[j + 1:]  # set of the (j+1)th-kth elements in pi
    for m in range(No):  # loop through outer loop
        xjc = rand(1, len(Sjc)).flatten()

        if (0 in Sjc) & (1 in Sjc):
            loc_0 = np.where(Sjc == 0)[0][0]
            loc_1 = np.where(Sjc == 1)[0][0]
            a = multivariate_normal([0, 0], [[1, rho], [rho, 1]], 1).flatten()
            xn = norm.cdf(a)
            xjc[loc_0] = xn[0]
            xjc[loc_1] = xn[1]
        elif 0 in Sjc:
            loc_0 = np.where(Sjc == 0)[0][0]
            a = norm.rvs()
            xn = norm.cdf(a)
            xjc[loc_0] = xn
        elif 1 in Sjc:
            loc_1 = np.where(Sjc == 1)[0][0]
            a = norm.rvs()
            xn = norm.cdf(a)
            xjc[loc_1] = xn

        for n in range(Ni):  # loop through inner loop
            xj = rand(1, len(Sj)).flatten()
            if (0 in Sjc) & (1 in Sjc):
                pass
            elif 0 in Sjc:
                loc_1 = np.where(Sj == 1)[0][0]
                b = norm.rvs(rho * a, np.sqrt(1 - rho ** 2))
                xn2 = norm.cdf(b)
                xj[loc_1] = xn2
            elif 1 in Sjc:
                loc_0 = np.where(Sj == 0)[0][0]
                b = norm.rvs(rho * a, np.sqrt(1 - rho ** 2))
                xn2 = norm.cdf(b)
                xj[loc_0] = xn2
            else:
                loc_0 = np.where(Sj == 0)[0][0]
                loc_1 = np.where(Sj == 1)[0][0]
                a = multivariate_normal([0, 0], [[1, rho], [rho, 1]], 1).flatten()
                xn2 = norm.cdf(a)
                xj[loc_0] = xn2[0]
                xj[loc_1] = xn2[1]

            x = np.concatenate([xj, xjc])
            x_t[m * Ni + n, :] = x[s_index]
        x_D = trans_wt(x_t, d)
    return x_D

