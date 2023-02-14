from scipy.stats import norm, rayleigh, weibull_min, halfnorm
from numpy.random import multivariate_normal, rand
import pandas as pd
import numpy as np
import json


fn_wb = "results/sigma_wb_pars.txt"
wb_pars = json.load(open(fn_wb))


#%% wind turbine example
def trans_wt(M, d):
    x_v = pd.DataFrame(data=np.zeros((len(M), d)), columns=['wsp', 'sigma', 'cl', 'bladeIx', 'towerIx'])
    v_ave = 8.5
    r_scale = np.sqrt(2 / np.pi) * v_ave
    x_v.wsp = rayleigh.ppf(M[:, 0], loc=0, scale=r_scale)
    x_v.sigma = weibull_min.ppf(M[:, 1], c=wb_pars['c'], loc=wb_pars['loc'], scale=wb_pars['scale'])
    mu_hn = 1
    scale_hn = 0.05/np.sqrt(1-2/np.pi)
    x_v.cl = 2-halfnorm.ppf(M[:, 2], loc=mu_hn, scale=scale_hn)
    x_v.bladeIx = norm.ppf(M[:, 3], loc=1, scale=0.05)
    x_v.towerIx = norm.ppf(M[:, 4], loc=1, scale=0.05)
    return x_v


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

        xj = rand(Ni, len(Sj))
        #xj = rand(1, len(Sj)).flatten()
        if (0 in Sjc) & (1 in Sjc):
            pass
        elif 0 in Sjc:
            loc_1 = np.where(Sj == 1)[0][0]
            b = norm.rvs(rho * a, np.sqrt(1 - rho ** 2), Ni)
            xn2 = norm.cdf(b)
            xj[:, loc_1] = xn2
        elif 1 in Sjc:
            loc_0 = np.where(Sj == 0)[0][0]
            b = norm.rvs(rho * a, np.sqrt(1 - rho ** 2), Ni)
            xn2 = norm.cdf(b)
            xj[:, loc_0] = xn2
        else:
            loc_0 = np.where(Sj == 0)[0][0]
            loc_1 = np.where(Sj == 1)[0][0]
            a = multivariate_normal([0, 0], [[1, rho], [rho, 1]], Ni)
            xn2 = norm.cdf(a)
            xj[:, loc_0] = xn2[:, 0]
            xj[:, loc_1] = xn2[:, 1]

        x = np.hstack([xj, np.tile(xjc, (Ni, 1))])
        x_t[m*Ni:(m+1)*Ni, :] = x[:, s_index]
    x_d = trans_wt(x_t, d)
    return x_d

