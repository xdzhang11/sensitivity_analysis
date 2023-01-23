from scipy.stats import norm, rayleigh, weibull_min, halfnorm
from cmath import pi
from numpy.random import multivariate_normal, rand
import pandas as pd
import numpy as np
from numpy.linalg import multi_dot, inv

from m_usigma import rvs_j, rvs_u, rvs_s, rvs_su, rvs_us


def trans_us(M, d):
    x_v = pd.DataFrame(data=np.zeros((len(M), d)), columns=['wsp', 'sigma', 'cl', 'bladeIx', 'towerIx'])
    x_v.wsp = M[:, 0]
    x_v.sigma = M[:, 1]
    # half normal distribution parameters
    mu_hn = 1
    scale_hn = 0.05/np.sqrt(1-2/pi)
    x_v.cl = 2-halfnorm.ppf(M[:, 2], loc=mu_hn, scale=scale_hn)

    # x_v.cl = norm.ppf(M[:, 2], loc=1, scale=0.05)
    x_v.bladeIx = norm.ppf(M[:, 3], loc=1, scale=0.05)
    x_v.towerIx = norm.ppf(M[:, 4], loc=1, scale=0.05)
    return x_v


def X_j_us(Nv, d, rho):
    A = rand(Nv, d)
    u, sigma = rvs_j(Nv)
    A[:, 0] = u
    A[:, 1] = sigma
    x = trans_us(A, d)
    return x


def X_dep_us(pi, s_index, No, Ni, d, j, rho):
    x_t = np.zeros((No * Ni, d))
    Sj = pi[:j + 1]  # set of the 1st-jth elements in pi
    Sjc = pi[j + 1:]  # set of the (j+1)th-kth elements in pi
    for m in range(No):  # loop through outer loop
        xjc = rand(1, len(Sjc)).flatten()

        if (0 in Sjc) & (1 in Sjc):
            loc_0 = np.where(Sjc == 0)[0][0]
            loc_1 = np.where(Sjc == 1)[0][0]
            u, sigma = rvs_j(1)
            xjc[loc_0] = u
            xjc[loc_1] = sigma
        elif 0 in Sjc:
            loc_0 = np.where(Sjc == 0)[0][0]
            u = rvs_u(1)
            xjc[loc_0] = u
        elif 1 in Sjc:
            loc_1 = np.where(Sjc == 1)[0][0]
            sigma = rvs_s(1)
            xjc[loc_1] = sigma

        xj = rand(Ni, len(Sj))
        if (0 in Sjc) & (1 in Sjc):
            pass
        elif 0 in Sjc:
            loc_1 = np.where(Sj == 1)[0][0]
            sigma = rvs_su(Ni, u)
            xj[:, loc_1] = sigma
        elif 1 in Sjc:
            loc_0 = np.where(Sj == 0)[0][0]
            u = rvs_us(Ni, sigma)
            xj[:, loc_0] = u
        else:
            loc_0 = np.where(Sj == 0)[0][0]
            loc_1 = np.where(Sj == 1)[0][0]
            u, sigma = rvs_j(Ni)
            xj[:, loc_0] = u
            xj[:, loc_1] = sigma

        x = np.hstack([xj, np.tile(xjc, (Ni, 1))])
        x_t[m*Ni:(m+1)*Ni, :] = x[:, s_index]
    x_d = trans_us(x_t, d)

    return x_d





