from scipy.stats import norm, rayleigh, weibull_min
from numpy.random import multivariate_normal, rand
import pandas as pd
import numpy as np
from numpy.linalg import multi_dot, inv

from m_usigma import rvs_j, rvs_u, rvs_s, rvs_su, rvs_us


def trans_us(M, d):
    X_v = pd.DataFrame(data=np.zeros((len(M), d)), columns=['wsp', 'ti', 'cl', 'bladeIx', 'towerIx'])
    X_v.wsp = M[:, 0]

    X_v.ti = M[:, 1]/M[:, 0]
    X_v.cl = norm.ppf(M[:, 2], loc=1, scale=0.05)
    X_v.bladeIx = norm.ppf(M[:, 3], loc=1, scale=0.05)
    X_v.towerIx = norm.ppf(M[:, 4], loc=1, scale=0.05)
    return X_v

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


#%%

def X_j_a(Nv, d, rho):
    sigma_i = [1, 1, 2]
    Rho = [[1, 0, 0],
           [0, 1, rho],
           [0, rho, 1]]
    mu = np.array([0, 0, 0])
    sigma = np.zeros((d, d))
    for k in range(d):
        for j in range(d):
            sigma[k][j] = sigma_i[k] * sigma_i[j] * Rho[k][j]
    x = multivariate_normal(mu, sigma, Nv)
    return x


def X_dep_a(pi, s_index, No, Ni, d, j, rho):
    sigma_i = [1, 1, 2]
    Rho = [[1, 0, 0],
           [0, 1, rho],
           [0, rho, 1]]
    mu = np.array([0, 0, 0])

    mu_p = mu[pi]
    sigma_p = np.zeros((d, d))
    for t in range(d):
        for s in range(d):
            sigma_p[t][s] = sigma_i[pi[t]] * sigma_i[pi[s]] * Rho[pi[t]][pi[s]]

    x_t = np.zeros((No * Ni, d))
    # Sj = pi[:j+1]         # set of the 1st-jth elements in pi
    # Sjc = pi[j+1:]        # set of the (j+1)th-kth elements in pi
    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
    # https://stats.stackexchange.com/questions/348941/general-conditional-distributions-for-multivariate-gaussian-mixtures
    mu_1 = mu_p[:j + 1]
    mu_2 = mu_p[j + 1:]
    sigma_11 = sigma_p[:j + 1, :j + 1]
    sigma_12 = sigma_p[:j + 1, j + 1:]
    sigma_21 = sigma_p[j + 1:, :j + 1]
    sigma_22 = sigma_p[j + 1:, j + 1:]

    sigma_hat = sigma_11 - multi_dot([sigma_12, inv(sigma_22), sigma_21])
    xjc = multivariate_normal(mu_2, sigma_22, No)

    for m in range(No):  # loop through outer loop
        mu_hat = mu_1.reshape((j + 1, 1)) + multi_dot(
            [sigma_12, inv(sigma_22), xjc[m].reshape((d - j - 1, 1)) - mu_2.reshape((d - j - 1, 1))])
        mu_hat = mu_hat.flatten()
        xj = multivariate_normal(mu_hat, sigma_hat, Ni)
        x_t[m * Ni:(m + 1) * Ni, :] = np.hstack([xj, np.tile(xjc[m], (Ni, 1))])
    x_D = x_t[:, s_index]
    return x_D
