from numpy.random import multivariate_normal, rand
import numpy as np
from numpy.linalg import multi_dot, inv


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
