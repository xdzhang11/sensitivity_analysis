# Shapley effects evaluation in parallel
# Last update 2/6/2025
# @author: xiazang@dtu.dk; xiaodong.zhang@u.nus.edu
import joblib
from joblib import Parallel, delayed
import numpy as np
import itertools

num_cores = joblib.cpu_count()-1


def shapley_random(cost, d, Nv, Ni, No, m, X_dep, X_j, rho=0.2):
    X_A = X_j(Nv, d, rho)
    y = cost(X_A)
    VarY = np.var(y)

    def f_shapley_s(p, cost):
        Sh = np.zeros(d)
        Sh2 = np.zeros(d)
        pi = np.random.permutation(d)
        s_index = np.argsort(pi)  # sorted

        prevC = 0

        for j in range(d):  # loop through dimension

            if j == d - 1:
                c_hat = VarY
                # delta = c_hat-prevC
            else:
                X_D = X_dep(pi, s_index, No, Ni, d, j, rho)
                y_all = cost(X_D)
                cVar = []  # conditional variance
                for l in range(No):  # loop through outer loop
                    c_Y = y_all[l * Ni:(l + 1) * Ni]
                    cVar.append(np.var(c_Y))

                c_hat = np.mean(cVar)
            delta = c_hat - prevC

            Sh[pi[j]] = Sh[pi[j]] + delta
            Sh2[pi[j]] = Sh2[pi[j]] + delta ** 2
            prevC = c_hat
        return Sh, Sh2

    res = Parallel(n_jobs=num_cores)(delayed(f_shapley_s)(i, cost) for i in range(m))

    r = [item[0] for item in res]
    r2 = [item[1] for item in res]

    Sh = np.sum(r, axis=0)
    Sh2 = np.sum(r2, axis=0)

    Sh = Sh / m / VarY
    Sh2 = Sh2 / m
    SEShapley = np.sqrt((Sh2 - Sh ** 2) / m)

    return Sh

def shapley_exact(cost, d, Nv, Ni, No, m, X_dep, X_j, rho=0.2):
    X_A = X_j(Nv, d, rho)
    y = cost(X_A)
    VarY = np.var(y)

    prange = [k for k in range(d)]
    permutations = np.array(list(itertools.permutations(prange)))
    m = len(permutations)

    def f_shapley_s(p, cost):
        Sh = np.zeros(d)
        Sh2 = np.zeros(d)
        pi = permutations[p, :]
        s_index = np.argsort(pi)  # sorted

        prevC = 0

        for j in range(d):  # loop through dimension

            if j == d - 1:
                c_hat = VarY
                #delta = c_hat - prevC
            else:
                X_D = X_dep(pi, s_index, No, Ni, d, j, rho)
                y_all = cost(X_D)
                cVar = []  # conditional variance
                for l in range(No):  # loop through outer loop
                    c_Y = y_all[l * Ni:(l + 1) * Ni]
                    cVar.append(np.var(c_Y))

                c_hat = np.mean(cVar)
            delta = c_hat - prevC

            Sh[pi[j]] = Sh[pi[j]] + delta
            Sh2[pi[j]] = Sh2[pi[j]] + delta ** 2
            prevC = c_hat
        return Sh, Sh2

    res = Parallel(n_jobs=num_cores)(delayed(f_shapley_s)(i, cost) for i in range(m))

    r = [item[0] for item in res]
    r2 = [item[1] for item in res]

    Sh = np.sum(r, axis=0)
    Sh2 = np.sum(r2, axis=0)

    Sh = Sh / m / VarY
    Sh2 = Sh2 / m
    SEShapley = np.sqrt((Sh2 - Sh ** 2) / m)

    return Sh
