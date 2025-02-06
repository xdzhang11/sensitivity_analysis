import numpy as np
import copy
from sample_generation import x_all

def sobol(cost, d, n):
    X_A = x_all(n)
    X_B = x_all(n)
    X_R = x_all(n)

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