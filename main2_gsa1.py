#%% The theoretical case in paper:
import numpy as np
import matplotlib.pyplot as plt
from f_shapley import shapley
from f_X_gsa1 import X_dep_a as X_dep
from f_X_gsa1 import X_j_a as X_j

plt.rcParams["figure.figsize"] = (5, 6)


def cost(x):
    beta = np.array([1, 1, 1]).transpose()
    return x.dot(beta)


d = 3       # dimension of inputs
Nv = 100000  # MC sample size to estimate var(Y)
Ni = 30      # sample size for inner loop
No = 10      # sample size for outer loop
m = 10000

rho23_all = np.linspace(-0.9, .9, 20)
s1 = []
s2 = []
s3 = []

for rho in rho23_all:
    Sh = shapley(cost, d, Nv, Ni, No, m, X_dep, X_j, rho)
    s1.append(Sh[0])
    s2.append(Sh[1])
    s3.append(Sh[2])

fig, ax = plt.subplots(1, 1)

plt.scatter(rho23_all, s1, label="x1_numerical")
plt.scatter(rho23_all, s2, label="x2_numerical")
plt.scatter(rho23_all, s3, label="x3_numerical")

beta = np.array([1, 1, 1]).transpose()
sigma_i = [1, 1, 2]

sigma_s = 6+4*rho23_all
Sh1 = sigma_s**(-1)
plt.plot(rho23_all, Sh1, label="x1_theoretical")

Sh2 = (1+rho23_all*2+1.5*rho23_all**2)*sigma_s**(-1)
plt.plot(rho23_all, Sh2, label="x2_theoretical")

Sh3 = (4+rho23_all*2-1.5*rho23_all**2)*sigma_s**(-1)
plt.plot(rho23_all, Sh3, label="x3_theoretical")

plt.legend()
plt.show()