import joblib
import time
import shap
import numpy as np
import matplotlib.pyplot as plt
from f_shapley import shapley

plt.rcParams["figure.figsize"] = (5, 6)


#%% The theoretical case in paper:
from f_X import X_dep_a as X_dep
from f_X import X_j_a as X_j


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


#%% Shapley effects with wind parameters follows distribution in IEC
from f_X import X_dep_us as X_dep
from f_X import X_j_us as X_j


gbrm = joblib.load("models/randomforests.joblib")
def cost(x):
    return gbrm.predict(x)


d = 5        # dimension of inputs
Nv = 100000  # MC sample size to estimate var(Y)
Ni = 30      # sample size for inner loop
No = 10      # sample size for outer loop
m = 1000
t = time.time()
SH = shapley(cost, d, Nv, Ni, No, m, X_dep, X_j)
elapsed = time.time() - t


height = SH
bars = ['wsp', 'ti', 'cl', 'bladeIx', 'towerIx']
y_pos = np.arange(len(bars))
plt.bar(y_pos, height)
plt.xticks(y_pos, bars)
plt.title('Shapley effects')
plt.show()

#%% shap package
from f_X import X_j_us as X_j

X_test = X_j(1000, 5, 0.5)
gbrm = joblib.load("models/randomforests.joblib")

explainer = shap.TreeExplainer(gbrm)
shap_values = explainer.shap_values(X_test)

fig, ax = plt.subplots(1, 1)
shap.summary_plot(shap_values, X_test)

fig, ax = plt.subplots(1, 1)
shap.summary_plot(shap_values, X_test, plot_type="bar")


#%% Random forests feature importance
height = gbrm.feature_importances_
bars = ['wsp', 'ti', 'cl', 'bladeIx', 'towerIx']
y_pos = np.arange(len(bars))
# Create bars
plt.bar(y_pos, height)
plt.xticks(y_pos, bars)
plt.title('Random forests feature importance')
plt.show()


#%% Shapley effects with wind parameters with Nataf transformation
from f_X import X_dep_wt as X_dep
from f_X import X_j_wt as X_j


gbrm = joblib.load("models/randomforests.joblib")
def cost(x):
    return gbrm.predict(x)


d = 5        # dimension of inputs
Nv = 100000  # MC sample size to estimate var(Y)
Ni = 30      # sample size for inner loop
No = 10      # sample size for outer loop
m = 1000
t = time.time()
SH = shapley(cost, d, Nv, Ni, No, m, X_dep, X_j)
elapsed = time.time() - t


#%% Sobol index
from f_sobol import sobol

gbrm = joblib.load("models/randomforests.joblib")
def cost(x):
    return gbrm.predict(x)

n = 2 ** 20
d = 5
S_i, S_Ti = sobol(cost, d, n)
np.set_printoptions(precision=3, suppress=True)

print(S_i)
print(S_Ti)



