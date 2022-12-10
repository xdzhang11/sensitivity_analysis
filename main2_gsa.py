import joblib
import time
import shap
import numpy as np
import matplotlib.pyplot as plt

# from f_X import X_dep_wt as X_dep
# from f_X import X_j_wt as X_j

from f_shapley import shapley
from f_sobol import sobol
from f_X import X_dep_a as X_dep
from f_X import X_j_a as X_j


# gbrm = joblib.load("models/randomforests.joblib")
# def cost(x):
#     return gbrm.predict(x)

def cost(x):
    beta = np.array([1,1,1]).transpose()
    return x.dot(beta)

#%% Sobol index

n = 2 ** 20
d = 5

S_i, S_Ti = sobol(cost, d, n)

np.set_printoptions(precision=3, suppress=True)

print(S_i)
print(S_Ti)

#%% Shapley effects
d = 5       # dimension of inputs
Nv = 100000  # MC sample size to estimate var(Y)
Ni = 30      # sample size f
# or inner loop
No = 10      # sample size for outer loop
m = 10000
# X_test = X_j(100, d)

#%%

# t = time.time()# -*- coding: utf-8 -*-

# SH = shapley(cost, d, Nv, Ni, No, m, X_dep, X_j)


# elapsed = time.time() - t
# elapsed

#%%

d = 3       # dimension of inputs

rho23_all =  np.linspace(-0.9,.9,20)
s1 = []
s2 = []
s3 = []

for rho in rho23_all:
    Sh = shapley(cost, d, Nv, Ni, No, m, X_dep, X_j, rho)
    s1.append(Sh[0])
    s2.append(Sh[1])
    s3.append(Sh[2])

plt.scatter(rho23_all,s1,label="x1_numerical")
plt.scatter(rho23_all,s2,label="x2_numerical")
plt.scatter(rho23_all,s3,label="x3_numerical")



beta = np.array([1,1,1]).transpose()
sigma_i = [1, 1, 2]

sigma_s = 6+4*rho23_all
Sh1 = sigma_s**(-1)
plt.plot(rho23_all,Sh1,label="x1_theoretical")

Sh2 = (1+rho23_all*2+1.5*rho23_all**2)*sigma_s**(-1)
plt.plot(rho23_all,Sh2,label="x2_theoretical")

Sh3 = (4+rho23_all*2-1.5*rho23_all**2)*sigma_s**(-1)
plt.plot(rho23_all,Sh3,label="x3_theoretical")

plt.legend()
plt.show()



# plt.rcParams["figure.figsize"] = (5,6)

# height = SH
# bars = list(X_test.columns)
# y_pos = np.arange(len(bars))
# # Create bars
# plt.bar(y_pos, height)

# # Create names on the x-axis
# plt.xticks(y_pos, bars)
# plt.title('Shapley effects')

# # Show graphic
# plt.show()
# #%%


# plt.rcParams["figure.figsize"] = (5,6)

# height = gbrm.feature_importances_
# bars = list(X_test.columns)
# y_pos = np.arange(len(bars))
# # Create bars
# plt.bar(y_pos, height)

# # Create names on the x-axis
# plt.xticks(y_pos, bars)
# plt.title('Random forests feature importance')

# # Show graphic
# plt.show()

# #%%

# explainer = shap.TreeExplainer(gbrm)
# shap_values = explainer.shap_values(X_test)

# shap.summary_plot(shap_values, X_test)

# shap.summary_plot(shap_values, X_test, plot_type="bar")

#%%
