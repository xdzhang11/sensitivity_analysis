import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import joblib
from f_X_gsa2 import X_j_us as X_j
import numpy as np
import matplotlib.ticker as mticker

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams['text.usetex'] = True

plt.rcParams['axes.labelsize'] = 14  # Font size for x and y labels
plt.rcParams['xtick.labelsize'] = 16 # Font size for x ticks
plt.rcParams['ytick.labelsize'] = 16  # Font size for y ticks
plt.rcParams['legend.fontsize'] = 18  # Font size for legend

plt.rcParams['figure.figsize'] = (8, 2)

# %% Read data
varlist = ['bTD', 'Mx_blade', 'Mx_tower']

var = varlist[0]
fn_rf = "models/cb_{}.joblib".format(var)
cb_rg = joblib.load(fn_rf)
def cost(x):
    return cb_rg.predict(x)

Nv = 10000  # MC sample size to estimate var(Y)
d = 5        # dimension of inputs
rho = 0.2
X_A = X_j(Nv, d, rho)
y =  cost(X_A)
VarY0 = np.var(y)
std0 = np.std(y)
cov0 = std0/np.mean(y)
# Plot bTD, histogram plot using density, save figure as pdf to Figures folder
# set figure size
fig, ax = plt.subplots(1, 1)
ax.hist(y, bins=50, density=True)
# ax.set_xlabel('clearance (m)', fontsize=12)
# ax.xlabel('Blade tower clearance (m)', fontsize=14)
ax.set_ylabel('Probability density')
plt.tight_layout()
tick_values = [0, 0.1,0.2]  # Set your desired tick values here
ax.set_yticks(tick_values)
plt.savefig("Figures/hist_bTD.pdf", format="pdf", bbox_inches="tight")
# jpg
# plt.savefig("Figures/bTD_hist.jpg", format="jpg", bbox_inches="tight")
plt.show()


#%%
# Mx_blade
var = varlist[1]
fn_rf = "models/cb_{}.joblib".format(var)
cb_rg = joblib.load(fn_rf)
def cost(x):
    return cb_rg.predict(x)

Nv = 10000  # MC sample size to estimate var(Y)
d = 5        # dimension of inputs
rho = 0.2
X_A = X_j(Nv, d, rho)
y =  cost(X_A)
VarY1 = np.var(y)
std1 = np.std(y)
cov1 = std1/np.mean(y)
# Plot Mx_blade, histogram plot using density, save figure as pdf to Figures folder
fig, ax = plt.subplots(1, 1)
ax.hist(y, bins=50, density=True)
# ax.set_xlabel('Mx blade (kNm)', fontsize=12)
# set y label ticks two digits after decimal point for the scientific notation
ax.set_ylabel('Probability density')

tick_values = [0.1e-5, 2.00e-5, 4.00e-5]  # Set your desired tick values here
ax.set_yticks(tick_values)

plt.tight_layout()
plt.savefig("Figures/hist_Mx_blade.pdf", format="pdf", bbox_inches="tight")
# jpg
# plt.savefig("Figures/Mx_blade_hist.jpg", format="jpg", bbox_inches="tight")
plt.show()

# Mx_tower
var = varlist[2]
fn_rf = "models/cb_{}.joblib".format(var)
cb_rg = joblib.load(fn_rf)
def cost(x):
    return cb_rg.predict(x)

Nv = 10000  # MC sample size to estimate var(Y)
d = 5        # dimension of inputs
rho = 0.2
X_A = X_j(Nv, d, rho)
y =  cost(X_A)
VarY2 = np.var(y)
std2 = np.std(y)
cov2 = std2/np.mean(y)
# Plot Mx_tower, histogram plot using density, save figure as pdf to Figures folder
fig, ax = plt.subplots(1, 1)
ax.hist(y, bins=50, density=True)
# ax.set_xlabel('Mx tower (kNm)', fontsize=12)
ax.set_ylabel('Probability density')
tick_values = [0, 0.5e-5, 1.0e-5]  # Set your desired tick values here
ax.set_yticks(tick_values)
plt.tight_layout()

plt.savefig("Figures/hist_Mx_tower.pdf", format="pdf", bbox_inches="tight")
# jpg
# plt.savefig("Figures/Mx_tower_hist.jpg", format="jpg", bbox_inches="tight")
plt.show()
