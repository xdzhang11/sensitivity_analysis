#%% Shapley effects with wind parameters with Nataf transformation
import joblib
from f_shapley import shapley
from f_X_gsa3 import X_dep_wt as X_dep
from f_X_gsa3 import X_j_wt as X_j
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams['text.usetex'] = True

plt.rcParams['axes.labelsize'] = 18  # Font size for x and y labels
plt.rcParams['xtick.labelsize'] = 18 # Font size for x ticks
plt.rcParams['ytick.labelsize'] = 18  # Font size for y ticks
plt.rcParams['legend.fontsize'] = 18  # Font size for legend

#%%
start = time.time()
varlist = ['bTD', 'Mx_blade', 'Mx_tower']

for k in range(3):
    var = varlist[k]
    fn_rf = "models/cb_{}.joblib".format(var)
    cb_rg = joblib.load(fn_rf)

    def cost(x):
        return cb_rg.predict(x)

    d = 5        # dimension of inputs
    Nv = 1000000  # MC sample size to estimate var(Y)
    Ni = 1000      # sample size for inner loop
    No = 10      # sample size for outer loop
    m = 10000

    SHs = []
    cs = np.linspace(-0.98, 0.98, 8)
    for c in cs:
        SHs.append(shapley(cost, d, Nv, Ni, No, m, X_dep, X_j, c))

    fn_sh_nataf = "results/sh_nataf_{}.txt".format(var)
    data = np.stack(SHs)
    np.savetxt(fn_sh_nataf, data, fmt='%.5f')

end = time.time()
print(end - start)
# %%

cs = np.linspace(-0.98, 0.98, 8)
cs = np.insert(cs, 4, 0)

varlist = ['bTD', 'Mx_blade', 'Mx_tower']
var = varlist[2]
fn_sh_nataf = "results/sh_nataf_{}.txt".format(var)
data = np.loadtxt(fn_sh_nataf)
df = pd.DataFrame(data=data, index=cs, columns=['wsp', 'sigma', 'cl', 'bladeIx', 'towerIx'])

# plt.rcParams["figure.figsize"] = (6, 3)

fig, axes = plt.subplots(1, 2, figsize=(12, 3))

axes[0].plot(df.wsp, '--',  lw=3, alpha=0.9, label='$u$')
axes[0].plot(df.sigma, '-.',  lw=3, alpha=0.9,label='$\sigma$')
# plt.plot((df.wsp+df.sigma)/2, 'k', label='mean')
axes[0].set_xlabel(r"Correlation coefficient $\rho_{\mu,\sigma}$")
axes[0].set_ylabel("Shapley effects")
axes[0].legend(loc='upper right',fontsize=14)

# plt.savefig("Figures/shap_nataf1_{}.pdf".format(var), format="pdf", bbox_inches="tight")
# plt.show()

# fig, ax = plt.subplots(1, 1, figsize=(6, 3))
axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
axes[1].plot(df.cl, '--',  lw=3, alpha=0.9,label='$C_L$')
axes[1].plot(df.bladeIx, '-.',  lw=3, alpha=0.9,label='blade Ix')
axes[1].plot(df.towerIx, ':', lw=3, alpha=0.9, label='tower Ix')
# plt.plot((df.cl+df.bladeIx+df.towerIx)/3, 'k', label='mean')
axes[1].set_xlabel(r"Correlation coefficient $\rho_{\mu, \sigma}$")
axes[1].set_ylabel("Shapley effects")
tick_values = [0, 0.5e-2, 1e-2, 1.5e-2]  # Set your desired tick values here
# tick_values = [0, 0.5e-2, 1e-2]  # Set your desired tick values here
axes[1].set_yticks(tick_values)

axes[1].legend(loc='upper right',fontsize=14)
plt.tight_layout()
plt.savefig("Figures/shap_nataf_{}.pdf".format(var), format="pdf", bbox_inches="tight")
plt.show()

