#%% Shapley effects with wind parameters with Nataf transformation
import joblib
from f_shapley import shapley
from f_X_gsa3_2 import X_dep_wt as X_dep
from f_X_gsa3_2 import X_j_wt as X_j
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

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

plt.rcParams["figure.figsize"] = (6, 3)

fig, ax = plt.subplots(1, 1)
plt.plot(df.wsp, '--',  label='$u$')
plt.plot(df.sigma, '-.', label='$\sigma$')
plt.plot((df.wsp+df.sigma)/2, 'k', label='mean')
plt.xlabel(r"Correlation coefficient $\rho_{\mu,\sigma}$")
plt.legend()
plt.tight_layout()
plt.savefig("Figures/shap_nataf1_{}.pdf".format(var), format="pdf", bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(1, 1)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.plot(df.cl, '--', label='cl')
plt.plot(df.bladeIx, '-.', label='blade Ix')
plt.plot(df.towerIx, ':', label='tower Ix')
plt.plot((df.cl+df.bladeIx+df.towerIx)/3, 'k', label='mean')
plt.xlabel(r"Correlation coefficient $\rho_{\mu, \sigma}$")
plt.legend()
plt.tight_layout()
plt.savefig("Figures/shap_nataf2_{}.pdf".format(var), format="pdf", bbox_inches="tight")
plt.show()

