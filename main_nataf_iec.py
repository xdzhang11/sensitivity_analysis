#%% Shapley effects with wind parameters with Nataf transformation
import joblib
from f_shapley import shapley
from f_X_gsa3 import X_dep_wt as X_dep
from f_X_gsa3 import X_j_wt as X_j
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
    SHs.append(shapley(cost, d, Nv, Ni, No, m, X_dep, X_j, 0.743))

    fn_sh_nataf = "results/sh_nataf_iec_cov_{}.txt".format(var)
    data = np.stack(SHs)
    np.savetxt(fn_sh_nataf, data, fmt='%.5f')

end = time.time()
print(end - start)

