#%% Shapley effects with wind parameters with Nataf transformation
import joblib
import time
from f_shapley import shapley
from f_X_gsa3 import X_dep_wt as X_dep
from f_X_gsa3 import X_j_wt as X_j


varlist = ['bTD', 'Mx_blade', 'Mx_tower']

for k in range(3):
    var = varlist[k]
    fn_rf = "models/cb_{}.joblib".format(var)

    cb_rg = joblib.load(fn_rf)

    def cost(x):
        return cb_rg.predict(x)


    d = 5        # dimension of inputs
    Nv = 100000  # MC sample size to estimate var(Y)
    Ni = 30      # sample size for inner loop
    No = 10      # sample size for outer loop
    m = 1000
    t = time.time()
    SH = shapley(cost, d, Nv, Ni, No, m, X_dep, X_j)
    elapsed = time.time() - t

    c = 5
    r_sh_nataf = {'SH': SH, 'Time': elapsed, 'Nv': Nv, 'Ni': Ni, 'No': No, 'm': m, 'rho': 0.1*c}

    fn_sh_nataf = "results/sh_nataf{}_{}.txt".format(c, var)
    with open(fn_sh_nataf, 'w') as f:
        print(r_sh_nataf, file=f)




