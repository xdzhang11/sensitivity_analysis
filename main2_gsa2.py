#%% Shapley effects with wind parameters follows distribution in IEC
import joblib
import time
import numpy as np
import matplotlib.pyplot as plt
from f_shapley import shapley
from f_X_gsa2 import X_dep_us as X_dep
from f_X_gsa2 import X_j_us as X_j
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (5, 6)


varlist = ['bTD', 'Mx_blade', 'Mx_tower']

for k in range(3):
    var = varlist[k]
    fn_rf = "models/cb_{}.joblib".format(var)

    cb_rg = joblib.load(fn_rf)

    def cost(x):
        return cb_rg.predict(x)


    d = 5        # dimension of inputs
    Nv = 1000000  # MC sample size to estimate var(Y)
    Ni = 100      # sample size for inner loop
    No = 10      # sample size for outer loop
    m = 10000
    t = time.time()
    SH = shapley(cost, d, Nv, Ni, No, m, X_dep, X_j, 0.2)
    elapsed = time.time() - t


    r_sh_iec = {'SH': SH, 'Time': elapsed, 'Nv': Nv, 'Ni': Ni, 'No': No, 'm': m}

    fn_sh_iec = "results/sh_iec_{}.txt".format(var)
    with open(fn_sh_iec, 'w') as f:
        print(r_sh_iec, file=f)


height = SH
bars = ['wsp', 'sigma', 'cl', 'bladeIx', 'towerIx']
y_pos = np.arange(len(bars))
plt.bar(y_pos, height)
plt.xticks(y_pos, bars)
plt.title('Shapley effects')
plt.show()






