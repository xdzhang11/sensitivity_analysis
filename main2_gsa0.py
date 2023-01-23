#%% Sobol index
import joblib
import numpy as np
from f_sobol import sobol
import time

varlist = ['bTD', 'Mx_blade', 'Mx_tower']
var = varlist[2]
fn_rf = "models/cb_{}.joblib".format(var)

cb_rg = joblib.load(fn_rf)

def cost(x):
    return cb_rg.predict(x)

n = int(1e7)
d = 5

t = time.time()
S_i, S_Ti = sobol(cost, d, n)
elapsed = time.time() - t
np.set_printoptions(precision=3, suppress=True)

print(S_i)
print(S_Ti)


r_sobol = {'Si': S_i, 'STi': S_Ti, 'Time': elapsed, 'n': n}

fn_sobol = "results/sobol_{}.txt".format(var)
with open(fn_sobol, 'w') as f:
    print(r_sobol, file=f)


