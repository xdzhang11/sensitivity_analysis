import joblib
import numpy as np
import matplotlib.pyplot as plt
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



