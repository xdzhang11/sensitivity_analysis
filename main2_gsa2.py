import joblib
import time
import shap
import numpy as np
import matplotlib.pyplot as plt
from f_shapley import shapley
#%% Shapley effects with wind parameters follows distribution in IEC
from f_X_gsa2 import X_dep_us as X_dep
from f_X_gsa2 import X_j_us as X_j

plt.rcParams["figure.figsize"] = (5, 6)

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






