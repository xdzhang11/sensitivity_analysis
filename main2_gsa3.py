import joblib
import time
import shap
import numpy as np
import matplotlib.pyplot as plt
from f_shapley import shapley
#%% Shapley effects with wind parameters with Nataf transformation
from f_X_gsa3 import X_dep_wt as X_dep
from f_X_gsa3 import X_j_wt as X_j


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





