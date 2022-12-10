from sklearn.ensemble import RandomForestRegressor
import joblib
import pandas as pd
import numpy as np
import os

filename_Xy = os.path.join('Xy', 'Xy.h5')
df = pd.read_hdf(filename_Xy, 'Xy')
df.head()
#%%
X = df.iloc[:, 0:5]
X = X.copy()
X.cl = 0.7+(1.3-0.7)*X.cl
X.bladeIx = 0.7+(1.3-0.7)*X.bladeIx
X.towerIx = 0.7+(1.3-0.7)*X.towerIx

#%%
u = 3+X.loc[:, 'wsp']*(27-3)
sigma_min = np.maximum(0, 0.1*(u-20))
sigma_max = 0.18*(6.8+0.75*u)
sigma = sigma_min+(sigma_max-sigma_min)*X.loc[:,'ti']
ti = sigma/u
X.loc[:, 'wsp'] = u
X.loc[:, 'ti'] = ti
X.head()

#%%
y = df.Mx_tower # blade tip clearance
gbr = RandomForestRegressor()
gbrm = gbr.fit(X, y)

joblib.dump(gbrm, "models/randomforests.joblib")
