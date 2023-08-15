# Convert normalized X to their physical space
import os
import pandas as pd
import numpy as np

filename_Xy = os.path.join('Xy', 'Xy.h5')
df0 = pd.read_hdf(filename_Xy, 'Xy')
df0.rename({'ti': 'sigma'}, axis=1, inplace=True)
df = df0.copy()


u = 3+df0.wsp*(27-3)
df.wsp = u

sigma_min = np.maximum(0, 0.1*(u-20))
sigma_max = 0.18*(6.8+0.75*u)
sigma = sigma_min+(sigma_max-sigma_min)*df0.sigma
df.sigma = sigma

# df.cl = 0.7+(1.3-0.7)*df0.cl
# df.bladeIx = 0.7+(1.3-0.7)*df0.bladeIx
# df.towerIx = 0.7+(1.3-0.7)*df0.towerIx

df = df.apply(lambda x: 0.7+(1.3-0.7)*x if x.name in ['cl', 'bladeIx', 'towerIx'] else x)

filename_Xy = os.path.join('Xy', 'data_dtu10mwt.h5')
df.to_hdf(filename_Xy, key='Xy', mode='w')
