#%% all variables
#from torch.quasirandom import SobolEngine
%reset -f
from cmath import pi
from scipy.stats import qmc
from matplotlib import style
from numpy.random import rand
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
#%%
filename_Xy = os.path.join('Xy', 'Xy.h5')
df = pd.read_hdf(filename_Xy, 'Xy')
df.head()
#%%
X = df.iloc[:,0:5]
X.loc[:,'cl'] = 0.7+(1.3-0.7)*X.loc[:,'cl']
X.loc[:,'bladeIx']  = 0.7+(1.3-0.7)*X.loc[:,'bladeIx'] 
X.loc[:,'towerIx']  = 0.7+(1.3-0.7)*X.loc[:,'towerIx'] 
#%%
u = 3+X.loc[:,'wsp']*(27-3)
#%%
sigma_min = np.maximum(0, 0.1*(u-20))
sigma_max = 0.18*(6.8+0.75*u)
sigma = sigma_min+(sigma_max-sigma_min)*X.loc[:,'ti']
ti = sigma/u
X.loc[:,'wsp'] = u
X.loc[:,'ti'] = ti
X.head()

#%%
y = df.Mx_tower # blade tip clearance
gbr = RandomForestRegressor()
gbrm = gbr.fit(X, y)
#y_pred = gbrm.predict(X_test)


#%%
n = 2**20
d = 5
sampler = qmc.Sobol(d=2*d,scramble=False)
AB = sampler.random_base2(m=int(np.log2(n)))

#%%
AB = AB[1:n,:]
#A = AB[:,0:d]
#B = AB[:,d:2*d]
A = rand(n-1,d)
B = rand(n-1,d)
R = rand(n-1,d)

#%%
from scipy.stats import norm, halfnorm, rayleigh, weibull_min

def trans(M):
    v_ave = 8.5
    r_scale = np.sqrt(2/np.pi)*v_ave
    X_v = pd.DataFrame(data = np.zeros((n-1,d)), columns=['wsp', 'ti', 'cl', 'bladeIx', 'towerIx'])

    X_v.wsp = rayleigh.ppf(M[:,0], loc=0, scale=r_scale)
    wb_shape = 0.27*X_v.wsp+1.4
    Iref = 0.12
    wb_scale = Iref*(0.75*X_v.wsp+3.3)
    sigma = weibull_min.ppf(M[:,1], wb_shape, 0, wb_scale)
    ti = sigma/X_v.wsp
    X_v.ti = ti
    # #Fixed win paramters
    # X_v.wsp = 11.4
    # X_v.ti = 0.14
    
    # #half normal distribution parameters
    # mu_hn = 1
    # scale_hn = 0.05/np.sqrt(1-2/pi)
    # X_v.cl = 2-halfnorm.ppf(M[:,2], loc=mu_hn, scale=scale_hn)
    
    X_v.cl = norm.ppf(M[:,2], loc=1, scale=0.05)
    X_v.bladeIx = norm.ppf(M[:,3], loc=1, scale=0.05)
    X_v.towerIx = norm.ppf(M[:,4], loc=1, scale=0.05)
    return X_v

#%%

X_A = trans(A)
X_B = trans(B)
X_R = trans(R)
index = ((X_A.wsp>3)&(X_A.wsp<27)) & ((X_B.wsp>3)&(X_B.wsp<27)) & ((X_R.wsp>3)&(X_R.wsp<27))
X_A = X_A[index].reset_index(drop=True)
X_B = X_B[index].reset_index(drop=True)
X_R = X_R[index].reset_index(drop=True)
n = len(X_A)

y_A =  gbrm.predict(X_A).reshape((n,1))
y_B =  gbrm.predict(X_B).reshape((1,n))
Y =  gbrm.predict(X_R)
varY = np.var(Y)
#%%


# %%
import copy
S_i = np.zeros(d)
S_Ti = np.zeros(d)
for i in range(d):
# for i in range(2,d): #for three parameters
    X_A_Bi = copy.deepcopy(X_A)
    X_A_Bi.iloc[:,i] = X_B.iloc[:,i]
    y_A_Bi = gbrm.predict(X_A_Bi).reshape((n,1))
    S_i[i] = np.dot(y_B, y_A_Bi-y_A)/n/varY
    S_Ti[i] = np.sum((y_A-y_A_Bi)**2)/2/n/varY
    
    
# %%
np.set_printoptions(precision=3, suppress=True)

print(S_i)
print(S_Ti)
# %%


# %% Half Normal distribution
from scipy.stats import halfnorm
from cmath import pi
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(1, 1)



mu_hn = 1
scale_hn = 0.05/np.sqrt(1-2/pi)

mean, var, skew, kurt = halfnorm.stats(moments='mvsk',loc=mu_hn, scale=scale_hn)

x = np.linspace(halfnorm.ppf(0.001,loc=mu_hn, scale=scale_hn),
                halfnorm.ppf(0.999,loc=mu_hn, scale=scale_hn), 100)
ax.plot(2-x, halfnorm.pdf(x,loc=mu_hn, scale=scale_hn),
       'r-', lw=5, alpha=0.6, label='halfnorm pdf')

ax.set_xlabel('x')
ax.set_ylabel('pdf')

r = 2-halfnorm.rvs(size = 100000,loc=mu_hn, scale=scale_hn)