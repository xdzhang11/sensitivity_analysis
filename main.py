#%%
%reset -f
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import norm, qmc

#%%
filename_Xy = os.path.join('Xy', 'Xy.h5')
df = pd.read_hdf(filename_Xy, 'Xy')

# %%
# plt.figure()
plt.scatter(df.wsp, df.Mx_blade)

#%%
df.head()
# %%
X = df.iloc[:,0:5]
X.loc[:,'cl'] = 0.7+(1.3-0.7)*X.loc[:,'cl']
X.loc[:,'bladeIx']  = 0.7+(1.3-0.7)*X.loc[:,'bladeIx'] 
X.loc[:,'towerIx']  = 0.7+(1.3-0.7)*X.loc[:,'towerIx'] 
X.head()


#%%
plt.rcParams["figure.figsize"] = (5,6)
plt.scatter(df.cl, df.Mx_blade)
plt.xlabel('Cl', fontsize=18)
plt.ylabel('Mx (blade)', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(r'./Figures/Cl_Mx_blade.jpg')
plt.show()
#%%
plt.rcParams["figure.figsize"] = (5,6)
plt.scatter(df.bladeIx, df.Mx_blade)
plt.xlabel('bladeIx', fontsize=18)
plt.ylabel('Mx (blade)', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(r'./Figures/blade_Ix_Mx_blade.jpg')
plt.show()
#%%

plt.rcParams["figure.figsize"] = (5,6)
plt.scatter(df.towerIx, df.Mx_blade)
plt.xlabel('towerIx', fontsize=18)
plt.ylabel('Mx (blade)', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(r'./Figures/tower_Ix_Mx_blade.jpg')
plt.show()



#%%
n = 1000
r_v = norm.rvs(loc=1, scale=0.05, size=n)
sampler = qmc.Halton(2)
u_sample = sampler.random(n)

X_r = pd.DataFrame(data = np.zeros((n,5)), columns=['wsp', 'ti', 'cl', 'bladeIx', 'towerIx'])
X_r.iloc[:,:2] = u_sample
X_r.cl = 1
X_r.bladeIx = 1
X_r.towerIx = 1

#%%
y = df.Mx_tower  # blade tip clearance
gbr = RandomForestRegressor()
gbrm = gbr.fit(X, y)

#%%

y_n_list = []
var_list = ['cl','bladeIx','towerIx']

for k in range(len(var_list)):
    
    var_name = var_list[k]
    y_n = []
    
    for j in range(n):
    
        X_n = X_r.copy(deep=True)
        X_n.loc[:,var_name] = r_v[j]

        # u = 3+u_n*(27-3)
        # sigma_min = np.max([0, 0.1*(u-20)])
        # sigma_max = 0.18*(6.8+0.75*u)
        # sigma = sigma_min+(sigma_max-sigma_min)*ti_n
        # ti = sigma/u
        y_n.append(np.percentile(gbrm.predict(X_n),10))

    y_n_list.append(y_n)
    

plt.rcParams["figure.figsize"] = (5,6)
plt.boxplot(y_n_list, labels = ['cl','bladeIx','towerIx'])
plt.title('Boxplot')
plt.ylabel('Mx (tower)', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(r'./Figures/Boxplot.jpg')
plt.show()







#%%
# y = df.bTD  # blade tip clearance
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# #%% Ensemble-random forest
# gbr = RandomForestRegressor()
# gbrm = gbr.fit(X_train, y_train)
# print('reg_score: %.3f' % gbrm.score(X_test, y_test))
# y_pred = gbrm.predict(X_test)
# mae = mean_absolute_error(y_test, y_pred)
# print('MAE: %.3f' % mae)
# plt.scatter(y_test, y_pred)

# plt.title('Random forest')
# plt.xlabel('predit', fontsize = 18)
# plt.ylabel('true', fontsize=18)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.tight_layout()
# plt.savefig(r'./Figures/randomForest.jpg')

# plt.show()

#%% tensorflow
# scaler = MinMaxScaler()
# y_train_n = y_train.values.reshape(len(y_train), 1)
# y_train_n = scaler.fit_transform(y_train_n)

# def plot_loss(history):
#   plt.plot(history.history['loss'], label='loss')
#   plt.plot(history.history['val_loss'], label='val_loss')
#   plt.ylim([0, 10])
#   plt.xlabel('Epoch')
#   plt.ylabel('Error [MPG]')
#   plt.legend()
#   plt.grid(True)
#   plt.show()

# def build_and_compile_model():
#   model = keras.Sequential([
#       layers.Dense(16, activation='relu'),
#       layers.Dense(8, activation='relu'),
#       layers.Dense(1, activation='linear')
#   ])
#   model.compile(loss=tf.keras.losses.MeanSquaredError(),
#                 optimizer=tf.keras.optimizers.Adam())
#   return model


# model = build_and_compile_model()
# history = model.fit(
#     X_train,
#     y_train_n,
#     validation_split=0.2,
#     verbose=0, epochs=100)

# #model.summary()
# #plt.figure()
# ##plot_loss(history)
# y_pred_n = model.predict(X_test)

# y_pred = scaler.inverse_transform(y_pred_n)

# print('r2: %.3f' % r2_score(y_test, y_pred))
# mae = mean_absolute_error(y_test, y_pred)
# print('MAE: %.3f' % mae)
# plt.figure()
# plt.scatter(y_test, y_pred)


#%% Surrogate model
# gbr = RandomForestRegressor()
# gbrm = gbr.fit(X, y)
#%%

# n = 1000
# r_v = norm.rvs(loc=1, scale=0.05, size=n)

# y_n_list = []
# var_list = ['cl','bladeIx','towerIx']

# for k in range(len(var_list)):

#     var_name = var_list[k]

#     X_r = pd.DataFrame(data = np.zeros((n,5)), columns=['wsp', 'ti', 'cl', 'bladeIx', 'towerIx'])

#     u_n = 0.75 # normalized wind speed value
#     ti_n = 0.5 # normalized turbulence intensity
#     X_r.wsp = u_n
#     X_r.ti = ti_n
#     X_r.bladeIx = 1
#     X_r.towerIx = 1
#     X_r.cl = 1

#     X_n = X_r.copy(deep=True)
#     X_n.loc[:,var_name] = r_v

#     u = 3+u_n*(27-3)
#     sigma_min = np.max([0, 0.1*(u-20)])
#     sigma_max = 0.18*(6.8+0.75*u)
#     sigma = sigma_min+(sigma_max-sigma_min)*ti_n
#     ti = sigma/u
    
#     y_r = gbrm.predict(np.array(X_r.iloc[0,:]).reshape(1,-1))
#     y_n = gbrm.predict(X_n)

#     y_n_list.append(y_n)
    
#     plt.rcParams["figure.figsize"] = (20,6)
#     plt.scatter(r_v,y_n)
#     plt.xlabel('{}'.format(var_name), fontsize=18)
#     plt.ylabel('blade tower clearance', fontsize=18)
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     plt.savefig(r'./Figures/{}_01.jpg'.format(var_name))

#     plt.show()
    
    
#     plt.rcParams["figure.figsize"] = (20,6)
#     plt.subplot(1, 2, 1)
#     plt.hist(r_v, density=True, bins=100)
#     plt.xlabel('{}'.format(var_name), fontsize=18)
#     plt.ylabel('pdf', fontsize=18)
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     plt.subplot(1, 2, 2)
#     plt.hist(y_n, density=True, bins = 100)
#     plt.xlabel('blade tower clearance', fontsize=18)
#     plt.ylabel('pdf', fontsize=18)
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     plt.plot(float(y_r),0,marker="o", markersize= 10)

#     plt.savefig(r'./Figures/{}_02.jpg'.format(var_name))
#     plt.show()


# plt.rcParams["figure.figsize"] = (5,6)
# plt.boxplot(y_n_list, labels = ['cl','bladeIx','towerIx'])
# plt.title('Boxplot')
# plt.ylabel('blade tower clearance', fontsize=18)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.tight_layout()
# plt.savefig(r'./Figures/Boxplot.jpg')
# plt.show()
# # %%
# X_r = pd.DataFrame(data = np.zeros((n,5)), columns=['wsp', 'ti', 'cl', 'bladeIx', 'towerIx'])

# sampler = qmc.Halton(2)
# halton = sampler.random(n)

# X_r.iloc[:,:2] = halton
# X_r.bladeIx = 1
# X_r.towerIx = 1
# X_r.cl = 1

# X_n = X_r.copy(deep=True)

# y_r = gbrm.predict(np.array(X_r.iloc[0,:]).reshape(1,-1))
# y_n = gbrm.predict(X_n)

# y_n_list.append(y_n)

# fig = plt.figure(figsize = (10, 7))
# ax = plt.axes(projection ="3d")
# ax.scatter3D(X_n.iloc[:,0],X_n.iloc[:,1], y_n)
# ax.set_xlabel('wind speed', fontsize=18)
# ax.set_ylabel('ti', fontsize=18)
# ax.set_zlabel('blade tower clearance', fontsize=18)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.savefig(r'./Figures/wind_01.jpg')

# plt.show()
# #%%

# plt.rcParams["figure.figsize"] = (20,6)
# plt.subplot(1, 2, 2)
# plt.hist(y_n, density=True, bins = 100)
# plt.xlabel('blade tower clearance', fontsize=18)
# plt.ylabel('pdf', fontsize=18)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.plot(float(y_r),0,marker="o", markersize= 10)

# plt.savefig(r'./Figures/wind_02.jpg')
# plt.show()
# #%%

# plt.rcParams["figure.figsize"] = (5,6)
# plt.boxplot(y_n_list, labels = ['cl','bladeIx','towerIx','wind'])
# plt.title('Boxplot')
# plt.ylabel('blade tower clearance', fontsize=18)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.tight_layout()
# plt.savefig(r'./Figures/Boxplot.jpg')
# plt.show()
# %%


# %%
