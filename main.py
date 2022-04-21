#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_absolute_error

#%%
filename_Xy = os.path.join('Xy', 'Xy.h5')
df = pd.read_hdf(filename_Xy, 'Xy')


#%%
df = df.drop(df[(df.wsp > 0.8) & (df.Mx_blade < 17500)].index)
df = df.drop(df[(df.wsp > 0.8) & (df.Mx_blade > 41000)].index)
df = df.drop(df[(df.wsp > 0.8) & (df.My_blade < 10800)].index)
df = df.drop(df[(df.wsp > 0.8) & (df.Mres_tower > 280000)].index)

#%%
df.head()
# %%
X = df.iloc[:,0:5]
y = df.Mx_tower  # blade tip clearance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


#%% Ensemble-random forest
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
gbr = RandomForestRegressor()
gbrm = gbr.fit(X_train, y_train)
print('reg_score: %.3f' % gbrm.score(X_test, y_test))
y_pred = gbrm.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print('MAE: %.3f' % mae)
plt.scatter(y_test, y_pred)
plt.show()



#%% tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
y_train_n = y_train.values.reshape(len(y_train), 1)
y_train_n = scaler.fit_transform(y_train_n)

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)
  plt.show()

def build_and_compile_model():
  model = keras.Sequential([
      layers.Dense(16, activation='relu'),
      layers.Dense(8, activation='relu'),
      layers.Dense(1, activation='linear')
  ])
  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam())
  return model


model = build_and_compile_model()
history = model.fit(
    X_train,
    y_train_n,
    validation_split=0.2,
    verbose=0, epochs=100)

#model.summary()
#plt.figure()
##plot_loss(history)
y_pred_n = model.predict(X_test)

y_pred = scaler.inverse_transform(y_pred_n)

print('r2: %.3f' % r2_score(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print('MAE: %.3f' % mae)
plt.figure()
plt.scatter(y_test, y_pred)

# %%
plt.figure()
plt.scatter(df.wsp, df.bTD)
# %%
