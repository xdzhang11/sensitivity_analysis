#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
X = df.iloc[:,0:5]
X = X.copy()
X.cl = 0.7+(1.3-0.7)*X.cl
X.bladeIx = 0.7+(1.3-0.7)*X.bladeIx
X.towerIx = 0.7+(1.3-0.7)*X.towerIx

u = 3+X.loc[:, 'wsp']*(27-3)
sigma_min = np.maximum(0, 0.1*(u-20))
sigma_max = 0.18*(6.8+0.75*u)
sigma = sigma_min+(sigma_max-sigma_min)*X.loc[:,'ti']
ti = sigma/u
X.loc[:, 'wsp'] = u
X.loc[:, 'ti'] = ti
X.head()
y = df.Mx_tower


#%%
y = df.bTD  # blade tip clearance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
#%% Ensemble-random forest
gbr = RandomForestRegressor()
gbrm = gbr.fit(X_train, y_train)
print('reg_score: %.3f' % gbrm.score(X_test, y_test))
y_pred = gbrm.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print('MAE: %.3f' % mae)
plt.scatter(y_test, y_pred)

plt.title('Random forest')
plt.xlabel('predit', fontsize = 18)
plt.ylabel('true', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(r'./Figures/randomForest.jpg')

plt.show()

#%% tensorflow
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

model.summary()
plt.figure()
plot_loss(history)
y_pred_n = model.predict(X_test)

y_pred = scaler.inverse_transform(y_pred_n)

print('r2: %.3f' % r2_score(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print('MAE: %.3f' % mae)
plt.figure()
plt.scatter(y_test, y_pred)
plt.show()

