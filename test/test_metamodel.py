import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import os
import joblib


# %% Read data
filename_Xy = os.path.join('data', 'data_dtu10mwt.h5')
df = pd.read_hdf(filename_Xy, 'Xy')
X = df.iloc[:, 0:5]

#%%
varlist = ['bTD', 'Mx_blade', 'Mx_tower']
var = varlist[0]
y = df[var]

#%% Ensemble-random forest
# Best parameter selection
reg_rf = RandomForestRegressor()
param_grid = {
                 'n_estimators': [100, 120, 150, 180, 200]
             }
grid_rf = GridSearchCV(reg_rf, param_grid, n_jobs=-1, cv=10, verbose=2)
grid_rf.fit(X, y)

best_est_rf = grid_rf.best_estimator_
best_par_rf = grid_rf.best_params_
grid_cvs_rf = np.max(grid_rf.cv_results_['mean_test_score'])

#cvs_rf = np.mean(cross_val_score(reg_rf, X, y, cv=10))
reg_rf = RandomForestRegressor(**best_par_rf).fit(X, y)
fn_rf = "models/rf_{}.joblib".format(var)
joblib.dump(reg_rf, fn_rf)
# fn_rf = joblib.load(fn_rf)


#%% xgboost
reg_xg = xgb.XGBRegressor()
# xg_reg.fit(X_train, y_train)
# y_pred = xg_reg.predict(X_test)
# Cross validation score
param_grid = {
    'max_depth': range(2, 10, 2),
    'n_estimators': range(50, 200, 50),
    'booster': ['gbtree', 'dart']
}
grid_xg = GridSearchCV(reg_xg, param_grid, n_jobs=-1, cv=10, verbose=2).fit(X, y)

best_est_xg = grid_xg.best_estimator_
best_par_xg = grid_xg.best_params_
grid_cvs_xg = np.max(grid_xg.cv_results_['mean_test_score'])

# cvs_xg = np.mean(cross_val_score(reg_xg, X, y, cv=10, verbose=2))

reg_xg = xgb.XGBRegressor(**grid_xg.best_params_).fit(X, y)
fn_xg = "models/xg_{}.joblib".format(var)
joblib.dump(reg_xg, fn_xg)


#%% LightGBM
reg_lg = lgb.LGBMRegressor()

# USING GRID SEARCH
param_lg = {
    'num_leaves': [10*i for i in range(1, 10)]
}

grid_lg = GridSearchCV(estimator=reg_lg, param_grid=param_lg, verbose=2, cv=10, n_jobs=-1).fit(X, y)

best_est_lg = grid_lg.best_estimator_
best_par_lg = grid_lg.best_params_
grid_cvs_lg = np.max(grid_lg.cv_results_['mean_test_score'])

reg_lg = lgb.LGBMRegressor(**grid_lg.best_params_).fit(X, y)
# cvs_lg = np.mean(cross_val_score(reg_lg, X, y, cv=10, verbose=2))

fn_lg = "models/lg_{}.joblib".format(var)
joblib.dump(reg_lg, fn_lg)


# %% CatBoost
reg_cb = CatBoostRegressor()

param_cb = {'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'depth': [6, 8, 10, 12],
            'l2_leaf_reg': [1, 3, 5]}
grid_cb = GridSearchCV(estimator=reg_cb, param_grid=param_cb, verbose=2, cv=10, n_jobs=-1).fit(X, y)

best_est_cb = grid_cb.best_estimator_
best_par_cb = grid_cb.best_params_
grid_cvs_cb = np.max(grid_cb.cv_results_['mean_test_score'])

reg_cb = CatBoostRegressor(**grid_cb.best_params_).fit(X, y)

fn_cb = "models/cb_{}.joblib".format(var)
joblib.dump(reg_cb, fn_cb)

# %%
cvs = {'rf': grid_cvs_rf, 'xg': grid_cvs_xg, 'lg': grid_cvs_lg, 'cb': grid_cvs_cb}
fn_cvs = "results/cvs_{}.txt".format(var)
with open(fn_cvs, 'w') as f:
    print(cvs, file=f)

#%% linear regression
lm = LinearRegression()
cvs_lr = np.mean(cross_val_score(lm, X, y, cv=10))
