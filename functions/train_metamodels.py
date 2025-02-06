import joblib
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression

from main1_metamodel import cvs_lr


def train_and_save_metamodels(X, y, var):
    # Random Forest
    print("Training Random Forest...")
    reg_rf = RandomForestRegressor()
    param_grid_rf = {'n_estimators': [100, 120, 150, 180, 200]}
    grid_rf = GridSearchCV(reg_rf, param_grid_rf, n_jobs=-1, cv=10, verbose=2).fit(X, y)
    joblib.dump(grid_rf.best_estimator_, f"models/rf_{var}.joblib")

    # XGBoost
    print("Training XGBoost...")
    reg_xg = xgb.XGBRegressor()
    param_grid_xg = {'max_depth': range(2, 10, 2), 'n_estimators': range(50, 200, 50), 'booster': ['gbtree', 'dart']}
    grid_xg = GridSearchCV(reg_xg, param_grid_xg, n_jobs=-1, cv=10, verbose=2).fit(X, y)
    joblib.dump(grid_xg.best_estimator_, f"models/xg_{var}.joblib")

    # LightGBM
    print("Training LightGBM...")
    reg_lg = lgb.LGBMRegressor()
    param_grid_lg = {'num_leaves': [10 * i for i in range(1, 10)]}
    grid_lg = GridSearchCV(reg_lg, param_grid_lg, n_jobs=-1, cv=10, verbose=2).fit(X, y)
    joblib.dump(grid_lg.best_estimator_, f"models/lg_{var}.joblib")

    # CatBoost
    print("Training CatBoost...")
    reg_cb = CatBoostRegressor(verbose=0)  # Suppress CatBoost logging
    param_grid_cb = {'learning_rate': [0.01, 0.03, 0.05, 0.1], 'depth': [6, 8, 10, 12], 'l2_leaf_reg': [1, 3, 5]}
    grid_cb = GridSearchCV(reg_cb, param_grid_cb, n_jobs=-1, cv=10, verbose=2).fit(X, y)
    joblib.dump(grid_cb.best_estimator_, f"models/cb_{var}.joblib")

    # Linear Regression
    print("Training Linear Regression...")
    reg_lr = LinearRegression()
    cvs_lr = np.mean(cross_val_score(reg_lr, X, y, cv=10, n_jobs=-1))
    reg_lr.fit(X, y)
    joblib.dump(reg_lr, f"models/lr_{var}.joblib")

    # Save cross-validation results (for evaluation purposes)
    print("Saving cross-validation results...")
    cross_val_scores = {
        'rf': np.max(grid_rf.cv_results_['mean_test_score']),
        'xg': np.max(grid_xg.cv_results_['mean_test_score']),
        'lg': np.max(grid_lg.cv_results_['mean_test_score']),
        'cb': np.max(grid_cb.cv_results_['mean_test_score']),
        'lr': cvs_lr
    }
    with open(f"results/cvs_{var}.txt", 'w') as f:
        f.write(str(cross_val_scores))

    print(f"Training for {var} completed.\n")
