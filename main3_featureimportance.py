import joblib
import time
import shap
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from catboost import Pool
from sklearn.model_selection import train_test_split
plt.rcParams["figure.figsize"] = (5, 6)


#%%
filename_Xy = os.path.join('data', 'data_dtu10mwt.h5')
df = pd.read_hdf(filename_Xy, 'Xy')
X = df.iloc[:, 0:5]

varlist = ['bTD', 'Mx_blade', 'Mx_tower']

for k in range(3):
    var = varlist[k]
    y = df[var]  # blade tip clearance

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    fn_rf = "models/cb_{}.joblib".format(var)
    cb_rg = joblib.load(fn_rf)


    #%% Random forests feature importance
    cb_fi = cb_rg.get_feature_importance()
    fn_cb_fi = "results/cb_fi_{}.txt".format(var)
    np.savetxt(fn_cb_fi, cb_fi, fmt='%.3f')

    bars = ['$u$', '$\sigma$', 'cl', 'blade Ix', 'tower Ix']
    # y_pos = np.arange(len(bars))
    # # Create bars
    # plt.bar(y_pos, cb_fi)
    # plt.xticks(y_pos, bars)
    # plt.title('catBoost feature importance')
    # plt.show()



    #%% shap package

    # X_test = X.iloc[:1000, :]
    explainer = shap.TreeExplainer(cb_rg)
    shap_values = explainer.shap_values(X_test)

    fig, ax = plt.subplots(1, 1)
    shap.summary_plot(shap_values, X_test, feature_names=bars, show=False)
    plt.savefig("Figures/shap1_{}.pdf".format(var), format="pdf", bbox_inches="tight")
    plt.show()

    shap.summary_plot(shap_values, X_test, feature_names=bars, plot_type="bar",show=False)
    plt.savefig("Figures/shap2_{}.pdf".format(var), format="pdf", bbox_inches="tight")
    plt.show()






