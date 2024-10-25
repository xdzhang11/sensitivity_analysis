import joblib
import time
import shap
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams['text.usetex'] = True

plt.rcParams['axes.labelsize'] = 48 # Font size for x and y labels
plt.rcParams['xtick.labelsize'] = 48 # Font size for x ticks
plt.rcParams['ytick.labelsize'] = 48  # Font size for y ticks
plt.rcParams['legend.fontsize'] = 48  # Font size for legend

# plt.rcParams['figure.figsize'] = (12, 3)


#%%
filename_Xy = os.path.join('data', 'data_dtu10mwt.h5')
df = pd.read_hdf(filename_Xy, 'Xy')
X = df.iloc[:, 0:5]
varlist = ['bTD', 'Mx_blade', 'Mx_tower']

for k in range(3):
    var = varlist[k]
    y = df[var]  # blade tip clearance

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    X_test = X
    fn_rf = "models/cb_{}.joblib".format(var)
    cb_rg = joblib.load(fn_rf)

    # Random forests feature importance
    # cb_fi = cb_rg.get_feature_importance()
    # fn_cb_fi = "results/cb_fi_{}.txt".format(var)
    # np.savetxt(fn_cb_fi, cb_fi, fmt='%.3f')
    #
    bars = ['$u$', '$\sigma$', '$C_L$', 'blade Ix', 'tower Ix']
    # y_pos = np.arange(len(bars))
    # # Create bars
    # plt.bar(y_pos, cb_fi)
    # plt.xticks(y_pos, bars)
    # plt.title('catBoost feature importance')
    # plt.show()

    # shap package
    explainer = shap.TreeExplainer(cb_rg)
    shap_values = explainer.shap_values(X_test)

    plt.subplot(1, 2, 1)

    shap.summary_plot(shap_values, X_test, feature_names=bars, plot_size=(8,2) ,show=False, title=False)
    ax = plt.gca()
    ax.set_xlabel('Shap value')
    # SHAP value (impact on model output)
    plt.subplot(1, 2, 2)
    # Generate the second plot and save it
    shap.summary_plot(shap_values, X_test, feature_names=bars, plot_size=(8,2), plot_type="bar", show=False)
    ax = plt.gca()
    ax.set_xlabel('Mean Shap value')
    # mean(|SHAP value|)(average impact on model output magnitude)
    plt.tight_layout()
    plt.savefig("Figures/shap_feature_{}.pdf".format(var), format="pdf", bbox_inches="tight")
    plt.show()

