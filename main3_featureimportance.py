import joblib
import time
import shap
import numpy as np
import matplotlib.pyplot as plt
from f_shapley import shapley

plt.rcParams["figure.figsize"] = (5, 6)


#%% shap package
from f_X import X_j_us as X_j

X_test = X_j(1000, 5, 0.5)
gbrm = joblib.load("models/randomforests.joblib")

explainer = shap.TreeExplainer(gbrm)
shap_values = explainer.shap_values(X_test)

fig, ax = plt.subplots(1, 1)
shap.summary_plot(shap_values, X_test)

fig, ax = plt.subplots(1, 1)
shap.summary_plot(shap_values, X_test, plot_type="bar")


#%% Random forests feature importance
height = gbrm.feature_importances_
bars = ['wsp', 'ti', 'cl', 'bladeIx', 'towerIx']
y_pos = np.arange(len(bars))
# Create bars
plt.bar(y_pos, height)
plt.xticks(y_pos, bars)
plt.title('Random forests feature importance')
plt.show()





