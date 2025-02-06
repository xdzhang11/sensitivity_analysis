import joblib
import shap
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams['text.usetex'] = True

plt.rcParams['axes.labelsize'] = 48 # Font size for x and y labels
plt.rcParams['xtick.labelsize'] = 48 # Font size for x ticks
plt.rcParams['ytick.labelsize'] = 48  # Font size for y ticks
plt.rcParams['legend.fontsize'] = 48  # Font size for legend

def compute_and_plot_shap(X, model_path, variable_name, feature_names, output_path):
    """
    Computes SHAP values for a given model and plots the feature importance.

    Parameters:
    - X (DataFrame): Features used in the model.
    - model_path (str): Path to the trained model (joblib file).
    - variable_name (str): Variable name for plot filenames.
    - feature_names (list of str): Feature names for SHAP plots.
    - output_path (str): Directory to save the plots.
    """
    # Load model
    model = joblib.load(model_path)

    # SHAP explanations with shap package
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Create plots
    plt.figure()
    plt.subplot(1, 2, 1)
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_size=(8, 2), show=False, title=False)
    ax = plt.gca()
    ax.set_xlabel('Shap value') # SHAP value (impact on model output)

    plt.subplot(1, 2, 2)
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_size=(8, 2), plot_type="bar", show=False)
    ax = plt.gca()
    ax.set_xlabel('Mean Shap value') # mean(|SHAP value|)(average impact on model output magnitude)

    # Save plots
    plt.tight_layout()
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, f"shap_feature_{variable_name}.pdf"), format="pdf", bbox_inches="tight")
    plt.show()

