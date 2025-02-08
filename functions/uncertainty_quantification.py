import matplotlib.pyplot as plt
import joblib
from functions.sample_generation import X_j_us
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams['text.usetex'] = True

plt.rcParams['axes.labelsize'] = 14  # Font size for x and y labels
plt.rcParams['xtick.labelsize'] = 16 # Font size for x ticks
plt.rcParams['ytick.labelsize'] = 16  # Font size for y ticks
plt.rcParams['legend.fontsize'] = 18  # Font size for legend
plt.rcParams['figure.figsize'] = (8, 4)

def get_plot_settings(var):
    """
    Returns variable-specific settings for uncertainty quantification.
    """
    uq_settings = {
        'bTD': {'tick_values': [0, 0.1, 0.2], 'xlabel_text': 'Blade tower clearance (m)'},
        'Mx_blade': {'tick_values': [0.1e-5, 2.0e-5, 4.0e-5], 'xlabel_text': 'Mx blade (kNm)'},
        'Mx_tower': {'tick_values': [0, 0.5e-5, 1.0e-5], 'xlabel_text': 'Mx tower (kNm)'}
    }
    return uq_settings[var]

def plot_histogram(var, model_path, Nv=10000, rho=0.2):
    """
    Generate and save a histogram plot for the given variable and metamodel.
    Args
    ----------
        var(str): The variable being analyzed.
        model_path (str): Path to the trained metamodel.
        Nv: Monte Carlo sample size
        rho: correlation coefficient

    Returns
    -------
        y statistics
        histogram plot saved as a PDF file into the figures folder.
    """


    # Load metamodel
    meta_model = joblib.load(model_path)
    def cost(x):
        return meta_model.predict(x)

    d = 5  # Dimension of inputs

    # Generate samples and predictions
    X_A = X_j_us(Nv, d, rho)
    y = cost(X_A)

    # Calculate statistics
    var_y = np.var(y)
    std_y = np.std(y)
    cov_y = std_y / np.mean(y)

    # Histogram plot
    # Get variable-specific settings
    settings = get_plot_settings(var)
    tick_values = settings['tick_values']
    xlabel_text = settings['xlabel_text']

    fig, ax = plt.subplots(1, 1)
    ax.hist(y, bins=50, density=True)
    ax.set_ylabel('Probability density')

    if xlabel_text:
        ax.set_xlabel(xlabel_text)

    # Set y-tick values
    ax.set_yticks(tick_values)

    plt.tight_layout()
    plt.savefig(f"figures/hist_{var}.pdf", format="pdf", bbox_inches="tight")
    plt.close()  # Close the plot to free memory

    return var_y, std_y, cov_y
