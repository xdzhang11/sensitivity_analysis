import joblib
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions.shapley_effects import shapley_random as shapley
## run_shapley_iec
from functions.sample_generation import X_dep_us
from functions.sample_generation import X_j_us
## run_shapley_nataf
from functions.sample_generation import X_dep_wt
from functions.sample_generation import X_j_wt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams['text.usetex'] = True

plt.rcParams['axes.labelsize'] = 18  # Font size for x and y labels
plt.rcParams['xtick.labelsize'] = 18 # Font size for x ticks
plt.rcParams['ytick.labelsize'] = 18  # Font size for y ticks
plt.rcParams['legend.fontsize'] = 18  # Font size for legend


## Shapley effects with wind parameters follows distribution in IEC
def run_shapley_iec(Nv, Ni, No, m, model_path, var):
    """
    Perform Shapley effects analysis for a specific variable using wind parameter joint distributions from IEC.
    Args:
        Nv   # MC sample size to estimate var(Y)
        Ni   # sample size for inner loop
        No   # sample size for outer loop
        m    # number of random run
        model_path   # path to model 
        var (str): The target variable to analyze.
    """

    meta_model = joblib.load(model_path)

    def cost(x):
        return meta_model.predict(x)

    # Define parameters for the Shapley analysis
    d = 5        # dimension of inputs

    # Perform Shapley effects analysis
    t_start = time.time()
    SH = shapley(cost, d, Nv, Ni, No, m, X_dep_us, X_j_us, 0.2)
    elapsed_time = time.time() - t_start

    # Save results
    result = {'SH': SH, 'Time': elapsed_time, 'Nv': Nv, 'Ni': Ni, 'No': No, 'm': m}
    result_file = f"results/sh_iec_{var}.txt"
    with open(result_file, 'w') as f:
        f.write(str(result))
    print(f"Results saved to {result_file}")

    # Plotting the Shapley effects
    feature_names = [r'$u$', r'$\sigma$', r'$C_L$', 'blade Ix', 'tower Ix']
    y_pos = np.arange(len(feature_names))
    plt.bar(y_pos, SH)
    plt.xticks(y_pos, feature_names)
    plt.title(f'Shapley Effects for {var}')
    plt.savefig(f"figures/shapley_iec_{var}.pdf", format="pdf", bbox_inches="tight")
    plt.close()

#%% Shapley effects with wind parameters with Nataf transformation
def run_shapley_nataf(Nv, Ni, No, m, model_path, var):
    """
    Perform Shapley effects analysis for a specific variable using wind parameter joint distributions with Nataf transformation.
    Args:
        Nv, Ni, No, m: Monte Carlo sampling parameters
        model_path (str): Path to the trained metamodel
        var (str): The target variable to analyze
    """
    # Load metamodel
    meta_model = joblib.load(model_path)

    def cost(x):
        return meta_model.predict(x)

    # Compute Shapley effects for different correlation coefficients
    cs = np.linspace(-0.98, 0.98, 8)
    SHs = []
    t_start = time.time()
    for c in cs:
        SHs.append(shapley(cost, d=5, Nv=Nv, Ni=Ni, No=No, m=m, X_dep=X_dep_wt, X_j=X_j_wt, rho=c))
    elapsed_time = time.time() - t_start

    # Save the results
    result_file = f"results/sh_nataf_{var}.txt"
    data = np.stack(SHs)
    np.savetxt(result_file, data, fmt='%.5f')
    print(f"Shapley effects saved to {result_file}")

    # Plotting the results
    plot_shapley_nataf_results(data, cs, var)
    print(f"Elapsed time for {var}: {elapsed_time:.2f} seconds")


def plot_shapley_nataf_results(data, cs, var):
    """
    Plot Shapley effects results for a variable using Nataf transformation.
    """
    cs = np.insert(cs, 4, 0)  # Insert 0 into the correlation coefficients
    df = pd.DataFrame(data=data, index=cs, columns=['wsp', 'sigma', 'cl', 'bladeIx', 'towerIx'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 3))

    # Plot wsp and sigma
    axes[0].plot(df['wsp'], '--', lw=3, alpha=0.9, label='$u$')
    axes[0].plot(df['sigma'], '-.', lw=3, alpha=0.9, label='$\sigma$')
    axes[0].set_xlabel(r"Correlation coefficient $\rho_{\mu,\sigma}$")
    axes[0].set_ylabel("Shapley effects")
    axes[0].legend(loc='upper right', fontsize=14)

    # Plot cl, bladeIx, and towerIx
    axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axes[1].plot(df['cl'], '--', lw=3, alpha=0.9, label='$C_L$')
    axes[1].plot(df['bladeIx'], '-.', lw=3, alpha=0.9, label='blade Ix')
    axes[1].plot(df['towerIx'], ':', lw=3, alpha=0.9, label='tower Ix')
    axes[1].set_xlabel(r"Correlation coefficient $\rho_{\mu,\sigma}$")
    axes[1].set_ylabel("Shapley effects")
    tick_values = [0, 0.5e-2, 1e-2, 1.5e-2]
    axes[1].set_yticks(tick_values)
    axes[1].legend(loc='upper right', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"figures/shap_nataf_{var}.pdf", format="pdf", bbox_inches="tight")
    plt.close()