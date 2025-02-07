import joblib
import time
import numpy as np
import matplotlib.pyplot as plt
from functions.shapley_effects import shapley_random as shapley
from functions.sample_generation import X_dep_us as X_dep
from functions.sample_generation import X_j_us as X_j

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
    SH = shapley(cost, d, Nv, Ni, No, m, X_dep, X_j, 0.2)
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
