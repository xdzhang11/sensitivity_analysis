import argparse
import pandas as pd
import os
## task train_metamodels
from functions.train_metamodels import train_and_save_metamodels
## task feature_importance
from functions.feature_importance import compute_and_plot_shap
## task sobol
from functions.sobol_indices import sobol_analysis
from functions.sample_generation import x_all
## task gsa_theoretical
from functions.shapley_analysis import run_shapley_iec

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Run feature importance, model training, or global sensitivity analysis.")
    parser.add_argument("task", choices=["train_metamodels", "feature_importance", "sobol", "shapley_iec"], help="Task to execute")
    args = parser.parse_args()

    # Define common parameters
    varlist = ['bTD', 'Mx_blade', 'Mx_tower']

    # Load dataset
    if args.task in ["train_metamodels", "feature_importance"]:
        filename_Xy = os.path.join('data', 'data_dtu10mwt.h5')  # intentionally not uploaded
        df = pd.read_hdf(filename_Xy, 'Xy')
        X = df.iloc[:, 0:5]

    for var in varlist:
        # Execute tasks
        if args.task == "train_metamodels":
            print(f"\nTraining metamodels for target: {var}")
            y = df[var]
            train_and_save_metamodels(X, y, var)

        elif args.task == "feature_importance":
            feature_names = [r'$u$', r'$\sigma$', r'$C_L$', 'blade Ix', 'tower Ix']
            output_path = 'figures'
            print(f"Running  feature importance for {var}...")
            model_path = f"models/cb_{var}.joblib"  # CatBoost model
            compute_and_plot_shap(X, model_path, var, feature_names, output_path)

        elif args.task == "sobol":
            print(f"\nRunning Sobol analysis for {var}...")
            model_path = f"models/cb_{var}.joblib"
            sobol_results = sobol_analysis(model_path, n=int(5e7), d=5, sample_inputs=x_all)
            # Save results
            result_file = f"results/sobol_{var}.txt"
            with open(result_file, 'w') as f:
                f.write(str(sobol_results))
            print(f"Sobol results saved to {result_file}")

        elif args.task == "shapley_iec":
            print(f"Running Shapley effects analysis with IEC distributions for {var}...")
            model_path = f"models/cb_{var}.joblib"
            Nv = 1000000  # MC sample size to estimate var(Y)
            Ni = 100  # sample size for inner loop
            No = 10  # sample size for outer loop
            m = 10000
            run_shapley_iec(Nv, Ni, No, m, model_path, var)
            print(f"Shapley effects analysis completed for {var}")

if __name__ == "__main__":
    main()
