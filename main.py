import argparse
import pandas as pd
import os
from functions.feature_importance import compute_and_plot_shap
# from functions.gsa_shapley import compute_shapley_effects, plot_shapley_results


def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Run feature importance or global sensitivity analysis.")
    parser.add_argument("task", choices=["feature_importance", "gsa"], help="Task to execute")
    args = parser.parse_args()

    # Define common parameters
    varlist = ['bTD', 'Mx_blade', 'Mx_tower']

    if args.task == "feature_importance":
        print("Running feature importance ...")

        # Load dataset
        filename_Xy = os.path.join('data', 'data_dtu10mwt.h5')
        df = pd.read_hdf(filename_Xy, 'Xy')
        X = df.iloc[:, 0:5]
        feature_names = ['$u$', '$\sigma$', '$C_L$', 'blade Ix', 'tower Ix']
        output_path = 'figures'

        # Compute SHAP feature importance for each variable
        for var in varlist:
            print(f"Plotting {var}...")
            model_path = f"models/cb_{var}.joblib" # catboost model
            compute_and_plot_shap(X, model_path, var, feature_names, output_path)

    elif args.task == "gsa":
        print("Running global sensitivity analysis task...")

        # # Compute GSA and save results
        # compute_shapley_effects(varlist)
        #
        # # Generate plots for each variable
        # for var in varlist:
        #     plot_shapley_results(var)


if __name__ == "__main__":
    main()
