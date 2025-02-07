import argparse
from functions.gsa_theoretical import run_gsa_theoretical

def main():
    # Argument parser setup for future extensibility
    parser = argparse.ArgumentParser(description="Run theoretical Shapley effects analysis.")
    parser.add_argument("task", choices=["gsa"], help="Task to execute")
    args = parser.parse_args()

    if args.task == "gsa":
        print("Running Shapley effects analysis for the theoretical case...")
        Nv = 1000000  # MC sample size to estimate var(Y)
        Ni = 3  # sample size for inner loop
        No = 1  # sample size for outer loop
        m = 1000000
        run_gsa_theoretical(Nv, Ni, No, m)

if __name__ == "__main__":
    main()
