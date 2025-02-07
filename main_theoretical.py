import argparse
from functions.gsa_theoretical import run_gsa_theoretical

def main():
    # Argument parser setup for future extensibility
    parser = argparse.ArgumentParser(description="Run theoretical Shapley effects analysis.")
    parser.add_argument("task", choices=["gsa"], help="Task to execute")
    args = parser.parse_args()

    if args.task == "gsa":
        print("Running Shapley effects analysis for the theoretical case...")
        run_gsa_theoretical()

if __name__ == "__main__":
    main()
