import numpy as np
import matplotlib.pyplot as plt
from functions.shapley_effects import shapley_random as shapley
from functions.sample_generation import X_dep_a, X_j_a

def run_gsa_theoretical(Nv, Ni, No, m):
    """
    Run Shapley effects analysis for the theoretical case with Gaussian inputs.
    Nv   # MC sample size to estimate var(Y)
    Ni   # sample size for inner loop
    No   # sample size for outer loop
    m = 1000000
    """
    def cost(x):
        beta = np.array([1, 1, 1]).transpose()
        return x.dot(beta)

    d = 3       # dimension of inputs

    rho23_all = np.linspace(-0.99, 0.99, 20)
    s1, s2, s3 = [], [], []

    for rho in rho23_all:
        Sh = shapley(cost, d, Nv, Ni, No, m, X_dep_a, X_j_a, rho)
        s1.append(Sh[0])
        s2.append(Sh[1])
        s3.append(Sh[2])


    plt.rcParams["figure.figsize"] = (5, 4)
    fig, ax = plt.subplots(1, 1)

    plt.scatter(rho23_all, s1, label="$x_1$ sh")
    plt.scatter(rho23_all, s2, label="$x_2$ sh")
    plt.scatter(rho23_all, s3, label="$x_3$ sh")

    sigma_s = 6 + 4 * rho23_all
    Sh1 = sigma_s**(-1)
    plt.plot(rho23_all, Sh1, label="$x_1$")

    Sh2 = (1 + rho23_all * 2 + 1.5 * rho23_all**2) * sigma_s**(-1)
    plt.plot(rho23_all, Sh2, label="$x_2$")

    Sh3 = (4 + rho23_all * 2 - 1.5 * rho23_all**2) * sigma_s**(-1)
    plt.plot(rho23_all, Sh3, label="$x_3$")

    plt.xlabel(r"$\rho_{X_2,X_3}$")
    plt.legend()
    plt.savefig("figures/sh_gaussian_test.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    plt.close()
    print("Shapley effects analysis (theoretical case) completed and figure saved.")
