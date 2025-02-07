import numpy as np
import copy
import joblib
import time

#%%
def sobol(cost, d, n, sample_inputs):
    """
    Compute Sobol sensitivity indices.

    Args:
        cost (function): Cost function to evaluate.
        d (int): Number of input dimensions.
        n (int): Number of samples.
        sample_inputs (function): Function to sample input variables.

    Returns:
        tuple: (First-order indices, total-order indices)
    """
    X_A = sample_inputs(n)
    X_B = sample_inputs(n)
    X_R = sample_inputs(n)

    y_A = cost(X_A).reshape((n, 1))
    y_B = cost(X_B).reshape((1, n))
    Y = cost(X_R)
    varY = np.var(Y)

    S_i = np.zeros(d)
    S_Ti = np.zeros(d)
    for i in range(d):
        # for i in range(2,d): #for three parameters
        X_A_Bi = copy.deepcopy(X_A)
        X_A_Bi.iloc[:, i] = X_B.iloc[:, i]
        y_A_Bi = cost(X_A_Bi).reshape((n, 1))
        S_i[i] = np.dot(y_B, y_A_Bi - y_A) / n / varY
        S_Ti[i] = np.sum((y_A - y_A_Bi) ** 2) / 2 / n / varY

    return S_i, S_Ti


#%%
def sobol_analysis(model_path, n, d, sample_inputs):
    """
    Perform Sobol sensitivity analysis using the Sobol indices method.

    Args:
        model_path (str): Path to the saved model (joblib file).
        n (int): Number of samples for analysis.
        d (int): Number of input dimensions.
        sample_inputs (function): Function to sample input variables.

    Returns:
        dict: Sobol indices, elapsed time, and sample size.
    """
    # Load the model
    cb_rg = joblib.load(model_path)

    # Cost function for predictions
    def cost(x):
        return cb_rg.predict(x)

    # Measure time
    t_start = time.time()
    S_i, S_Ti = sobol(cost, d, n, sample_inputs)
    elapsed_time = time.time() - t_start

    # Return results as a dictionary
    return {'Si': S_i.tolist(), 'STi': S_Ti.tolist(), 'Time': elapsed_time, 'n': n}