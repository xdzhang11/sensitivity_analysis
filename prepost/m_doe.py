# %% Installation
# pip install --upgrade pip
# pip install qmcpy==1.2 --quiet
# conda install -c pytorch pytorch
# pip install tensorflow
# pip install --upgrade tensorflow-probability

# %% Import packages
from pickle import FALSE
# import qmcpy  # we import the environment at the start to use it
import numpy as np  # basic numerical routines in Python
import time  # timing routines
import warnings  # to suppress warnings when needed
# import torch  # only needed for PyTorch Sobol' backend
import matplotlib.pyplot as plt  # plotting
# import tensorflow as tf
# import tensorflow_probability as tfp
import pandas as pd

from scipy.stats import qmc
# from torch.quasirandom import SobolEngine
from matplotlib import style
from numpy.random import rand

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams['text.usetex'] = True


#%%

# #%% defaut figure setting
style.use('seaborn-white')
plt.rc('font', size=16)  #set defaults so that the plots are readable
plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=16)
plt.rc('figure', titlesize=16)
pt_clr='bgkcmyr' #plot colors

# %% parametric setting
n = 512
d = 2
n_rows = 3
n_cols = 3
fig,ax = plt.subplots(nrows=n_rows,ncols=n_cols,figsize=(5*n_cols,5*n_rows))
qmc_dis = pd.DataFrame(index=['Sobol','Halton'], columns=['qmcpy','scipy','tensorflow'])

# %% Random point
X = rand(n,d)
# qmc_dis[0,0]=qmc.discrepancy(X)
ax[0,0].scatter(X[:,0],X[:,1],color=pt_clr[1])
ax[0,0].set_title("Numpy random")

# %% Latin Hypercube Sampling
sampler = qmc.LatinHypercube(d)
X = sampler.random(n)
# qmc_dis[1,0]=qmc.discrepancy(X)
ax[0,1].scatter(X[:,0],X[:,1],color=pt_clr[1])
ax[0,1].set_title("Latin Hypercube (Scipy)")


# %% Sobol's qmcpy
sobol = qmcpy.Sobol(d)
print(sobol)
X = sobol.gen_samples(n)
qmc_dis.iloc[0,0]=qmc.discrepancy(X)
ax[1,0].scatter(X[:,0],X[:,1],color=pt_clr[1])
ax[1,0].set_title("Sobol (qmcpy)")

# %% Sobol Scipy
sampler = qmc.Sobol(d=2,scramble=False)
X = sampler.random_base2(m=int(np.log2(n)))
qmc_dis.iloc[0,1]=qmc.discrepancy(X)
ax[1,1].scatter(X[:,0],X[:,1],color=pt_clr[1])
ax[1,1].set_title("Sobol (Scipy)")


# %% Sobol Tensorflow
X = tf.math.sobol_sample(d,n)
# X = tf.math.sobol_sample(d,n, skip=int(1e3))
qmc_dis.iloc[0,2]=qmc.discrepancy(X)
ax[1,2].scatter(X[:,0],X[:,1],color=pt_clr[1])
ax[1,2].set_title("Sobol (Tensorflow)")

# #%% Sobol PyTorch SobolEngine
# soboleng = SobolEngine(dimension=d)
# X = soboleng.draw(n).numpy()
# ax[3].scatter(X[:,0],X[:,1],color=pt_clr[1])
# ax[3].set_title("Sobol (PyTorch)")
#%%

# %% Halton Qmcpy
halton = qmcpy.Halton(d)
print(halton)
X = halton.gen_samples(n)
qmc_dis.iloc[1,0]=qmc.discrepancy(X)
ax[2,0].scatter(X[:,0],X[:,1],color=pt_clr[1])
ax[2,0].set_title("Halton (qmcpy)")

# %% Halton Scipy
sampler = qmc.Halton(d,scramble=False)
X = sampler.random(n)
qmc_dis.iloc[1,1]=qmc.discrepancy(X)
ax[2,1].scatter(X[:,0],X[:,1],color=pt_clr[1])
ax[2,1].set_title("Halton (Scipy)")


# %% Halton tensorflow_probability
sequence_indices = tf.range(start=0, limit= n,
                            dtype=tf.int32)
# X = tfp.mcmc.sample_halton_sequence(
#     dim=d, sequence_indices=sequence_indices, randomized=False)
X = tfp.mcmc.sample_halton_sequence(
    dim=d, num_results=n, sequence_indices=None, dtype=tf.float32, randomized=False)
qmc_dis.iloc[1,2]=qmc.discrepancy(X)
ax[2,2].scatter(X[:,0],X[:,1],color=pt_clr[1])
ax[2,2].set_title("Halton (Tensorflow)")


# generating extra points
# _ = sampler.fast_forward(n)
# sample_continued = sampler.random(n=n1)

plt.show()
print(qmc_dis)


# %%---------------------------Plot for paper----------------------------------------------
# Compare random with Halton sequence
# %% parametric setting
n = 1000
d = 2
qmc_dis_r = []
qmc_dis_h = []
for k in range(100):
    X_r = rand(n, d)
    qmc_dis_r.append(qmc.discrepancy(X_r))
    sampler = qmc.Halton(d)
    X_h = sampler.random(n)
    qmc_dis_h.append(qmc.discrepancy(X_h))

qmc_dis_r_m = np.mean(qmc_dis_r)
qmc_dis_h_m = np.mean(qmc_dis_h)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5*2, 4))
ax[0].scatter(X_r[:, 0], X_r[:, 1], color=pt_clr[1])
ax[0].set_title("Random",fontsize=16)
ax[1].scatter(X_h[:, 0], X_h[:, 1], color=pt_clr[1])
ax[1].set_title("Halton",fontsize=16)
ax[0].tick_params(axis='x', labelsize=14) # Change '10' to your desired font size for x-axis
ax[0].tick_params(axis='y', labelsize=14)
ax[1].tick_params(axis='x', labelsize=14) # Change '10' to your desired font size for x-axis
ax[1].tick_params(axis='y', labelsize=14)
plt.savefig("Figures/qms.pdf", format="pdf", bbox_inches="tight")
plt.show()

