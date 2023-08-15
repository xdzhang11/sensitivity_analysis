# Pre-processing:
# 1) generate random samples
# 2) generate Hawc2 simulation files to folder
# 3) remove Hawc2 simulation files from folder
# Last update 1/12/2013
# @author: xiazang@dtu.dk
from wtUq import wtUq
import os
import h5py
import pandas as pd


# %% Get random number for variables and turbulence random seed
n = int(1e4)   # number of sample
nSeed = int(12)  # number of seeds
# varList = ["wsp", "ti", "cl", "bladeIx",  "towerIx"]
pref = wtUq(n, nSeed)  # pre function

filename_seed = os.path.join('sample', 'seed_mcs.hdf5')   # turbulence seed
filename_qms = os.path.join('sample', 'qms_mcs.h5')       # quasi mc sample for random variables

# %% ---------------------------------------------Generate random sample------------------------------------------------
# pref.generate_seed(filename_seed)
# pref.generate_qms(varList, filename_qms)

# ----------------------------------------------------------------------------------------------------------------------
# %% Get random sample
f = h5py.File(filename_seed, 'r')
seed = f['seed'][:]
f.close()
qms = pd.read_hdf(filename_qms, 'qms')

varName = "mcs"

# %% generate HAWC2  files
if varName == "wsp":
    pref.generate_wsp(qms, seed)
elif varName == "ti":
    pref.generate_ti(qms, seed)
elif varName == "cl":
    pref.generate_cl(qms, seed)
elif varName == "bladeIx":
    pref.generate_bladeIx(qms, seed)
elif varName == "towerIx":
    pref.generate_towerIx(qms, seed)
elif varName == "mcs":
    pref.generate_mcs(qms, seed)

# %% remove files after simulation is completed
pref.remove_htc()
if varName == "cl":
    pref.remove_pc()
elif varName == "bladeIx":
    pref.remove_bladeSt() 
elif varName == "towerIx":
    pref.remove_towerSt() 
elif varName == "mcs":
    pref.remove_pc()
    pref.remove_bladeSt() 
    pref.remove_towerSt() 

