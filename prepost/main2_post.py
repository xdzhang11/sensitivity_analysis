# Post-processing Hawc2 simulation results to Xy.h5 file
# Last update 1/12/2013
# @author: xiazang@dtu.dk

import pandas as pd
from wtUq import wtUq
import os

n = 10000
nSeed = 12
postf = wtUq(n, nSeed)
folderName = os.path.join('C:', os.sep, 'Simulations', 'DTU10mw')

varName = "mcs"
# varList = ["wsp", "ti", "cl", "cd", "bladeSt", "towerSt", "soil"]
# for varName in varList:
#     res = postf.get_results(folderName, varName)
#     postf.save_figues(varName, res)

res = postf.get_results(folderName, varName)
y = pd.DataFrame(res)
filename_qms = os.path.join('sample', 'qms_mcs.h5')
qms = pd.read_hdf(filename_qms, 'qms')
X = qms.iloc[:n, :]
frames = [X, y]
result = pd.concat(frames, axis=1)
filename_Xy = os.path.join('Xy', 'Xy.h5')
result.to_hdf(filename_Xy, key='Xy', mode='w')
Xy = pd.read_hdf(filename_Xy, 'Xy')


