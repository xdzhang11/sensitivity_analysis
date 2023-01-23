import numpy as np
from scipy.stats import norm, halfnorm,weibull_min
import pandas as pd
from cmath import pi
import json
from m_usigma import rvs_u


fn_wb = "results/sigma_wb_pars.txt"
wb_pars = json.load(open(fn_wb))

def x_all(n):
    d = 5
    x_v = pd.DataFrame(data=np.zeros((n, d)), columns=['wsp', 'sigma', 'cl', 'bladeIx', 'towerIx'])
    x_v.wsp = rvs_u(n)
    x_v.sigma = weibull_min.rvs(size=n, c = wb_pars['c'], loc=wb_pars['loc'], scale=wb_pars['scale'])

    # half normal distribution parameters
    mu_hn = 1
    scale_hn = 0.05/np.sqrt(1-2/pi)
    x_v.cl = 2-halfnorm.rvs(size=n, loc=mu_hn, scale=scale_hn)
    # x_v.cl = norm.ppf(M[:, 2], loc=1, scale=0.05)
    x_v.bladeIx = norm.rvs(size=n, loc=1, scale=0.05)
    x_v.towerIx = norm.rvs(size=n, loc=1, scale=0.05)
    return x_v
