from scipy.stats import norm, rayleigh, weibull_min, halfnorm
import numpy as np
import json
from cmath import pi
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
from m_usigma import rvs_j
from scipy import stats

#%% Correlation coefficient of IEC
u,sigma = rvs_j(1e8)
#cov = np.cov(u, sigma)
res = stats.pearsonr(u, sigma)
#%%
fn_wb = "results/sigma_wb_pars.txt"
wb_pars = json.load(open(fn_wb))

v_ave = 8.5
r_scale = np.sqrt(2 / np.pi) * v_ave
mu_i = rayleigh.mean(loc=0, scale=r_scale)
mu_j = weibull_min.mean(c=wb_pars['c'], loc=wb_pars['loc'], scale=wb_pars['scale'])

sigma_i = rayleigh.std(loc=0, scale=r_scale)
sigma_j = weibull_min.std(c=wb_pars['c'], loc=wb_pars['loc'], scale=wb_pars['scale'])

xmin = -5
xmax = 5
ymin = -5
ymax = 5

rho_zs = np.linspace(0.998e-5, 1, num=10)
rho_xs = []


for rho_z in rho_zs:
    def f_nat(x, y):
        t1 = norm.cdf(x)
        m1 = rayleigh.ppf(t1, loc=0, scale=r_scale)
        t2 = norm.cdf(rho_z*x + np.sqrt(1 - rho_z**2)*y)
        m2 = weibull_min.ppf(t2, c=wb_pars['c'], loc=wb_pars['loc'], scale=wb_pars['scale'])
        z = m1*m2*np.exp(-(x**2 + y**2) / 2)
        return z

    q = dblquad(f_nat, xmin, xmax, ymin, ymax)[0]
    rho_x = -mu_i*mu_j/sigma_i/sigma_j + q/2/pi/sigma_i/sigma_j
    rho_xs.append(rho_x)


fig, ax = plt.subplots(1, 1)
plt.plot(rho_xs, rho_zs)
#plt.legend()
#plt.savefig("Figures/pdf_sigma_u.pdf", format="pdf", bbox_inches="tight")
plt.show()



