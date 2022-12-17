import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rayleigh, weibull_min
from scipy.integrate import quad
import time


#%% Validation of Rayleigh pdf
def pdf_rayleigh(x, loc=0, scale=1):
    x = (x-loc)/scale
    y = x*np.exp(-x**2/2)/scale
    return y


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rayleigh.html
fig, ax = plt.subplots(1, 1)
x = np.linspace(rayleigh.ppf(0.01, loc=1, scale=2),
                rayleigh.ppf(0.99, loc=1, scale=2), 100)
ax.plot(x, rayleigh.pdf(x, loc=1, scale=2), 'r-', lw=2, alpha=0.6, label='rayleigh pdf')
ax.plot(x, pdf_rayleigh(x, loc=1, scale=2), 'b--', lw=2, alpha=0.6, label='pdf_rayleigh')
plt.legend()
plt.show()


#%%
def pdf_weibull(x, c, loc=0, scale=1):
    x = (x-loc)/scale
    y = c*x**(c-1)*np.exp(-x**c)/scale
    return y


fig, ax = plt.subplots(1, 1)
c = 1.79
x = np.linspace(weibull_min.ppf(0.01, c, loc=1, scale=2),
                weibull_min.ppf(0.99, c, loc=1, scale=2), 100)
ax.plot(x, weibull_min.pdf(x, c, loc=1, scale=2), 'r-', lw=5, alpha=0.6, label='weibull_min pdf')
ax.plot(x, pdf_weibull(x, c, loc=1, scale=2), 'b--', lw=2, alpha=0.6, label='pdf_weibull')
plt.legend()
plt.show()



#%%
def pdf_sigma(s):
    def integrand(x, s):
        v_ave = 8.5
        r_scale = np.sqrt(2 / np.pi) * v_ave
        z1 = pdf_rayleigh(x, loc=0, scale=r_scale)
        # weibull distribution
        wb_shape = 0.27 * x + 1.4
        Iref = 0.12
        wb_scale = Iref * (0.75 * x + 3.3)
        z2 = pdf_weibull(s, wb_shape, loc=0, scale=wb_scale)
        return z1*z2
    t = quad(integrand,0,50,args=s)[0]
    return t



#%% probability density function of sigma
def pdf_sigma2(x):
    v_ave = 8.5
    r_scale = np.sqrt(2 / np.pi) * v_ave
    n = int(1e6)
    v_r = rayleigh.rvs(loc=0, scale=r_scale, size = n)
    wb_shape = 0.27 * v_r + 1.4
    Iref = 0.12
    wb_scale = Iref * (0.75 * v_r + 3.3)
    t = np.mean(weibull_min.pdf(x, wb_shape, 0, wb_scale))
    return t



#%% Acceptance-rejection sampling
t = time.time()# -*- coding: utf-8 -*-
rv = []
for k in range(10000):
    while True:
        x_t = np.random.uniform(low=0, high=4.5)
        y_t = np.random.uniform(low=0, high=0.8)
        if y_t <= pdf_sigma(x_t):
            rv.append(x_t)
            break

elapsed = time.time() - t
elapsed

#%%
x = np.linspace(0, 5, 100)
pdf_s = []
for k in range(100):
    pdf_s.append(pdf_sigma(x[k]))

plt.plot(x, pdf_s)
plt.hist(rv, 50, density=True)

plt.show()

#%%
n = int(1e4)
xy_min = [0, 0]
xy_max = [5, max(pdf_s)]
rs = np.random.uniform(low=xy_min, high=xy_max, size=(n, 2))
m = 0
for k in range(n):
    if rs[k, 1] < pdf_sigma(rs[k, 0]):
        m = m+1

area = m/n*max(pdf_s)*5


#%% probability density function of u conditional on sigma
def pdf_us(x, sigma):
    v_ave = 8.5
    r_scale = np.sqrt(2 / np.pi) * v_ave
    p_u = rayleigh.pdf(x, loc=0, scale=r_scale)
    wb_shape = 0.27 * x + 1.4
    Iref = 0.12
    wb_scale = Iref * (0.75 * x + 3.3)
    p_su = weibull_min.pdf(sigma, wb_shape, 0, wb_scale)
    p_s = pdf_sigma(sigma)
    t = p_su*p_u/p_s
    return t

#%% plot pdf
sigma = 2.5
x = np.linspace(3, 27, 100)
pdf_s = []
for k in range(100):
    pdf_s.append(pdf_us(x[k], sigma))



#%% Check area under pdf
n = int(1e6)
xy_min = [0, 0]
xy_max = [40, max(pdf_s)]
rs = np.random.uniform(low=xy_min, high=xy_max, size=(n, 2))
m = 0
pdf_v = pdf_us(rs[:, 0], sigma)

for k in range(n):
    if rs[k, 1] < pdf_v[k]:
        m = m+1

area = m/n*max(pdf_s)*40

#%% sampling
t = time.time()# -*- coding: utf-8 -*-
rv = []
for k in range(10000):
    while True:
        x_t = np.random.uniform(low=3, high=27)
        y_t = np.random.uniform(low=0, high=max(pdf_s))
        if y_t <= pdf_us(x_t,sigma):
            rv.append(x_t)
            break

elapsed = time.time() - t
elapsed


#%%
plt.plot(x, pdf_s)
plt.hist(rv, 50, density=True)

plt.show()



#%% Interval of sigma

u = np.linspace(3,27,100)
sigma_min = np.maximum(0, 0.1*(u-20))
sigma_max = 0.18*(6.8+0.75*u)
plt.plot(u, sigma_min)
plt.plot(u, sigma_max)
plt.show()