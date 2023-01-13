# Joint and conditional distributions of u (wind speed) and sigma (standard deviation)
# Last update 1/13/2013
# @author: xiazang@dtu.dk
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rayleigh, weibull_min
from scipy.integrate import quad
import time


#%% Rayleigh pdf and its validation against scipy package
def pdf_rayleigh(x, loc=0, scale=1):
    x = (x-loc)/scale
    y = x*np.exp(-x**2/2)/scale
    return y


# fig, ax = plt.subplots(1, 1)
# x = np.linspace(rayleigh.ppf(0.01, loc=1, scale=2),
#                 rayleigh.ppf(0.99, loc=1, scale=2), 100)
# ax.plot(x, rayleigh.pdf(x, loc=1, scale=2), 'r-', lw=2, alpha=0.6, label='rayleigh pdf')
# ax.plot(x, pdf_rayleigh(x, loc=1, scale=2), 'b--', lw=2, alpha=0.6, label='pdf_rayleigh')
# plt.legend()
# plt.savefig("Figures/test.pdf", format="pdf", bbox_inches="tight")
# plt.show()


#%% Rayleigh pdf and its validation against scipy package
def pdf_weibull(x, c, loc=0, scale=1):
    x = (x-loc)/scale
    y = c*x**(c-1)*np.exp(-x**c)/scale
    return y


# fig, ax = plt.subplots(1, 1)
# c = 1.79
# x = np.linspace(weibull_min.ppf(0.01, c, loc=1, scale=2),
#                 weibull_min.ppf(0.99, c, loc=1, scale=2), 100)
# ax.plot(x, weibull_min.pdf(x, c, loc=1, scale=2), 'r-', lw=5, alpha=0.6, label='weibull_min pdf')
# ax.plot(x, pdf_weibull(x, c, loc=1, scale=2), 'b--', lw=2, alpha=0.6, label='pdf_weibull')
# plt.legend()
# plt.show()


#%% Pdf of sigma, integral of total probability
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
    t = quad(integrand, 0, 50, args=s)[0]
    return t


#%% Pdf of sigma, sampling based integral
# def pdf_sigma2(x):
#     v_ave = 8.5
#     r_scale = np.sqrt(2 / np.pi) * v_ave
#     n = int(1e6)
#     v_r = rayleigh.rvs(loc=0, scale=r_scale, size = n)
#     wb_shape = 0.27 * v_r + 1.4
#     Iref = 0.12
#     wb_scale = Iref * (0.75 * v_r + 3.3)
#     t = np.mean(weibull_min.pdf(x, wb_shape, 0, wb_scale))
#     return t


#%% Acceptance-rejection sampling of sigma
def rvs_s(n):
    rv = []
    for k in range(int(n)):
        while True:
            x_t = np.random.uniform(low=0, high=4.869)  # max value of sigma
            y_t = np.random.uniform(low=0, high=0.8)
            if y_t <= pdf_sigma(x_t):
                rv.append(x_t)
                break
    rv = np.array(rv)
    return rv


#%% Compare pdf with generated sample

# rv = rvs_s(1e5)
# x = np.linspace(0, 5, 100)
# pdf_s = []
# for k in range(100):
#     pdf_s.append(pdf_sigma(x[k]))
#
# fig, ax = plt.subplots(1, 1)
# plt.plot(x, pdf_s, 'r-', lw=2, alpha=0.6, label='pdf')
# plt.hist(rv, 50, density=True, label='sample')
# plt.xlabel('standard deviation of wind speed (m/s)')
# plt.ylabel('probability density')
# plt.legend()
# plt.savefig("Figures/pdf_sigma.pdf", format="pdf", bbox_inches="tight")
# plt.show()
#
# #%% Monte Carlo method, check area under pdf
# n = int(1e5)
# xy_min = [0, 0]
# xy_max = [5, 0.8]
# rs = np.random.uniform(low=xy_min, high=xy_max, size=(n, 2))
# m = 0
# for k in range(n):
#     if rs[k, 1] <= pdf_sigma(rs[k, 0]):
#         m = m+1
#
# area = m/n*0.8*5


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


#%% sampling u conditional on sigma
def rvs_us(n, sigma):
    # find the maximum of pdf
    x = np.linspace(3, 27, 100)
    pdf_s = []
    for k in range(100):
        pdf_s.append(pdf_us(x[k], sigma))

    rv = []
    for k in range(int(n)):
        while True:
            x_t = np.random.uniform(low=3, high=27)
            y_t = np.random.uniform(low=0, high=1.1*max(pdf_s))
            if y_t <= pdf_us(x_t, sigma):
                rv.append(x_t)
                break
    rv = np.array(rv)
    return rv


# #%% check theoretical pdf and generated data
# sigma = 1
# rv = rvs_us(1e5, sigma)
#
# fig, ax = plt.subplots(1, 1)
# x = np.linspace(3, 27, 100)
# pdf_s = []
# for k in range(100):
#     pdf_s.append(pdf_us(x[k],sigma))
#
# plt.plot(x, pdf_s)
# plt.hist(rv, 50, density=True)
#
# plt.show()


# #%% Check area under pdf
# n = int(1e6)
# xy_min = [0, 0]
# xy_max = [40, 0.5]
# rs = np.random.uniform(low=xy_min, high=xy_max, size=(n, 2))
# m = 0
# pdf_v = pdf_us(rs[:, 0], sigma)
#
# for k in range(n):
#     if rs[k, 1] < pdf_v[k]:
#         m = m+1
#
# area = m/n*0.5*40


#%% Generate random variables from joint distribution
def rvs_j(n):
    v_ave = 8.5
    r_scale = np.sqrt(2 / np.pi) * v_ave
    v_u = rayleigh.cdf(27, loc=0, scale=r_scale)
    v_l = rayleigh.cdf(3, loc=0, scale=r_scale)

    r = np.random.uniform(low=v_u, high=v_l, size=int(n))
    u = rayleigh.ppf(r, loc=0, scale=r_scale)
    wb_shape = 0.27 * u + 1.4
    Iref = 0.12
    wb_scale = Iref * (0.75 * u + 3.3)
    sigma = weibull_min.rvs(wb_shape, loc=0, scale=wb_scale)
    sigma_min = np.maximum(0, 0.1 * (u - 20))
    sigma_max = 0.18 * (6.8 + 0.75 * u)
    if n>1:
        for k in range(int(n)):
            while (sigma[k] < sigma_min[k]) or (sigma[k] > sigma_max[k]):
                sigma[k] = weibull_min.rvs(wb_shape[k], loc=0, scale=wb_scale[k])
    else:
        while (sigma < sigma_min) or (sigma > sigma_max):
            sigma = weibull_min.rvs(wb_shape, loc=0, scale=wb_scale)
    u = np.array(u)
    sigma = np.array(sigma)
    return u, sigma


#rv1, rv2 = rvs_j(1e5)
#
# v_ave = 8.5
# r_scale = np.sqrt(2 / np.pi) * v_ave
# x = np.linspace(0, 40, 100)
# pdf_u = []
# for k in range(100):
#     pdf_u.append(rayleigh.pdf(x[k], loc=0, scale=r_scale))
# fig, ax = plt.subplots(1, 1)
# plt.plot(x, pdf_u)
# plt.hist(rv1, 50, density=True)
# plt.show()
#
# x = np.linspace(0, 5, 100)
# pdf_s = []
# for k in range(100):
#     pdf_s.append(pdf_sigma(x[k]))
#
# fig, ax = plt.subplots(1, 1)
# plt.plot(x, pdf_s)
# plt.hist(rv2, 50, density=True)
# plt.show()


#%% Random sample of wind speed u
def rvs_u(n):
    v_ave = 8.5
    r_scale = np.sqrt(2 / np.pi) * v_ave
    v_u = rayleigh.cdf(27, loc=0, scale=r_scale)
    v_l = rayleigh.cdf(3, loc=0, scale=r_scale)
    r = np.random.uniform(low=v_u, high=v_l, size=int(n))
    u = rayleigh.ppf(r, loc=0, scale=r_scale)
    u = np.array(u)
    return u


# #%% pdf versus sample
# rv= rvs_u(1e5)
# #
# v_ave = 8.5
# r_scale = np.sqrt(2 / np.pi) * v_ave
# x = np.linspace(0, 40, 100)
# pdf_u = []
# for k in range(100):
#     pdf_u.append(rayleigh.pdf(x[k], loc=0, scale=r_scale))
# fig, ax = plt.subplots(1, 1)
# plt.plot(x, pdf_u/(1-rayleigh.cdf(3, loc=0, scale=r_scale)))
# # the pdf is truncated pdf, not original Rayleigh
# plt.hist(rv, 50, density=True)
# plt.show()


#%% Random sample of sigma conditional on u
def rvs_su(n, u):
    wb_shape = 0.27 * u + 1.4
    Iref = 0.12
    wb_scale = Iref * (0.75 * u + 3.3)
    sigma = weibull_min.rvs(wb_shape, loc=0, scale=wb_scale, size=int(n))
    sigma_min = np.maximum(0, 0.1 * (u - 20))
    sigma_max = 0.18 * (6.8 + 0.75 * u)
    for k in range(int(n)):
        while (sigma[k] < sigma_min) or (sigma[k] > sigma_max):
            sigma[k] = weibull_min.rvs(wb_shape, loc=0, scale=wb_scale)
    sigma = np.array(sigma)
    return sigma

# #%%
# u = 15
# rv = rvs_su(1e5, u)
# wb_shape = 0.27 * u + 1.4
# Iref = 0.12
# wb_scale = Iref * (0.75 * u + 3.3)
#
# x = np.linspace(0, 5, 100)
# pdf_s = []
# for k in range(100):
#     pdf_s.append(weibull_min.pdf(x[k], wb_shape, loc=0, scale=wb_scale))
#
#
# fig, ax = plt.subplots(1, 1)
# plt.plot(x, pdf_s, 'r-', lw=2, alpha=0.6, label='pdf')
# plt.hist(rv, 50, density=True, label='sample')
# plt.xlabel('standard deviation of wind speed (m/s) ($x$ = 15 m/s)')
# plt.ylabel('conditional probability density')
# plt.legend()
# plt.savefig("Figures/pdf_sigma_u.pdf", format="pdf", bbox_inches="tight")
# plt.show()