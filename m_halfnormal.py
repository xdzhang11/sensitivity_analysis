# %% Half Normal distribution
from scipy.stats import halfnorm
from cmath import pi
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(1, 1)

mu_hn = 1
scale_hn = 0.05/np.sqrt(1-2/pi)

mean, var, skew, kurt = halfnorm.stats(moments='mvsk',loc=mu_hn, scale=scale_hn)

x = np.linspace(halfnorm.ppf(0.001,loc=mu_hn, scale=scale_hn),
                halfnorm.ppf(0.999,loc=mu_hn, scale=scale_hn), 100)
ax.plot(2-x, halfnorm.pdf(x,loc=mu_hn, scale=scale_hn),
       'r-', lw=5, alpha=0.6, label='halfnorm pdf')

ax.set_xlabel('x')
ax.set_ylabel('pdf')

r = 2-halfnorm.rvs(size = 100000,loc=mu_hn, scale=scale_hn)
plt.show()