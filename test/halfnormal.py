# %% Half Normal distribution
from scipy.stats import halfnorm
from cmath import pi
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(1, 1)

plt.rcParams["figure.figsize"] = (8, 6)

mu_hn = 1
scale_hn = 0.05/np.sqrt(1-2/pi)

mean, var, skew, kurt = halfnorm.stats(moments='mvsk',loc=mu_hn, scale=scale_hn)

x = np.linspace(halfnorm.ppf(0.001,loc=mu_hn, scale=scale_hn),
                halfnorm.ppf(0.999,loc=mu_hn, scale=scale_hn), 100)
plt.plot(2-x, halfnorm.pdf(x,loc=mu_hn, scale=scale_hn),
       'b-', lw=3, alpha=0.6, label='halfnorm pdf')

plt.xlabel('cl multiplier', fontsize=12)
plt.ylabel('probability density', fontsize=12)

# r = 2-halfnorm.rvs(size = 100000,loc=mu_hn, scale=scale_hn)

plt.savefig("Figures/pdf_half_normal.pdf", format="pdf", bbox_inches="tight")
plt.show()