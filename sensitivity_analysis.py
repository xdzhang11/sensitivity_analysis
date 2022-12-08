#%% ExactPerm
import numpy as np
import matplotlib.pyplot as plt; 
from numpy.random import multivariate_normal
from itertools import permutations
from numpy.linalg import multi_dot,inv
import time

#%% Sobol indeces with 
t = time.time()

#%% Shapley effects
d = 3       # dimension of inputs
Nv = 10000   # MC sample size to estimate var(Y)
Ni = 50      # sample size for inner loop
No = 2000      # sample size for outer loop

Sh = np.zeros(d)
Sh2 = np.zeros(d)

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html

beta = np.array([1,1,1]).transpose()
sigma_i = [1, 1, 2]
rho23_all =  np.linspace(-0.9,.9,20)
s1 = []
s2 = []
s3 = []

for rho23 in rho23_all:

    rho = [[1,    0,    0], 
        [0,    1,  rho23],
        [0,  rho23,    1]]

    mu = np.array([0,0,0])

    sigma = np.zeros((d,d))
    for k in range(d):
        for j in range(d):
            sigma[k][j] = sigma_i[k]*sigma_i[j]*rho[k][j]

    X = multivariate_normal(mu, sigma, Nv)
   #X = rv.rvs(size=Nv)
    y = np.dot(X,beta)

    EY = np.mean(y)
    VarY = np.var(y)

    # np.random.permutation(10)
    perms = np.array(list(permutations(range(d))))
    m = len(perms) # number of permutations

    for p in range(m):
        pi = perms[p,:]
        mu_p = mu[pi]
        sigma_p = np.zeros((d,d))
        for k in range(d):
            for j in range(d):
                sigma_p[k][j] = sigma_i[pi[k]]*sigma_i[pi[j]]*rho[pi[k]][pi[j]]
                
        prevC = 0
        for j in range(d): # loop through dimension
            if j == d-1:
                c_hat = VarY
                delta = c_hat-prevC
            else:
                cVar = []           # conditional variance    
                Sj = pi[:j+1]         # set of the 1st-jth elements in pi 
                Sjc = pi[j+1:]        # set of the (j+1)th-kth elements in pi
                # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
                # https://stats.stackexchange.com/questions/348941/general-conditional-distributions-for-multivariate-gaussian-mixtures
                mu_1 = mu_p[:j+1]
                mu_2 = mu_p[j+1:]
                sigma_11 = sigma_p[:j+1,:j+1]
                sigma_12 = sigma_p[:j+1,j+1:]
                sigma_21 = sigma_p[j+1:,:j+1]
                sigma_22 = sigma_p[j+1:,j+1:]

                sigma_hat = sigma_11-multi_dot([sigma_12, inv(sigma_22), sigma_21])
                
                xjc = multivariate_normal(mu_2, sigma_22, No)

                # rv_jc = multivariate_normal(mu_2, sigma_22)
                # for l in range(No): # loop through outer loop
                #     c_Y = []            # conditional Y for calcualting cVar
                #     xjc = rv_jc.rvs()         # sample from X_{-J}
                #     for h in range(Ni):     # loop through inner loop

                #         mu_hat = mu_1.transpose()+multi_dot([sigma_12, inv(sigma_22), xjc.transpose()-mu_2.transpose()])
                #         mu_hat = mu_hat.transpose()

                #         rv_j = multivariate_normal(mu_hat, sigma_hat)

                #         xj =  rv_j.rvs()              # sample xj conditional on xjc

                #         x = np.hstack([xj, xjc])
                #         s_index = np.argsort(pi)       # sorted indices
                #         y_t = np.dot(x[s_index],beta)
                #         c_Y.append(y_t)
                #     cVar.append(np.var(c_Y))
                
                # xjc = rv_jc.rvs(No)         # sample from X_{-J}
                for l in range(No): # loop through outer loop
                    c_Y = []            # conditional Y for calcualting cVar
                    mu_hat = mu_1.reshape((j+1,1))+multi_dot([sigma_12, inv(sigma_22), xjc[l].reshape((d-j-1,1))-mu_2.reshape((d-j-1,1))])
                    mu_hat = mu_hat.flatten()
                    xj = multivariate_normal(mu_hat, sigma_hat, Ni)
                    #xj =  rv_j.rvs(Ni)              # sample xj conditional on xjc
                  
                    s_index = np.argsort(pi)       # sorted indices
                    for h in range(Ni):     # loop through inner loop
                        x = np.hstack([xj,np.tile(xjc[l],(Ni,1))])
                        y_t = np.dot(x[:,s_index],beta)
                        c_Y.append(y_t)
                    cVar.append(np.var(c_Y))

                c_hat = np.mean(cVar)
                delta = c_hat-prevC
            
            Sh[pi[j]] = Sh[pi[j]] + delta
            Sh2[pi[j]] = Sh2[pi[j]] + delta**2
            prevC = c_hat

    Sh = Sh/m/VarY
    Sh2 = Sh2/m
    Shapley = Sh
    SEShapley = np.sqrt((Sh2 - Sh**2)/m)
    VarY = VarY
    EY = EY

    s1.append(Sh[0])
    s2.append(Sh[1])
    s3.append(Sh[2])



# %%
plt.plot(rho23_all,s1,label="x1")
plt.plot(rho23_all,s2,label="x2")
plt.plot(rho23_all,s3,label="x3")
plt.legend()
plt.show()
# %%
elapsed = time.time() - t
# %%
