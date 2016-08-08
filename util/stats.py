import scipy.stats as stats
import numpy as np

def truncnorm(shape,a,b,mu,sigma):
    T = stats.truncnorm(
        (a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma)
    return T.rvs(shape)

def percentile(a,q):
    return np.percentile(a,q)

a, b = 3.5, 6
mu, sigma = 5, 0.7
a= truncnorm([1024,1024],a,b,mu,sigma)
print(a.shape)
print(np.mean(a))
