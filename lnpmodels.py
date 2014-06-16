import numpy as np
from scipy.special import gammaln

def lngamma(x, a, b):
  '''
  Log likelihood for Gamma(x|a,b) where mean = a/b and var = a/b**2.
  Dimensions of a and b must match x.
  '''
  if np.any(x<=0.) or np.any(a<=0.) or np.any(b<=0.):
    return -np.inf
  return np.sum(a*np.log(b) - gammaln(a) + (a-1)*np.log(x) - b*x)

def lnpoiss(x,mu):
  '''
  Multivariate log likelihood for Poisson(x|mu)
  '''
  if np.any(x<0.) or np.any(mu<=0.):
    return -np.inf
  return np.sum(x*np.log(mu) - gammaln(x+1.0) - mu)

def lngauss(x, mu, prec):
  '''
  Log likelihood for multivariate Gaussian N(x | mu, prec)
  where prec is the inverse covariance matrix.
  '''
  diff = x-mu
  return -np.dot(diff,np.dot(prec,diff))/2.

def lnuniform(x, xref, alpha):
  '''
  Uniform in xref - xref/alpha < x < xref + xref/alpha
  alpha can be a scalar or a vector like x
  '''
  if np.all(alpha > 0.):
    xmin = xref - xref/alpha
    xmax = xref + xref/alpha
    if np.any(x < xmin) or np.any(x > xmax):
      return -np.inf
  return 0.

# (Log) posterior pdf functions for MCMC samplers
# Designed for linear systems Ax = b where
# A = transfer matrix
# x = vector of parameters to be found
# b = vector of measured data points
# Prior is p(x|prior pars)
# Likelihood is p(b|A, x, prior pars)
# Posterior is \propto prior*likelihood
# -----------------------------------------------

# Gaussian prior x Poisson likelihood
def gaussian_poisson(x, A, b, icov_data, x_prior, icov_prior):
  return lngauss(x,x_prior,icov_prior) + lnpoiss(b, np.dot(A,x))

# Gamma prior x Poisson likelihood
def gamma_poisson(x, A, b, icov_data, x_prior, gamma_a, gamma_b):
  '''
  x_prior is a vector of prior means.
  '''
  return lngamma(x, gamma_a, gamma_b) + lnpoiss(b, np.dot(A,x))

# Gaussian prior x Gaussian likelihood
def gaussian_gaussian(x, A, b, icov_data, x_prior, icov_prior):
  return lngauss(x,x_prior,icov_prior) + lngauss(b, np.dot(A,x), icov_data)
