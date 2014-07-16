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

def fd2(ndim, disc=None):
  '''
  Generate a square second-order finite-difference matrix.
  The optional disc parameter excludes a discontinuity in the solution at
  position disc.
  '''
  d = np.ones(ndim)
  a = np.diag(d[:-1], k=-1) + np.diag(-2*d) + np.diag(d[:-1], k=1)
  a[0,0] = a[-1,-1] = -1 # For homogeneous Neumann BCs: 1st deriv = 0
  a *= ndim/2

  if disc is None:
    return a

  q1, q2 = np.zeros((ndim,1)), np.zeros((ndim,1))
  q1[0:disc:] = 1.
  q2[disc::] = 1.
  q1 *= 1./np.sqrt(np.dot(q1.T,q1))
  q2 *= 1./np.sqrt(np.dot(q2.T,q2))
  tmp = np.identity(ndim) - np.dot(q1,q1.T) - np.dot(q2,q2.T)
  return np.dot(a, tmp)

# (Log) posterior pdf functions for MCMC samplers
# Designed for linear systems Ax = b where
# A = transfer matrix
# x = vector of parameters to be found
# b = vector of measured data points
# Prior is p(x|prior pars)
# Likelihood is p(b|A, x, prior pars)
# Posterior is \propto prior*likelihood
# -----------------------------------------------

def gaussian_poisson(x, A, b, icov_data, x_prior, icov_prior):
  '''
  Gaussian prior; Poisson likelihood
  '''
  return lngauss(x,x_prior,icov_prior) + lnpoiss(b, np.dot(A,x))

def gamma_poisson(x, A, b, icov_data, x_prior, gamma_a, gamma_b):
  '''
  Gamma prior; Poisson likelihood
  x_prior is a vector of prior means.
  '''
  return lngamma(x, gamma_a, gamma_b) + lnpoiss(b, np.dot(A,x))

def gaussian_gaussian(x, A, b, icov_data, x_prior, icov_prior):
  '''
  Gaussian prior; Gaussian likelihood
  '''
  return lngauss(x,x_prior,icov_prior) + lngauss(b, np.dot(A,x), icov_data)

def l2_poisson(x, A, b, x_prior, alpha, xmin, xmax, L=None):
  '''
  2-norm of (L*) x/x_prior; Poisson likelihood
  Useful options for L:
  1. Unit matrix (standard-form Tikhonov regularization). Default if no L.
  2. A smoothing matrix (like fd2 above) for general-form Tikhonov reg.  
  '''
  if np.any(x < xmin) or np.any(x > xmax):
    return -np.inf

  xr = x/x_prior
  if L is not None:
    xr = np.dot(L,xr)

  xr = xr[1:-1] # Truncate to exclude boundary points
  return -alpha*alpha*np.dot(xr,xr) + lnpoiss(b, np.dot(A,x))

if __name__=='__main__':
  import matplotlib.pyplot as plt
  ndim = 50
  L = fd2(ndim)

  # Draw matrix from fd2() function
  fig, ax = plt.subplots()
  p = ax.pcolormesh(L, cmap='Greys_r')
  fig.colorbar(p)
  fig.savefig('pdfs/fd2.pdf')
  
  # Plot differentiation example showing y'' where y = x^3
  # The result is linear and proportional to y'' = 6*x except at the edges.
  x  = np.linspace(-5,5.,ndim)
  y  = x**3
  ypp = np.dot(L,y)

  dx = x[1] - x[0]
  print 'y[0]=', y[0], 'ypp[0]=', ypp[0], '6x[0]=', 6*x[0], 'dx=', dx
  # ypp[0]  += y[0]/dx/dx
  # ypp[-1] += y[-1]/dx/dx

  fig, ax = plt.subplots()
  ax.plot(x,y, label=r'$y = x^3$')
  ax.plot(x[1:-1],ypp[1:-1], label=r'$Ly \approx d^2y/dx^2$')
  ax.plot(x,6*x, label=r'$6x$')
  ax.legend()
  fig.savefig('pdfs/deriv.pdf')
