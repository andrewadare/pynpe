import numpy as np
from scipy.special import gammaln
import unfold_input as ui


def lngamma(x, a, b):
    '''
    Log likelihood for Gamma(x|a,b) where mean = a/b and var = a/b**2.
    Dimensions of a and b must match x.
    '''
    if np.any(x <= 0.) or np.any(a <= 0.) or np.any(b <= 0.):
        return -np.inf
    return np.sum(a * np.log(b) - gammaln(a) + (a - 1) * np.log(x) - b * x)


def lnpoiss(x, mu):
    '''
    Multivariate log likelihood for Poisson(x|mu)
    '''
    if np.any(x < 0.) or np.any(mu <= 0.):
        return -np.inf
    return np.sum(x * np.log(mu) - gammaln(x + 1.0) - mu)


def lngauss(x, mu, prec):
    '''
    Log likelihood for multivariate Gaussian N(x | mu, prec)
    where prec is the inverse covariance matrix.
    Neglecting terms that don't depend on x.
    '''
    diff = x - mu
    return -0.5 * np.dot(diff, np.dot(prec, diff))


def lnuniform(x, xref, alpha):
    '''
    Uniform in xref - xref/alpha < x < xref + xref/alpha
    alpha can be a scalar or a vector like x
    '''
    if np.all(alpha > 0.):
        xmin = xref - xref / alpha
        xmax = xref + xref / alpha
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
    a = np.diag(d[:-1], k=-1) + np.diag(-2 * d) + np.diag(d[:-1], k=1)
    a[0, 0] = a[-1, -1] = -1  # For homogeneous Neumann BCs: 1st deriv = 0
    a *= ndim / 2

    if disc is None:
        return a

    q1, q2 = np.zeros((ndim, 1)), np.zeros((ndim, 1))
    q1[0:disc:] = 1.
    q2[disc::] = 1.
    q1 *= 1. / np.sqrt(np.dot(q1.T, q1))
    q2 *= 1. / np.sqrt(np.dot(q2.T, q2))
    tmp = np.identity(ndim) - np.dot(q1, q1.T) - np.dot(q2, q2.T)
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
    return lngauss(x, x_prior, icov_prior) + lnpoiss(b, np.dot(A, x))


def gamma_poisson(x, A, b, icov_data, x_prior, gamma_a, gamma_b):
    '''
    Gamma prior; Poisson likelihood
    x_prior is a vector of prior means.
    '''
    return lngamma(x, gamma_a, gamma_b) + lnpoiss(b, np.dot(A, x))


def gaussian_gaussian(x, A, b, icov_data, x_prior, icov_prior, alpha, 
                      xmin, xmax, L=None):
    '''
    Gaussian prior; Gaussian likelihood
    '''
    if np.any(x < xmin) or np.any(x > xmax):
        return -np.inf

    return lngauss(x, x_prior, icov_prior) + lngauss(b, np.dot(A, x), icov_data)


def l2_gaussian(x, A, data, icov_data, x_prior, alpha, xlim, L=None):
    '''
    L2 regularization; Gaussian likelihood
    x: trial solution
    A: matrix mapping x -> prediction
    data: measurements for prediction to be compared against. 
    '''
    if np.any(x < xlim[:, 0]) or np.any(x > xlim[:, 1]):
        return -np.inf

    f = x[-1]
    c, b = ui.idx['c'], ui.idx['b']

    # Log likelihood
    prediction = (1 - f) * np.dot(A[:,c], x[c]) + f * np.dot(A[:,b], x[b])
    ll = lngauss(data, prediction, icov_data)

    # Regularization
    rc, rb = x[c]/x_prior[c], x[b]/x_prior[b]
    if L is not None:
        rc, rb = np.dot(L[:,c], rc), np.dot(L[:,b], rb)
    reg = -alpha * alpha * (np.dot(rc, rc) + np.dot(rb, rb))

    return ll + reg


def l2_poisson(x, A, b, x_prior, alpha, xmin, xmax, L=None):
    '''
    2-norm of (L*) x/x_prior; Poisson likelihood
    Useful options for L:
    1. Unit matrix (standard-form Tikhonov regularization). Default if no L.
    2. A smoothing matrix (like fd2 above) for general-form Tikhonov reg.  
    '''
    if np.any(x < xmin) or np.any(x > xmax):
        return -np.inf

    xr = x / x_prior
    if L is not None:
        xr = np.dot(L, xr)

    xr = xr[1:-1]  # Truncate to exclude boundary points

    return -alpha * alpha * np.dot(xr, xr) + lnpoiss(b, np.dot(A, x))


def l2_poisson_shape(x, Alist, blist, x_prior, alpha, xmin, xmax, L=None):
    '''
    Compute likelihood based only on shape comparison between Ax and b.
    Otherwise similar to l2_poisson.
    '''

    # Make a 2-vector with c-hadron and b-hadron integrated yields from x.
    # ndim = x.shape[0]
    # xint = np.array([np.sum(x[:ndim/2]), np.sum(x[ndim/2:ndim])])

    result = 0.0
    for A, b in zip(Alist, blist):
        if np.any(x < xmin) or np.any(x > xmax):
            return -np.inf

        xr = x / x_prior
        if L is not None:
            xr = np.dot(L, xr)

        xr = xr[1:-1]  # Truncate to exclude boundary points

        reg = -alpha * alpha * np.dot(xr, xr)
        Ax = np.dot(A, x)
        Ax = np.sum(b) / np.sum(Ax) * Ax
        result += reg + lnpoiss(b, Ax)
        # result += lnpoiss(b, Ax)

    return result


def l2_poisson_combined(x, Alist, blist, x_prior, alpha, xmin, xmax, L=None):
    '''
    Intended for use with electron pt model (data) as first element of 
    Alist (blist), and electron DCA model/data as remaining elements.
    '''
    ll_ept = l2_poisson(x, Alist[0], blist[0],
                        x_prior, alpha, xmin, xmax, L)
    ll_dca = 0.
    if (ll_ept > -np.inf):
        ll_dca = l2_poisson_shape(x, Alist[1:], blist[1:],
                                  x_prior, alpha, xmin, xmax, L)
    return ll_ept + 1e-3 * ll_dca


if __name__ == '__main__':
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
    x = np.linspace(-5, 5., ndim)
    y = x ** 3
    ypp = np.dot(L, y)

    dx = x[1] - x[0]
    print 'y[0]=', y[0], 'ypp[0]=', ypp[0], '6x[0]=', 6 * x[0], 'dx=', dx

    fig, ax = plt.subplots(figsize=(4, 5))
    ax.plot(x, y, label=r'$y = x^3$')
    ax.plot(x, 6 * x, label=r'$6x$')
    ax.plot(x[1:-1], ypp[1:-1], label=r'$Ly$')
    ax.set_title(r'$Ly$ vs. exact $d^2y/dx^2$')
    ax.set_xlabel('x')
    ax.legend(loc=2)
    fig.savefig('pdfs/deriv.pdf')
