import numpy as np
from scipy.special import gammaln
import unfold_input as ui

# np.set_printoptions(precision=3)
# np.set_printoptions(suppress=True) # suppress exponents on small numbers

def lngamma(x, a, b):
    '''
    Log likelihood for Gamma(x|a,b) where mean = a/b and var = a/b**2.
    Dimensions of a and b must match x.
    '''
    if np.any(x <= 0.) or np.any(a <= 0.) or np.any(b <= 0.):
        return -np.inf
    return np.sum(a * np.log(b) - gammaln(a) + (a - 1) * np.log(x) - b * x)


def lnpoisson(x, mu):
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
    # ln_det_sigma = np.sum(np.log(1. / np.diag(prec)))
    # ln_prefactors = -0.5 * (x.shape[0] * np.log(2 * np.pi) + ln_det_sigma)
    # return ln_prefactors - 0.5 * np.dot(diff, np.dot(prec, diff))
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


def logp_ept_dca(x, matlist, datalist, w, x_prior, alpha, xlim, L):
    '''
    Intended for use with electron pt model (data) as first element of 
    matlist (datalist), and electron DCA model/data as remaining elements.
    '''
    if np.any(x < xlim[:, 0]) or np.any(x > xlim[:, 1]):
        return -np.inf

    lp_ept = w[0] * mvn(x, matlist[0], datalist[0])
    lp_dca = w[1] * dca_shape(x, matlist[1:], datalist[1:])

    return lp_ept + lp_dca + l2reg(x, x_prior, alpha, L)


def l2reg(x, x_prior, alpha, L=None):
    c, b = ui.idx['c'], ui.idx['b']

    rc, rb = x[c] / x_prior[c], x[b] / x_prior[b]

    if L is not None:
        rc, rb = np.dot(L[:, c], rc), np.dot(L[:, b], rb)

    # Truncate to exclude boundary points
    rc, rb = rc[1:-1], rb[1:-1]
    return -alpha * alpha * (np.dot(rc, rc) + np.dot(rb, rb))


def mvn(x, A, data):
    '''
    Multivariate normal (Gaussian) log likelihood. 
    x: trial solution
    A: concatenated A_c, A_b matrices mapping x -> prediction
    data: column 0 contains points, column 1 contains errors. 
    Currently assumes data points are independent (diagonal covariance).
    '''
    c, b = ui.idx['c'], ui.idx['b']
    prediction = np.dot(A[:, c], x[c]) + np.dot(A[:, b], x[b])
    icov_data = np.diag(1. / (data[:, 1] ** 2))
    return lngauss(data[:, 0], prediction, icov_data)


def l2_gaussian(x, A, data, x_prior, alpha, xlim, L=None):
    '''
    L2 regularization; Gaussian likelihood
    x: trial solution
    A: matrix mapping x -> prediction
    data: measurements for prediction to be compared against. 
    '''
    if np.any(x < xlim[:, 0]) or np.any(x > xlim[:, 1]):
        return -np.inf

    return mvn(x, A, data) + l2reg(x, x_prior, alpha, L)


def dca_shape(x, matlist, datalist):
    '''
    Poisson log likelihood. DCA prediction is scaled to integral of data.
    x: trial solution (h_c, h_b invariant yields)
    A: matrix mapping x -> DCA predictions
    data: DCA dists for prediction to be compared against.
        - column 0: inclusive data (signal + background).
        - column 1: background. 
    '''
    c, b = ui.idx['c'], ui.idx['b']
    result = 0.0
    i = 0
    maxnpts = max([d.shape[0] for d in datalist])
    dca_shape.prediction = np.zeros((len(datalist),maxnpts))
    for A, data in zip(matlist, datalist):

        # Calculate prediction from this sample vector x
        pred = np.dot(A[:, c], x[c]) + np.dot(A[:, b], x[b])

        # Scale predicted yield to match data signal, then add background
        scf = (np.sum(data[:, 0]) - np.sum(data[:,1])) / np.sum(pred)
        pred = pred*scf + data[:, 1]

        # Store prediction for monitoring purposes
        dca_shape.prediction[i,:pred.shape[0]] = pred

        result += lnpoisson(data[:, 0], pred)
        i += 1
    return result
dca_shape.prediction = None


def l2_poisson_shape(x, matlist, datalist, x_prior, alpha, xlim, L=None):
    '''
    L2 regularization; Poisson likelihood
    x: trial solution
    A: matrix mapping x -> prediction
    data: measurements for prediction to be compared against.
        - column 0: inclusive data (signal + background).
        - column 1: background. 
    '''
    if np.any(x < xlim[:, 0]) or np.any(x > xlim[:, 1]):
        return -np.inf
    return dca_shape(x, matlist, datalist) + l2reg(x, x_prior, alpha, L)


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
