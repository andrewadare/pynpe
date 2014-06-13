
from ROOT import TFile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LogNorm
from h2np import h2a, binedges, binctrs
import emcee
import triangle
import sys

alpha = 1.001
nwalkers = 200
nburnin = 250 #500
nsteps = 750 #1000

rcParams['font.size'] = 14
rcParams['font.family'] = ['sans-serif']
rcParams['font.sans-serif'] = ['Arial']
rcParams['mathtext.default'] = 'sf'
rcParams['figure.max_open_warning'] = 50


# # This works...too well.
# def lnprior(x, xref, alpha):
#   if x.any < 0.:
#     return -np.inf
#   if alpha == 0.:
#     return 0.
#   diff = x - xref
#   return -np.dot(diff,diff)/alpha/alpha

# # Very low mean acc. fraction during burn-in. runtime warning:
# #  invalid value encountered in subtract
# #  invalid value encountered in greater
# def lnprior(x, xref, alpha):
#   if alpha == 0.:
#     return 0.
#   diff = np.abs(x - xref)/xref
#   if diff.any > alpha: 
#     return -np.inf
#   return 0.0


# def lnpoiss(n,mu):
#   '''
#   Multivariate log likelihood for Poisson(n | mu)
#   '''
#   return 
  
# # An uninformative prior.
# def lnprior(x, xref, alpha):
#   if x.any < 0.:
#     return -np.inf
#   return 0.

def lngauss(x, mu, prec):
  '''
  Log likelihood for multivariate Gaussian N(x | mu, prec)
  where prec is the inverse covariance matrix.
  '''
  diff = x-mu
  return -np.dot(diff,np.dot(prec,diff))/2.

# def lnlike(x, A, b, icov):
#   diff = np.dot(A,x) - b
#   return -np.dot(diff,np.dot(icov,diff))/2.

# # Original
# def lnprob(x, A, b, icov, xini, alpha):
#   lp = lnprior(x, xini, alpha)
#   if not np.isfinite(lp):
#     return -np.inf
#   return lp + lnlike(x, A, b, icov)

# x = parameter vector like hpt
# mu_prior = hpt
# mu_data = A*x
# icov_prior = np.diagflat((alpha/mu)**2)
# icov_data  = np.diagflat(1./ept_err**2)
# def logmvnprob(x, data, icov_data, mu_prior, icov_prior):
#   return lngauss(x, mu_prior, icov_prior) + lngauss(x, mu_data, icov_data)

dtype = 'MC'
modelFile = TFile('rootfiles/bfrac0.030-dcares0um.root')
eptFile   = TFile('rootfiles/simspectra.root')
dcaFile   = TFile('rootfiles/simdca.root')

# Pythia truth (for initializing walkers)
hptGen = eptFile.Get('hptGen')

# Transfer matrices from PYTHIA / DcaGen
hAept = modelFile.Get('hAept')
dcaMatrix = [] # TH2D

# Electron pt spectra
hEpt = eptFile.Get('hEptMC')
hEpt.SetName('hEpt')
eptIntegral = hEpt.Integral()

# Measured (or maybe simulated) DCA and combined background
measdca = [None]*6 # TH1D
measbkg = [None]*6 # TH1D
for i in range(6):
  prefix = '' if dtype=='MC' else 'qm12'
  measdca[i] = dcaFile.Get('{}{}dca{}'.format(prefix, dtype, i))
  # measbkg[i] = bkgFile.Get('{}{}bkg{}'.format(prefix, dtype, i))
  # print measbkg[i].Integral()

# Convert histograms to numpy arrays
aept    = h2a(hAept)
hpt     = aept.sum(axis=0) #h2a(hptGen)
ept     = aept.sum(axis=1) #h2a(hEpt)
ept_err = np.sqrt(ept)     #h2a(hEpt, 'e')
hptx    = binctrs(hptGen,'x')
eptx    = binctrs(hEpt,'x')
ndim    = aept.shape[1]

psigma  = np.zeros(hpt.size)
if alpha > 0: psigma = hpt/alpha

aept /= aept.sum(axis=0)

print("ndim: {}, nwalkers: {}".format(ndim, nwalkers))

# Ensemble of starting points for the walkers. 
# p0 is an array (length nwalkers) of vectors (each length ndim).
# p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
# p0 = np.random.randn(nwalkers, ndim) + hpt
p0 = hpt*(1. + 0.5*np.random.randn(nwalkers, ndim))
# print p0.shape # (250, 20)

# Draw matrix
fig, ax = plt.subplots()
p = ax.pcolormesh(aept, norm=LogNorm(vmin=aept.min()+1e-8, vmax=aept.max()), 
                  cmap='Spectral_r') #cmap='RdYlBu_r') #cmap='Blues')
ax.set_xlabel(r'h $p_T$ bin index')
ax.set_ylabel(r'$e^{\pm}$ $p_T$ bin index')
ax.set_ylim([0, aept.shape[1]+1])
fig.colorbar(p)
fig.savefig('aept.pdf')

# Draw ept
fig, ax = plt.subplots()
ax.set_yscale('log')
ax.set_xlabel(r'$e^{\pm}$ $p_T$ [GeV/c]')
ax.errorbar(eptx, ept, yerr=ept_err, lw=2, ls='*', marker='o', color='k')
ax.plot(eptx, np.dot(aept,hpt), lw=2, ls='*', marker='o', color='r')

fig.savefig('ept.pdf')
# sys.exit()

# Define the distribution to sample from.
# Gaussian posterior:
def lnprob(x, A, b, icov_data, x_prior, icov_prior):
  return lngauss(x,x_prior,icov_prior) + lngauss(b, np.dot(A,x), icov_data)
icov_data  = np.diagflat(1./ept_err**2)
icov_prior = np.diagflat((alpha/hpt)**2)

# Construct an ensemble sampler object from emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, 
                                args=[aept, ept, icov_data, hpt, icov_prior])

# Burn-in phase of 100 steps (100 chosen very arbitrarily).
# The position of the walker is saved to pos.
# reset() clears the bookkeeping parameters to prepare for a fresh start.
print("Burning in for {} steps...".format(nburnin))
pos, prob, state = sampler.run_mcmc(p0, nburnin)
sampler.reset()
print("prob: len({})".format(len(prob)))
print("state: {} len({}) {} {} {}".format(state[0], 
      len(state[1]), state[2], state[3], state[4]))

# Full run with 1000 steps. 
# Output object is sample.chain array with shape (nwalkers, nsteps, ndim)
# or, better, sample.flatchain with shape (nwalkers x nsteps, ndim)
print("Running sampler for {} steps...".format(nsteps))
sampler.run_mcmc(pos, nsteps)
print("Finished sampling.")

# Draw the results
# sample.chain has shape (nwalkers, nsteps, ndim)
print("Histogramming results for plotting...")
for i in range(ndim): # there are ndim plots, but only show first three.
  plt.figure()
  plt.hist(sampler.flatchain[:,i], 100, color="k", histtype="step")
  plt.title("Dimension {0:d}".format(i))
  plt.savefig('bin{0:02d}.pdf'.format(i))

acc_frac = np.mean(sampler.acceptance_fraction)
print("Mean acceptance fraction: {0:.3f}".format(acc_frac))

# Flatten down the output
samples = sampler.chain.reshape((-1, ndim))
print(samples.shape)

# Posterior quantiles: list of ndim (16,50,84) percentile tuples
pq = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
         zip(*np.percentile(samples, [16, 50, 84], axis=0)))
pq = np.array(pq)

# Draw prior, walkers, initial guess, and result.
fig, ax = plt.subplots()
ax.set_yscale('log')
ax.set_xlabel(r'bin index')
# prior
ax.fill_between(hptx, np.maximum(hpt-psigma,1.0*np.ones_like(hpt)), hpt+psigma, 
                color='slategray', alpha=0.1)
# walkers
for i in range(nwalkers): 
  ax.plot(hptx, p0[i,:], ls='*', marker='s', ms=14, color='deepskyblue', alpha=0.01)
# truth
ax.plot(hptx, hpt, lw=2, ls='*', marker='o', color='black')
# result
# ax.bar(hptx, pq[:,0], yerr=[pq[:,2], pq[:,1]],
#             align='center',color='red', edgecolor='orange', linewidth=2,
#             error_kw=dict(ecolor='orange', lw=2, capsize=5, capthick=2))
ax.errorbar(hptx, pq[:,0], yerr=[pq[:,2], pq[:,1]],
            ls='*', fmt='o', color='crimson', ecolor='crimson', capthick=2)
fig.savefig('hpt.pdf')

# Make a triangle plot.
# figc = triangle.corner(samples[:, 0:ndim/2])
# figc.savefig("ctri.png")
# figb = triangle.corner(samples[:, ndim/2:ndim])
# figb.savefig("btri.png")


# Keep for later reference
# fig = triangle.corner(samples, labels=["$m$", "$b$", "$\ln\,f$"],
#                       truths=[m_true, b_true, np.log(f_true)])


