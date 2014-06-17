'''
samplehpt.py
Sample from joint probability to find heavy-flavor hadron yields vs pt
'''

from ROOT import TFile
from h2np import h2a, binedges, binctrs
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.colors import LogNorm
import emcee
import triangle
import sys
import lnpmodels

alpha = 10
nwalkers = 500
nburnin = 500
nsteps = 1000

def draa(x):
  return 1.3*np.sqrt(2*np.pi*1.1*1.1)*norm.pdf(x, loc=1.5, scale=1.1) + \
  0.2/(1 + np.exp(-x+3))

def braa(x):
  return 0.6*np.exp(-x/3) + \
  1.1*np.sqrt(2*np.pi*1.5*1.5)*norm.pdf(x, loc=3.4, scale=1.5) + \
  0.3/(1 + np.exp(-x+7))

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
hptx    = binctrs(hptGen,'x')
eptx    = binctrs(hEpt,'x')
ndim    = aept.shape[1]
raa     = np.concatenate((draa(hptx[:10]), braa(hptx[:10])))
hptmod  = raa*hpt
hptbins = binedges(hptGen,'x')

# psigma  = np.zeros(hpt.size)
# if alpha > 0: psigma = hpt/alpha

aept /= aept.sum(axis=0)
eptmod  = np.dot(aept,hptmod)
ept_err = np.sqrt(eptmod)     #h2a(hEpt, 'e')

# Ensemble of starting points for the walkers. 
print("ndim: {}, nwalkers: {}".format(ndim, nwalkers))
p0 = hpt*(1. + 0.1*np.random.randn(nwalkers, ndim))

# Construct an ensemble sampler object from emcee

# Gaussian prior, Poisson likelihood
icov_data  = np.diagflat(1./ept_err**2)
icov_prior = np.diagflat((alpha/hpt)**2)
# sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpmodels.gaussian_poisson, 
#                                 args=[aept, eptmod, icov_data, hpt, icov_prior])

# # Gamma prior, Poisson likelihood
# sampler = emcee.EnsembleSampler(nwalkers, ndim, gamma_poisson, 
#                                 args=[aept, ept, icov_data, 
#                                 hpt, hpt/alpha, np.ones_like(hpt)/alpha])

L = lnpmodels.fd2(ndim)
ymin = 0.01*hpt
ymax = 2*hpt
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpmodels.l2_poisson, 
                                args=[aept, eptmod, icov_data, 
                                hpt, alpha, ymin, ymax, L])

print("Burning in for {} steps...".format(nburnin))
pos, prob, state = sampler.run_mcmc(p0, nburnin)
sampler.reset()
print("prob: len({})".format(len(prob)))
print("state: {} len({}) {} {} {}".format(state[0], 
      len(state[1]), state[2], state[3], state[4]))

print("Running sampler for {} steps...".format(nsteps))
sampler.run_mcmc(pos, nsteps)
print("Finished sampling.")

acc_frac = np.mean(sampler.acceptance_fraction)
print("Mean acceptance fraction: {0:.3f}".format(acc_frac))

# Flatten down the output
samples = sampler.chain.reshape((-1, ndim))
print(samples.shape)

# Posterior quantiles: list of ndim (16,50,84) percentile tuples
pq = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
         zip(*np.percentile(samples, [16, 50, 84], axis=0)))
pq = np.array(pq)




print("Drawing results...")
# -----------------------------------------------------------------------------
# Plotting only after this point
# -----------------------------------------------------------------------------

# Draw prior, walkers, initial guess, and result.
fig, ax = plt.subplots()
ax.set_yscale('log')
ax.set_xlabel(r'bin index')
# ymin and ymax (part of prior)
ax.fill_between(hptx, ymin, ymax, color='slategray', alpha=0.1)
# walkers
for i in range(nwalkers): 
  ax.plot(hptx, p0[i,:], ls='*', marker='s', ms=14, color='deepskyblue', alpha=0.01)
# gen
ax.plot(hptx, hpt, lw=2, ls='*', marker='o', color='white')
# mod
ax.plot(hptx, hptmod, lw=2, ls='*', marker='s', color='black')
# result
ax.errorbar(hptx, pq[:,0], yerr=[pq[:,2], pq[:,1]],
            ls='*', fmt='o', color='crimson', ecolor='crimson', capthick=2)
fig.savefig('pdfs/hpt.pdf')

# Draw matrix
fig, ax = plt.subplots()
p = ax.pcolormesh(aept, norm=LogNorm(vmin=aept.min()+1e-8, vmax=aept.max()), 
                  cmap='Spectral_r') #cmap='RdYlBu_r') #cmap='Blues')
ax.set_xlabel(r'h $p_T$ bin index')
ax.set_ylabel(r'$e^{\pm}$ $p_T$ bin index')
ax.set_ylim([0, aept.shape[1]+1])
fig.colorbar(p)
fig.savefig('pdfs/aept.pdf')

# Draw ept
eptrefold = np.dot(aept,pq[:,0])
eptref_hi = np.dot(aept,pq[:,1])
eptref_lo = np.dot(aept,pq[:,2])

fig, ax = plt.subplots()
ax.set_yscale('log')
ax.set_xlabel(r'$e^{\pm}$ $p_T$ [GeV/c]')
ax.errorbar(eptx, ept, yerr=ept_err, lw=2, ls='*', marker='o', color='white')
ax.errorbar(eptx, eptrefold, yerr= [eptref_lo, eptref_hi],
            lw=2, ls='*', marker='s', ms=10, color='r')
ax.errorbar(eptx, eptmod, yerr=ept_err, lw=2, ls='*', marker='o', color='k')
fig.savefig('pdfs/ept.pdf')

# Draw posterior marginal distributions
# sample.chain has shape (nwalkers, nsteps, ndim)
print("Histogramming results for plotting...")
nr,nc = 4,5
fig, axes = plt.subplots(nr,nc)
for row in range(nr):
  for col in range(nc):
    i = nc*row + col
    a = axes[row,col]
    fc = 'yellow' if i < ndim/2 else 'lime'
    a.hist(sampler.flatchain[:,i], 100, 
           color='k',facecolor=fc, histtype='stepfilled')
    a.tick_params(axis='x', top='off', labelsize=4)
    a.tick_params(axis='y', left='off', right='off', labelsize=0)
    a.xaxis.get_offset_text().set_size(4)
    a.xaxis.get_major_formatter().set_powerlimits((0, 1))
    s = r'{0:.0f}-{1:.0f} GeV/c'.format(hptbins[i%10], hptbins[i%10+1])
    a.text(0.6, 0.9, s, fontsize=5, transform=a.transAxes)
    if i < ndim/2:
      a.text(0.05, 0.9, r'$h_{charm}$', fontsize=5, transform=a.transAxes)
    else:
      a.text(0.05, 0.9, r'$h_{beauty}$', fontsize=5, transform=a.transAxes)
fig.savefig('pdfs/posterior.pdf', bbox_inches='tight')

# Make a triangle plot.
# figc = triangle.corner(samples[:, 0:ndim/2])
# figc.savefig("ctri.png")
# figb = triangle.corner(samples[:, ndim/2:ndim])
# figb.savefig("btri.png")
