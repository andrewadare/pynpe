'''
samplehpt.py
Sample from joint probability to find heavy-flavor hadron yields vs pt
'''
from __future__ import print_function
from ROOT import TFile
from h2np import h2a, binedges, binctrs
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.colors import LogNorm
from time import clock
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

dtype     = 'MC'
dcabins   = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
modelFile = TFile('rootfiles/bfrac0.030-dcares0um.root')
eptFile   = TFile('rootfiles/simspectra.root')
dcaFile   = TFile('rootfiles/simdca.root')
hptGen    = eptFile.Get('hptGen')
hAept     = modelFile.Get('hAept') # Transfer matrices from PYTHIA / DcaGen
hEpt      = eptFile.Get('hEptMC') # Electron pt spectra
dcaMat    = [None]*6
measdca   = [None]*6
measbkg   = [None]*6

for i in range(6):
  dcaMat[i] = h2a(modelFile.Get('hdca{}'.format(i)))
  prefix = '' if dtype=='MC' else 'qm12'
  measdca[i] = h2a(dcaFile.Get('{}{}dca{}'.format(prefix, dtype, i)))
  # measbkg[i] = bkgFile.Get('{}{}bkg{}'.format(prefix, dtype, i))
  # print measbkg[i].Integral()

if False:
  nr,nc = 2,3
  fig, axes = plt.subplots(nr,nc)
  for row in range(nr):
    for col in range(nc):
      i = nc*row + col
      a = axes[row,col]
      a.tick_params(axis='x', top='off', labelsize=4)
      a.tick_params(axis='y', left='off', right='off', labelsize=4)
      s = r'{0:.1f}-{1:.1f} GeV/c'.format(dcabins[i], dcabins[i+1])
      a.text(0.6, 0.9, s, fontsize=5, transform=a.transAxes)
      m = dcaMat[i]
      p = a.pcolormesh(m, norm=LogNorm(vmin=m.min()+1e-8, vmax=m.max()),
                       cmap='Spectral_r')
      # fig.colorbar(p)
  fig.savefig('pdfs/dca_matrices.pdf', bbox_inches='tight')

# Convert histograms to numpy arrays
aept    = h2a(hAept)
hpt     = aept.sum(axis=0) #h2a(hptGen)
ept     = aept.sum(axis=1) #h2a(hEpt)
dca     = [m.sum(axis=1) for m in dcaMat]
bkg     = [np.mean(d[0:10])*np.ones_like(d) for d in dca]
hptx    = binctrs(hptGen,'x')
eptx    = binctrs(hEpt,'x')
dcax    = binctrs(modelFile.Get('hdca0'),'x')
ndim    = aept.shape[1]
raa     = np.concatenate((draa(hptx[:10]), braa(hptx[:10])))
hptmod  = raa*hpt
hptbins = binedges(hptGen,'x')

aept /= aept.sum(axis=0)
dcaMat = [m for m in dcaMat] # [m/m.sum(axis=0) for m in dcaMat]
eptmod  = np.dot(aept,hptmod)
dcamod  = [np.dot(m,hptmod) for m in dcaMat]
ept_err = np.sqrt(eptmod)     #h2a(hEpt, 'e')

if False:
  nr,nc = 2,3
  fig, axes = plt.subplots(nr,nc)
  for row in range(nr):
    for col in range(nc):
      i = nc*row + col
      a = axes[row,col]
      a.set_yscale('log')
      a.tick_params(axis='x', top='off', labelsize=6)
      a.tick_params(axis='y', labelsize=6)
      s = r'{0:.1f}-{1:.1f} GeV/c'.format(dcabins[i], dcabins[i+1])
      a.text(0.55, 0.9, s, fontsize=8, transform=a.transAxes)
      a.step(dcax, bkg[i], color='brown')
      a.step(dcax, dca[i], color='black', alpha = 0.6)
      a.step(dcax, dcamod[i], color='red', alpha = 0.6)
      # a.errorbar(eptx, eptrefold, yerr= [eptref_lo, eptref_hi],
      #         lw=2, ls='*', marker='s', ms=10, color='r')
      # a.errorbar(eptx, eptmod, yerr=ept_err, lw=2, ls='*', marker='o', color='k')
  
      # fig.colorbar(p)
  fig.savefig('pdfs/dca_dists.pdf', bbox_inches='tight')

# sys.exit()

# Ensemble of starting points for the walkers. 
print("Initializing {} {}-dim walkers...".format(nwalkers, ndim))
p0 = hpt*(1. + 0.1*np.random.randn(nwalkers, ndim))

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
                                args=[aept, eptmod, hpt, alpha, ymin, ymax, L])

print("Burning in for {} steps...".format(nburnin), end=' ')
sys.stdout.flush()
start = clock()
pos, prob, state = sampler.run_mcmc(p0, nburnin)
sampler.reset()
end = clock()
print(str(end-start) + ' s')
print("state: {} len({}) {} {} {}".format(state[0], 
      len(state[1]), state[2], state[3], state[4]))

print("Running sampler for {} steps...".format(nsteps), end=' ')
sys.stdout.flush()
start = clock()
sampler.run_mcmc(pos, nsteps)
end = clock()
print(str(end-start) + ' s')

acc_frac = np.mean(sampler.acceptance_fraction)
print("Mean acceptance fraction: {0:.3f}".format(acc_frac))

# Flatten down the output
samples = sampler.chain.reshape((-1, ndim))
print(samples.shape)

# Posterior quantiles: list of ndim (16,50,84) percentile tuples
pq = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
         zip(*np.percentile(samples, [16, 50, 84], axis=0)))
pq = np.array(pq)

# -----------------------------------------------------------------------------
#                            Plot results
# -----------------------------------------------------------------------------

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

# Draw prior, walkers, initial guess, and result.
print("Drawing results...")
fig, axes = plt.subplots(1,2)
ptx = hptx[:ndim/2]
for ax in axes:
  cb = 'charm' if ax == axes[0] else 'beauty'
  r  = range(ndim/2) if ax == axes[0] else range(ndim/2,ndim)
  ax.set_yscale('log')
  ax.set_xlabel(r'{} hadron $p_T$ [GeV/c]'.format(cb))
  # ymin and ymax (part of prior)
  ax.fill_between(ptx, ymin[r], ymax[r], color='slategray', alpha=0.1)
  # walkers
  for i in range(nwalkers): 
    ax.plot(ptx, p0[i,r], ls='*', marker='s', ms=14, color='deepskyblue', alpha=0.01)
  # gen
  ax.plot(ptx, hpt[r], lw=2, ls='*', marker='o', color='white')
  # mod
  ax.plot(ptx, hptmod[r], lw=2, ls='*', marker='s', color='black')
  # result
  ax.errorbar(ptx, pq[r,0], yerr=[pq[r,2], pq[r,1]],
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

print("Done.")

# Make a triangle plot.
# figc = triangle.corner(samples[:, 0:ndim/2])
# figc.savefig("ctri.png")
# figb = triangle.corner(samples[:, ndim/2:ndim])
# figb.savefig("btri.png")
