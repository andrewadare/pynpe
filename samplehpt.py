'''
samplehpt.py
Sample from joint probability to find heavy-flavor hadron yields vs pt
'''
from __future__ import print_function
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import norm
from time import clock
import emcee
import triangle
import lnpmodels
import raamodel
import npe_io as io

# Configuration
use_all_data = True
alpha = 0.2
nwalkers = 500
nburnin = 2000
nsteps = 1000
dtype = 'MC'
dcares = 0.007  # 0.007 cm in MB Au+Au, 0.014 cm in p+p.
bfrac = 0.007

io.mf = io.modelfile(dcares, bfrac)

# Get model components
eptmat, dcamat = io.matrices()
gpt, gptx, gptbins = io.genpt()
ndim = eptmat.shape[1]
raa = raamodel.getraa(gptx, ndim / 2)
gptmod = gpt * raa
eptmod = np.dot(eptmat, gptmod)
ept_err = np.sqrt(eptmod)
eptx, eptbins = io.eptbins()
gptw = io.binwidths(gptbins)
eptw = io.binwidths(eptbins)

dcax, dcabins, dcaeptx, dcaeptbins = io.dcabins()

# Get data
dca = np.zeros(dcamat[0].shape[0])
dcamod = np.zeros(dcamat[0].shape[0])
ept = np.zeros(eptmat.shape[0])
ept_err = np.zeros(eptmat.shape[0])
if dtype == 'MC':
    # statistics of QM12 DCA distributions:
    qm12ppbkg = [17948, 3386,  776, 212,  87, 13]
    qm12pptot = [29492, 7317, 2221, 811, 482, 98]
    qm12mbbkg = [91148, 11388, 2148, 556, 191, 34]
    qm12mbtot = [181649, 31795, 7172, 2171, 1010, 170]

    ept = io.eptmatrix().sum(axis=1)
    dca = [m.sum(axis=1) for m in io.dcamatrices()]
    dcamod = [np.dot(m, gptmod) for m in io.dcamatrices()]
    for i,d in enumerate(dcamod):
        d *= qm12mbtot[i]/d.sum()
        for j,mu in enumerate(d):
            d[j] = np.random.poisson(mu)
        dcamod[i] = d
        np.savetxt("csv/dcamod%d.csv" % i, d, delimiter=",")
sys.exit(0)
# ########################################################################3
# def plotdca_dists():
#     print("plotdca_dists()")
#     nr, nc = 2, 3
#     fig, axes = plt.subplots(nr, nc)
#     for row in range(nr):
#         for col in range(nc):
#             i = nc * row + col
#             a = axes[row, col]
#             # a.set_yscale('log')
#             # a.set_ylim([1, 1.2 * np.max(hptd[i])])
#             a.set_xlim([dcabins[0], dcabins[-1]])
#             a.tick_params(axis='x', top='off', labelsize=6)
#             a.tick_params(axis='y', labelsize=6)
#             s = r'{0:.1f}-{1:.1f} GeV/c'.format(
#                 dcaeptbins[i], dcaeptbins[i + 1])
#             a.text(0.55, 0.9, s, fontsize=8, transform=a.transAxes)
#             # a.step(dcax, hfold_dca[i], lw=1, alpha=0.8, color='crimson')
#             # a.step(dcax, cfold_dca[i], lw=1, alpha=0.8, color='darkorange')
#             # a.step(dcax, bfold_dca[i], lw=1, alpha=0.8, color='dodgerblue')
#             # if False:
#             #     a.step(dcax, bkg[i], color='brown')
#             #     a.step(dcax, dca[i], color='black', alpha=0.6)
#             if True:
#                 a.bar(dcabins[:-1], dcamod[i], width=dcabins[1]-dcabins[0], 
#                       log=True, color='red', edgecolor='red', linewidth=0, alpha=0.6)
#     fig.savefig('pdfs/dca_dists_mod.pdf', bbox_inches='tight')
#     return
# plotdca_dists()
# sys.exit(0)
########################################################################3


# Create a combined electron pt + electron DCA data list.
alldata = [eptmod]
[alldata.append(d) for d in dcamod]
# Create a combined electron pt + electron DCA matrix list.
allmats = [eptmat]
[allmats.append(m) for m in dcamat]

# # Rebin over all c or b hadron pt. 100 x 20 --> 100 x 2 matrices
# for m in dcamat:
#     cdca = m[:, :ndim / 2].sum(axis=1)
#     bdca = m[:, ndim / 2:].sum(axis=1)
#     allmats.append(np.c_[cdca, bdca])
#  #   print(allmats[-1].shape)

# Ensemble of starting points for the walkers.
print("Initializing {} {}-dim walkers...".format(nwalkers, ndim))
p0 = gpt * (1. + 0.1 * np.random.randn(nwalkers, ndim))

ymin = 0.01 * gpt
ymax = 2 * gpt
L = lnpmodels.fd2(ndim)
fcn = None  # Function returning values \propto posterior probability
args = None  # Argument list for fcn

if use_all_data:
    # Use electron pt + 6 electron DCA datasets.
    # alpha = 0.01
    fcn = lnpmodels.l2_poisson_combined
    args = [allmats, alldata, gpt, alpha, ymin, ymax, L]

else:
    # Use single electron pt dataset
    fcn = lnpmodels.l2_poisson
    args = [eptmat, eptmod, gpt, alpha, ymin, ymax, L]

sampler = emcee.EnsembleSampler(nwalkers, ndim, fcn, args=args, threads=2)

print("Burning in for {} steps...".format(nburnin), end=' ')
sys.stdout.flush()
start = clock()
pos, prob, state = sampler.run_mcmc(p0, nburnin)
sampler.reset()
end = clock()
print(str(end - start) + ' s')

print("Running sampler for {} steps...".format(nsteps), end=' ')
sys.stdout.flush()
start = clock()
sampler.run_mcmc(pos, nsteps)
end = clock()
print(str(end - start) + ' s')

acc_frac = np.mean(sampler.acceptance_fraction)
print("Mean acceptance fraction: {0:.3f}".format(acc_frac))

# Initial shape is (nwalkers, nsteps, ndim). Reshape to (nwalkers*nsteps,
# ndim).
samples = sampler.chain.reshape((-1, ndim))

# Posterior quantiles: list of ndim (16,50,84) percentile tuples
pq = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
         zip(*np.percentile(samples, [16, 50, 84], axis=0)))
pq = np.array(pq)

# -----------------------------------------------------------------------------
#                            Plot results
# -----------------------------------------------------------------------------

# Draw posterior marginal distributions
# sample.chain has shape (nwalkers, nsteps, ndim)
print("Marginalizing posterior distribution for summary plot...")
nr, nc = 4, 5
fig, axes = plt.subplots(nr, nc)
for row in range(nr):
    for col in range(nc):
        i = nc * row + col
        a = axes[row, col]
        fc = 'yellow' if i < ndim / 2 else 'lime'
        a.hist(sampler.flatchain[:, i], 100,
               color='k', facecolor=fc, histtype='stepfilled')
        a.tick_params(axis='x', top='off', labelsize=4)
        a.tick_params(axis='y', left='off', right='off', labelsize=0)
        a.xaxis.get_offset_text().set_size(4)
        a.xaxis.get_major_formatter().set_powerlimits((0, 1))
        s = r'{0:.0f}-{1:.0f} GeV/c'.format(
            gptbins[i % 10], gptbins[i % 10 + 1])
        a.text(0.6, 0.9, s, fontsize=5, transform=a.transAxes)
        if i < ndim / 2:
            a.text(
                0.05, 0.9, r'$h_{charm}$', fontsize=5, transform=a.transAxes)
        else:
            a.text(
                0.05, 0.9, r'$h_{beauty}$', fontsize=5, transform=a.transAxes)
fig.savefig('pdfs/posterior.pdf', bbox_inches='tight')

print("Plotting log probabilities...")
fig, ax = plt.subplots()
ax.set_xlabel('log likelihood distribution')
ax.set_ylabel('samples')
ax.hist(sampler.flatlnprobability, 100,
        color='k', facecolor='lightyellow', histtype='stepfilled')
fig.savefig('pdfs/lnprob.pdf')

print("Plotting log probabilities vs step...")
fig, ax = plt.subplots()
ax.set_xlabel('step')
# ax.set_ylabel(r'$\langle$log likelihood$\rangle$')
ax.set_ylabel(r'$\langle \ln(L) \rangle_{chains}$')
ax.set_title(r'$\langle \ln(L) \rangle_{chains}$ vs. sample step')
ax.plot(np.sum(sampler.lnprobability / nwalkers, axis=0), color='k')
ax.text(0.05, 0.95, '{} chains after {} burn-in steps'.format(nwalkers, nburnin),
        transform=ax.transAxes)
fig.savefig('pdfs/lnprob-vs-step.pdf')

# Draw prior, walkers, initial guess, and result.
print("Drawing results...")
fig, axes = plt.subplots(1, 2)
ptx = gptx[:ndim / 2]
for ax in axes:
    cb = 'c' if ax == axes[0] else 'b'
    r = range(ndim / 2) if ax == axes[0] else range(ndim / 2, ndim)
    w = gptw[r]
    ax.set_yscale('log')
    ax.set_xlabel(r'{} hadron $p_T$ [GeV/c]'.format(cb))
    ax.fill_between(
        ptx, ymin[r] / w, ymax[r] / w, color='slategray', alpha=0.1)
    for i in range(nwalkers):
        ax.plot(ptx, p0[i, r] / w, ls='*', marker='s',
                ms=14, color='deepskyblue', alpha=0.01)
    ax.plot(ptx, gpt[r] / w, lw=2, ls='*', marker='o', color='white')
    ax.plot(ptx, gptmod[r] / w, lw=2, ls='*', marker='s', color='black')
    ax.errorbar(ptx, pq[r, 0] / w, yerr=[pq[r, 2] / w, pq[r, 1] / w],
                ls='*', fmt='o', color='crimson', ecolor='crimson', capthick=2)
fig.savefig('pdfs/hpt.pdf')
np.savetxt("csv/pq.csv", pq, delimiter=",")

# Draw ept
eptrefold = np.dot(eptmat, pq[:, 0])
eptref_hi = np.dot(eptmat, pq[:, 1])
eptref_lo = np.dot(eptmat, pq[:, 2])

fig, ax = plt.subplots()
ax.set_yscale('log')
ax.set_xlabel(r'$e^{\pm}$ $p_T$ [GeV/c]')
ax.errorbar(eptx, ept / eptw, yerr=ept_err / eptw,
            lw=2, ls='*', marker='o', color='white')
ax.errorbar(eptx, eptrefold / eptw, yerr=[eptref_lo / eptw, eptref_hi / eptw],
            lw=2, ls='*', marker='s', ms=10, color='r')
ax.errorbar(eptx, eptmod / eptw, yerr=ept_err / eptw,
            lw=2, ls='*', marker='o', color='k')
fig.savefig('pdfs/ept.pdf')

print("Done.")

# Make a triangle plot.
# figc = triangle.corner(samples[:, 0:ndim/2])
# figc.savefig("ctri.png")
# figb = triangle.corner(samples[:, ndim/2:ndim])
# figb.savefig("btri.png")
