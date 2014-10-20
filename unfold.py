import numpy as np
import emcee
import lnpmodels
import unfold_input as ui
import plotting_functions as pf

#--------------------------------------------------------------------------
# Setup/configuration
#--------------------------------------------------------------------------
use_all_data = False
alpha = 0.2
nwalkers = 500
nburnin = 500
nsteps = 500
dtype = 'MC'  # 'AuAu200MB'  # 'pp200' 'MC'
bfrac = 0.0073
ndim = ui.nhpt + 1

# Weighted matrices - elements are joint probabilities
eptmat = ui.eptmatrix()
dcamat = [ui.dcamatrix(i) for i in range(6)]

# PHENIX data - (21 x 2) - column 0 (1) contains data (error).
ept_mb = ui.eptdata('AuAu200MB')
ept_pp = ui.eptdata('pp200')

# Model data from pythia - fully self-consistent problem for testing
ept_py = ui.eptmat_proj(bfrac, axis=1)
dca_py = [ui.dcamat_proj(i, bfrac, axis=1) for i in range(6)]

# Set ept to selected data type
ept = ept_py if dtype == 'MC' else ui.eptdata(dtype)
# TODO: set dca options ##########################################
dca = dca_py

# Generated pythia inclusive hadron pt.
# Used for MCMC initial point, for regularization, and for comparison to
# result.
gpt = ui.genpt()

# Normalize pythia spectra to PHENIX p+p data
norm_factor = np.sum(ept_pp[5:, 0]) / np.sum(ept_py[5:])
ept_py *= norm_factor
gpt *= norm_factor

# Create a combined electron pt + electron DCA data list,
# same for matrices.
alldata = [ept]
[alldata.append(d) for d in dca]
allmats = [eptmat]
[allmats.append(m) for m in dcamat]

#--------------------------------------------------------------------------
# Run sampler
#--------------------------------------------------------------------------
# Ensemble of starting points for the walkers - shape (nwalkers, ndim)
print("Initializing {} {}-dim walkers...".format(nwalkers, ndim))
p0 = np.zeros((nwalkers, ndim))
p0[:, :-1] = gpt * (1 + 0.1 * np.random.randn(nwalkers, ui.nhpt))
p0[:, -1] = bfrac * (1 + 0.1 * np.random.randn(nwalkers))

# Parameter limits - shape (ndim,2)
parlimits = np.vstack((0.01 * gpt, 2.0 * gpt)).T
parlimits = np.vstack((parlimits, (0.001, 0.02)))

L = np.hstack((lnpmodels.fd2(ui.ncpt), lnpmodels.fd2(ui.nbpt)))
fcn = None  # Function returning values \propto posterior probability
args = None  # Argument list for fcn
if use_all_data:
    # Use electron pt + 6 electron DCA datasets.
    fcn = lnpmodels.l2_poisson_combined
    args = [allmats, alldata, gpt, alpha, ymin, ymax, L]

else:
    # Gaussian likelihood model
    icov_data = np.diag(1. / (ept[:, 1] ** 2))
    fcn = lnpmodels.l2_gaussian
    args = [eptmat, ept[:, 0], icov_data, gpt, alpha, parlimits, L]

    # Poisson likelihood model
    # fcn = lnpmodels.l2_poisson
    # args = [eptmat, ept[:,0], gpt, alpha, ymin, ymax, L]

sampler = emcee.EnsembleSampler(nwalkers, ndim, fcn, args=args, threads=2)

print("Burning in for {} steps...".format(nburnin))
pos, prob, state = sampler.run_mcmc(p0, nburnin)
sampler.reset()
print("Running sampler for {} steps...".format(nsteps))
sampler.run_mcmc(pos, nsteps)
acc_frac = np.mean(sampler.acceptance_fraction)
print("Mean acceptance fraction: {0:.3f}".format(acc_frac))

# Initial shape of sampler.chain is (nwalkers, nsteps, ndim).
# Reshape to (nwalkers*nsteps, ndim).
# Posterior quantiles: list of ndim (16,50,84) percentile tuples
samples = sampler.chain.reshape((-1, ndim))
pq = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
         zip(*np.percentile(samples, [16, 50, 84], axis=0)))
pq = np.array(pq)
np.savetxt("csv/pq_{}.csv".format(dtype), pq, delimiter=",")
print sampler.flatchain.shape, samples.shape
bfrac_result = pq[-1, :]  # mid,errhi,errlo array
print 'b/(b+c) fraction: {:.3g} + {:.3g} - {:.3g}'.format(*bfrac_result)

#--------------------------------------------------------------------------
# Plot results
#--------------------------------------------------------------------------

# Refold arrays have shape (neptx, 3). Cols: mid, ehi, elo
c, b = ui.idx['c'], ui.idx['b']
ceptr = (1. - bfrac_result[0]) * np.dot(eptmat[:, c], pq[c, :])
beptr = bfrac_result[0] * np.dot(eptmat[:, b], pq[b, :])
heptr = ceptr + beptr

dir = 'pdfs/test/'
pf.plot_bfrac_samples(samples[:, -1], bfrac_result, dir + 'bfrac_samples.pdf')
pf.plotept_refold(ept, ceptr, beptr, heptr, dir + 'ept_refold.pdf')
pf.plot_result(parlimits[:-1, :], p0[:, :-1], gpt, pq, dir + 'hpt.pdf')
pf.plot_post_marg(samples[:, :-1], dir + 'posterior.pdf')
pf.plot_lnprob(sampler.flatlnprobability, dir + 'lnprob.pdf')
pf.plot_lnp_steps(sampler, nburnin, dir + 'lnprob-vs-step.pdf')
pf.plot_ept(0.1 * ept_mb, ept_pp, ept_py, dir + 'ept-comparison.pdf')
