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
dtype = 'MC' #'AuAu200MB'  # 'pp200' 'MC'
bfrac = 0.0073
ndim = ui.nhpt + 1

# Weighted matrices - elements are joint probabilities
eptmatw = ui.eptmatrix()
dcamatw = [ui.dcamatrix(bfrac, i, True) for i in range(6)]

# PHENIX data - (21 x 2) - column 0 (1) contains data (error).
ept_mb = ui.eptdata('AuAu200MB')
ept_pp = ui.eptdata('pp200')

# Model data from pythia - fully self-consistent problem for testing
ept_py = ui.eptmat_proj(bfrac, axis=1)
dca_py = [ui.dcamat_proj(i, bfrac, axis=1) for i in range(6)]

# Set ept to selected data type
ept = ept_py if dtype=='MC' else ui.eptdata(dtype)
dca = dca_py # TODO: set dca options ##########################################

# Generated pythia inclusive hadron pt. 
# Used for MCMC initial point and for regularization.
gpt = ui.genpt()
print np.sum(np.dot(eptmatw,gpt)), np.sum(ept[:,0])

# Create a combined electron pt + electron DCA data list,
# same for matrices.
alldata = [ept]
[alldata.append(d) for d in dca]
allmats = [eptmatw]
[allmats.append(m) for m in dcamatw]

#--------------------------------------------------------------------------
# Run sampler
#--------------------------------------------------------------------------
# Ensemble of starting points for the walkers - shape (nwalkers, ndim)
print("Initializing {} {}-dim walkers...".format(nwalkers, ndim))
p0 = np.zeros((nwalkers,ndim))
p0[:,:-1] = gpt * (1 + 0.1 * np.random.randn(nwalkers, ui.nhpt))
p0[:,-1]  = bfrac * (1 + 0.1 * np.random.randn(nwalkers))

# Parameter limits - shape (ndim,2)
parlimits = np.vstack((0.01*gpt,2.0*gpt)).T
parlimits = np.vstack((parlimits,(0.001, 0.02)))

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
    args = [eptmatw, ept[:,0], icov_data, gpt, alpha, parlimits, L]

    # # Poisson likelihood model
    # fcn = lnpmodels.l2_poisson
    # args = [eptmatw, ept[:,0], gpt, alpha, ymin, ymax, L]

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
np.savetxt("csv/pq.csv", pq, delimiter=",")
print sampler.flatchain.shape, samples.shape
#--------------------------------------------------------------------------
# Plot results
#--------------------------------------------------------------------------

# Normalize pythia spectrum to PHENIX p+p data, just for plotting.
ept_py *= np.sum(ept_pp[5:, 0]) / np.sum(ept_py[5:])

pf.plot_result(parlimits[:-1,:], p0[:,:-1], gpt, pq, 'pdfs/test/hpt.pdf')
pf.plot_post_marg(samples[:,:-1], 'pdfs/test/posterior.pdf')
pf.plot_lnprob(sampler.flatlnprobability, 'pdfs/test/lnprob.pdf')
pf.plot_lnp_steps(sampler, nburnin, 'pdfs/test/lnprob-vs-step.pdf')
pf.plot_ept(0.1 * ept_mb, ept_pp, ept_py, 'pdfs/test/ept-comparison.pdf')
