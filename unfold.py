import os, sys
import numpy as np
import emcee
import lnpmodels as lnp
import unfold_input as ui
import plotting_functions as pf
from refold import ept_refold, dca_refold

#--------------------------------------------------------------------------
# Setup/configuration
#--------------------------------------------------------------------------
step = 2 # 0: PYTHIA + bfrac used as priors. 1+: use previous output.
bfrac = 0.007
use_all_data = False
alpha = 0.2 # Regularization parameter
nwalkers = 500
nburnin = 1000
nsteps = 1000
dtype = 'AuAu200MB' # 'AuAu200MB' 'pp200' 'MC'
dcares = {'AuAu200MB' : 0.007, 'pp200' : 0.01, 'MC' : 0.0}
ndim = ui.nhpt
c, b = ui.idx['c'], ui.idx['b']
x_ini = None

# Output locations
pdfdir = 'pdfs/{}/{}/'.format(dtype, step + 1)
csvdir = 'csv/{}/{}/'.format(dtype, step + 1)

if use_all_data == False:
    pdfdir = 'pdfs/spectra_only/{}/{}/'.format(dtype, step + 1)
    csvdir = 'csv/spectra_only/{}/{}/'.format(dtype, step + 1)

if not os.path.isdir(pdfdir): os.makedirs(pdfdir)
if not os.path.isdir(csvdir): os.makedirs(csvdir)

if use_all_data:
    print 'Using combined spectra + DCA datasets. Datatype:', dtype
else:
    print 'Using electron spectra only. Datatype:', dtype

# Create text files for matrix creation (project_and_save() is idempotent).
ui.project_and_save(dcares[dtype])

# Weighted matrices - elements are joint probabilities
eptmat = ui.eptmatrix()
dcamat = [ui.dcamatrix(i) for i in range(6)]

# PHENIX data - (21 x 2) - column 0 (1) contains data (error).
ept_mb = ui.eptdata('AuAu200MB')
ept_pp = ui.eptdata('pp200')
ept_py = ui.eptdata_sim(bfrac, ept_pp[:,0].sum(), 'pp200')

# Set ept and dca to selected data type
ept, dca = None, []
if dtype == 'MC':
    ept = ept_py
    dca = [ui.dcadata_sim(i, bfrac) for i in range(6)]
elif dtype == 'AuAu200MB':
    ept = ept_mb
    dca = [ui.dcadata(i, dtype) for i in range(6)]
elif dtype == 'pp200':
    ept = ept_pp
    dca = [ui.dcadata(i, dtype) for i in range(6)]

# Chop out rows matching excluded DCA bins
subdca, subdcamat = ui.dca_subset(dca, dcamat, dtype)

# Generated pythia inclusive hadron pt.
# Used for MCMC initial point, for regularization, and for comparison to
# result.
gpt_full = ui.genpt(bfrac)
gpt = gpt_full[:,0] # Extract column with data, leaving out errors.

# Normalize pythia spectra to data
norm_factor = np.sum(ept[:, 0]) / np.sum(ui.eptmat_proj(bfrac, axis=1)[:,0])
gpt *= norm_factor

if step == 0:
    x_ini = gpt
else:
    csvi = 'csv/{}/{}/pq.csv'.format(dtype, step)
    x_ini = np.loadtxt(csvi, delimiter=',')[:,0]

# Create a combined electron pt + electron DCA data and matrix list
datalist = [ept]
[datalist.append(d) for d in subdca]
matlist = [eptmat]
[matlist.append(m) for m in subdcamat]

#--------------------------------------------------------------------------
# Run sampler
#--------------------------------------------------------------------------
# Set parameter limits - put in array with shape (ndim,2)
parlimits = np.vstack((0.001 * x_ini, 5.0 * x_ini)).T

# Matrix used to define seminorm for regularization
L = np.hstack((lnp.fd2(ui.ncpt), lnp.fd2(ui.nbpt)))

# Ensemble of starting points for the walkers - shape (nwalkers, ndim)
x0 = np.zeros((nwalkers, ndim))
print("Initializing {} {}-dim walkers...".format(nwalkers, ndim))
x0[:, c] = x_ini[c] * (1 + 0.1 * np.random.randn(nwalkers, ui.ncpt))
x0[:, b] = x_ini[b] * (1 + 0.1 * np.random.randn(nwalkers, ui.nbpt))

# Function returning values \propto posterior probability and arg tuple
fcn, args = None, None

if use_all_data:
    # Compute contribution to ln(L) from DCA vs electron pt using x0
    print 'Computing initial likelihood estimates...'
    ll_ept = np.zeros((nwalkers,))
    ll_dca = np.zeros((nwalkers,))
    preds = [np.zeros((nwalkers,subdca[i].shape[0])) for i in range(6)]
    for i in range(nwalkers):
        x = x0[i,:]
        ll_ept[i] = lnp.l2_gaussian(x, matlist[0], datalist[0], 
                                       x_ini, alpha, parlimits, L)
        ll_dca[i] = lnp.l2_poisson_shape(x, matlist[1:], datalist[1:], 
                                            x_ini, alpha, parlimits, L)
        for j in range(lnp.dca_shape.prediction.shape[0]):
            p = lnp.dca_shape.prediction[j,:preds[j].shape[1]]
            preds[j][i,:] = p

    le, ld = np.mean(ll_ept), np.mean(ll_dca)
    print 'e spectrum ln(L) initial estimate:', le, np.std(ll_ept)
    print 'e DCA sum  ln(L) initial estimate:', ld, np.std(ll_dca)

    # eptw, dcaw = ld/(le+ld), le/(le+ld)
    eptw, dcaw = 0.5, 0.5
    print 'Spectrum weight factor:', eptw
    print 'DCA weight factor:', dcaw

    # Write out initial predictions for watch_predictions.py animation
    for i in range(6):
        np.savetxt('csv/preds{}.csv'.format(i), preds[i], 
                   fmt='%.2f', delimiter=',')
    # sys.exit()
    fcn = lnp.logp_ept_dca
    dataweights = (eptw, dcaw)
    args = (matlist, datalist, dataweights, x_ini, alpha, parlimits, L)
else:
    fcn = lnp.l2_gaussian
    args = (eptmat, ept, x_ini, alpha, parlimits, L)

sampler = emcee.EnsembleSampler(nwalkers, ndim, fcn, args=args, threads=2)

print("Burning in for {} steps...".format(nburnin))
pos, prob, state = sampler.run_mcmc(x0, nburnin)
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
np.savetxt("{}pq.csv".format(csvdir), pq, delimiter=",")

#--------------------------------------------------------------------------
# Plot results
#--------------------------------------------------------------------------
ceptr, beptr, heptr, bfrac_ept = ept_refold(pq, eptmat)
cdcar, bdcar, hdcar, bfrac_dca = dca_refold(pq, dcamat, dca, add_bkg=True)

pf.plotept_refold(ept, ceptr, beptr, heptr, pdfdir + 'ept_refold.pdf')
pf.plotbfrac(bfrac_ept, None, ui.fonll, pdfdir + 'bfrac-ept.pdf')

pf.plot_result(parlimits, x0, gpt, pq, pdfdir + 'hpt.pdf')
pf.plot_post_marg(samples, parlimits, pdfdir + 'posterior.pdf')
pf.plot_lnprob(sampler.flatlnprobability, pdfdir + 'lnprob.pdf')
pf.plot_lnp_steps(sampler, nburnin, pdfdir + 'lnprob-vs-step.pdf')
pf.plot_ept(0.1 * ept_mb, ept_pp, ept_py, pdfdir + 'ept-comparison.pdf')
pf.plotdca_fold(dca, cdcar, bdcar, hdcar, pdfdir + 'dca-fold.pdf')
pf.plotbfrac(bfrac_ept, bfrac_dca, ui.fonll, pdfdir + 'bfrac.pdf')
