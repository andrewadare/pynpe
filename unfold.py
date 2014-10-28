import os, sys
import numpy as np
import emcee
import lnpmodels
import unfold_input as ui
import plotting_functions as pf


#--------------------------------------------------------------------------
# Setup/configuration
#--------------------------------------------------------------------------
use_all_data = True
alpha = [0.2, 2.0] # Regularization parameters for [spectra, dca]
nwalkers = 500
nburnin = 2500
nsteps = 500
dtype = 'AuAu200MB'  # 'pp200' 'MC'
bfrac = 1e-4 #0.0073
dcares = {'AuAu200MB' : 0.007, 'pp200' : 0.01, 'MC' : 0.0}
n_bfrac_pars = 7 if use_all_data else 1
ndim = ui.nhpt + n_bfrac_pars
c, b, f = ui.idx['c'], ui.idx['b'], ui.idx['f']

# Output locations
pdfdir = 'pdfs/' + dtype + '/'
csvdir = 'csv/' + dtype + '/'
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

# for i in range(6):
#     print subdca[i].shape, subdcamat[i].shape

# Generated pythia inclusive hadron pt.
# Used for MCMC initial point, for regularization, and for comparison to
# result.
gpt = ui.genpt()

# Normalize pythia spectra to data
norm_factor = np.sum(ept[:, 0]) / np.sum(ui.eptmat_proj(bfrac, axis=1)[:,0])
# ept_py *= norm_factor
gpt *= norm_factor

# Create a combined electron pt + electron DCA data and matrix list
datalist = [ept]
[datalist.append(d) for d in subdca]
matlist = [eptmat]
[matlist.append(m) for m in subdcamat]

#--------------------------------------------------------------------------
# Run sampler
#--------------------------------------------------------------------------
# Set parameter limits - put in array with shape (ndim,2)
hpt_parlimits = np.vstack((0.001 * gpt, 5.0 * gpt)).T
bfrac_parlimits = np.vstack((1e-6 * np.ones(n_bfrac_pars),
                             (1. - 1e-6) * np.ones(n_bfrac_pars))).T
parlimits = np.vstack((hpt_parlimits, bfrac_parlimits))

# Smoothing matrix
L = np.hstack((lnpmodels.fd2(ui.ncpt), lnpmodels.fd2(ui.nbpt)))

# Ensemble of starting points for the walkers - shape (nwalkers, ndim)
x0 = np.zeros((nwalkers, ndim))
print("Initializing {} {}-dim walkers...".format(nwalkers, ndim))
x0[:, c] = gpt[c] * (1 + 0.1 * np.random.randn(nwalkers, ui.ncpt))
x0[:, b] = gpt[b] * (1 + 0.1 * np.random.randn(nwalkers, ui.nbpt))
if use_all_data:
    bfrac_prior = np.array([0.0137, 0.0343, 0.0737, 
                           0.137, 0.246, 0.418, 0.0073])
    x0[:, f] = bfrac_prior * (np.random.beta(2.,2.,(nwalkers, ui.nfb)))
    # x0[:, f] = bfrac_prior * (1 + 0.001 * np.random.randn(nwalkers, ui.nfb))
    # x0[:, f] = np.random.rand(nwalkers, ui.nfb)
else:
    x0[:, -1] = bfrac * (1 + 0.1 * np.random.randn(nwalkers))

# Function returning values \propto posterior probability and arg tuple
fcn, args = None, None

if use_all_data:
    # Compute contribution to ln(L) from DCA vs electron pt using x0
    print 'Computing initial likelihood estimates for balance factors...'
    ll_ept = np.zeros((nwalkers,))
    ll_dca = np.zeros((nwalkers,))
    preds = [np.zeros((nwalkers,subdca[i].shape[0])) for i in range(6)]
    for i in range(nwalkers):
        x = x0[i,:]
        ll_ept[i] = lnpmodels.l2_gaussian(x, matlist[0], datalist[0], 
                                       gpt, alpha[0], parlimits, L)
        ll_dca[i] = lnpmodels.l2_poisson_shape(x, matlist[1:], datalist[1:], 
                                            gpt, alpha[1], parlimits, L)
        for j in range(6):
            p = lnpmodels.l2_poisson_shape.prediction[j,:preds[j].shape[1]]
            preds[j][i,:] = p

    le, ld = np.mean(ll_ept), np.mean(ll_dca)
    eptw, dcaw = ld/(le+ld), le/(le+ld)
    print 'e spectrum ln(L) initial estimate:', le, np.std(ll_ept)
    print 'e DCA sum  ln(L) initial estimate:', ld, np.std(ll_dca)

    ### TEMPORARY
    eptw = 1e-20
    dcaw = 1.0

    print 'Spectrum weight factor:', eptw
    print 'DCA weight factor:', dcaw

    # Write out initial predictions for watch_predictions.py animation
    for i in range(6):
        np.savetxt('csv/preds{}.csv'.format(i), preds[i], 
                   fmt='%.2f', delimiter=',')
    # sys.exit()
    fcn = lnpmodels.logp_ept_dca
    dataweights = (eptw, dcaw)
    args = (matlist, datalist, dataweights, gpt, alpha, parlimits, L)
else:
    fcn = lnpmodels.l2_gaussian
    args = (eptmat, ept, gpt, alpha[0], parlimits, L)
    # Poisson likelihood model
    # fcn = lnpmodels.l2_poisson
    # args = [eptmat, ept[:,0], gpt, alpha[0], ymin, ymax, L]
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
np.savetxt("{}pq_{}.csv".format(csvdir, dtype), pq, delimiter=",")

# b fraction unfold pars: rows are output points. cols: mid, errhi, errlo
# bf_int is integrated over electron pt \in 1-9 GeV/c
bf_int = pq[-1, :]
print 'b/(b+c) fraction: {:.3g} + {:.3g} - {:.3g}'.format(*bf_int)

#--------------------------------------------------------------------------
# Plot results
#--------------------------------------------------------------------------

ceptr = (1. - bf_int[0]) * np.dot(eptmat[:, c], pq[c, :])
beptr = bf_int[0] * np.dot(eptmat[:, b], pq[b, :])
heptr = ceptr + beptr
bfspec = beptr / heptr
bfspec[:,1] = beptr[:,1]/heptr[:,0] 
bfspec[:,2] = beptr[:,2]/heptr[:,0] 

pf.plotept_refold(ept, ceptr, beptr, heptr, pdfdir + 'ept_refold.pdf')
pf.plot_bfrac_samples(samples[:, -1], bf_int, pdfdir + 'bfrac_dist.pdf')
pf.plot_result(parlimits[:-1, :], x0[:, :-1], gpt, pq, pdfdir + 'hpt.pdf')
pf.plot_post_marg(samples[:, :-1], pdfdir + 'posterior.pdf')
pf.plot_lnprob(sampler.flatlnprobability, pdfdir + 'lnprob.pdf')
pf.plot_lnp_steps(sampler, nburnin, pdfdir + 'lnprob-vs-step.pdf')
pf.plot_ept(0.1 * ept_mb, ept_pp, ept_py, pdfdir + 'ept-comparison.pdf')
pf.plotbfrac(bfspec, None, pdfdir + 'bfrac-ept.pdf')

if use_all_data:
    # Refold arrays have shape (neptx, 3). Cols: mid, ehi, elo
    bfracs = pq[f,:]
    bfdca = bfracs[:-1, :]
    # Estimate error on bfspec - TODO: use something like BayesDivide
    pf.plotbfrac(bfspec, bfdca, pdfdir + 'bfrac.pdf')
    cfold = []
    bfold = []
    hfold = []
    for i, m in enumerate(dcamat):
        bf = pq[f[i], 0]
        cfold.append(np.dot(m[:, c], gpt[c]))
        bfold.append(np.dot(m[:, b], gpt[b]))
        cfold[i] *= (1-bf) / cfold[i].sum()
        bfold[i] *= bf / bfold[i].sum()
        hfold.append(cfold[i] + bfold[i])
        datasum = dca[i][:,0].sum()
        bkgsum  = dca[i][:,1].sum()
        foldsum = hfold[i].sum()
        normfac = (datasum - bkgsum) / foldsum
        hfold[i] *= normfac
        cfold[i] *= normfac
        bfold[i] *= normfac
        hfold[i] += dca[i][:,1]

    pf.plotdca_fold(dca, cfold, bfold, hfold, pdfdir + 'dca-fold.pdf')
