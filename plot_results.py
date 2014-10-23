import numpy as np
import unfold_input as ui
import plotting_functions as pf

dtype = 'MC'
c, b, f = ui.idx['c'], ui.idx['b'], ui.idx['f']

# Weighted matrices - elements are joint probabilities
eptmat = ui.eptmatrix()
dcamat = [ui.dcamatrix(i) for i in range(6)]
gpt = ui.genpt()

# Model data from pythia - fully self-consistent problem for testing
bfrac = 0.0073
ept_py = ui.eptmat_proj(bfrac, axis=1)
dca_py = [ui.dcadata_sim(i, bfrac) for i in range(6)]
# dca_py = [ui.dcamat_proj(i, bfrac, axis=1) for i in range(6)]

# Set ept and dca to selected data type
ept = ept_py if dtype == 'MC' else ui.eptdata(dtype)
dca = dca_py if dtype == 'MC' else [ui.dcadata(i, dtype) for i in range(6)]

# Get unfold results
pq = np.loadtxt('csv/pq_MC.csv', delimiter=',')
bfracs = pq[f,:]

# Refold arrays have shape (neptx, 3). Cols: mid, ehi, elo
ceptr = (1. - bfracs[-1,0]) * np.dot(eptmat[:, c], pq[c, :])
beptr = bfracs[-1,0] * np.dot(eptmat[:, b], pq[b, :])
heptr = ceptr + beptr
bfspec = beptr / heptr
bfdca = bfracs[:-1, :]

# Estimate error on bfspec - should really use something like BayesDivide
bfspec[:,1] = beptr[:,1]/heptr[:,0] 
bfspec[:,2] = beptr[:,2]/heptr[:,0] 

dir = 'pdfs/test/'
pf.plotbfrac(bfspec, bfdca, dir + 'bfrac.pdf')

if True:
    cfold = []
    bfold = []
    hfold = []
    for i, m in enumerate(dcamat):
        bf = pq[f[i], 0]
        cfold.append((1 - bf) * np.dot(m[:, c], gpt[c]))
        bfold.append(bf * np.dot(m[:, b], gpt[b]))
        hfold.append(cfold[i] + bfold[i])

        datasum = dca[i][:48,0].sum() + dca[i][52:,0].sum()
        foldsum = hfold[i][:48].sum() + hfold[i][52:].sum()
        normfac = datasum / foldsum
        # nf = dca[i][:,0].sum() / hfold[i].sum()
        hfold[i] *= normfac
        cfold[i] *= normfac
        bfold[i] *= normfac
    pf.plotdca_fold(dca, cfold, bfold, hfold, 'pdfs/test/dca-fold.pdf')




# pf.plot_bfrac_samples(samples[:, -1], bfrac_result, dir + 'bfrac_samples.pdf')
# pf.plotept_refold(ept, ceptr, beptr, heptr, dir + 'ept_refold.pdf')
# pf.plot_result(parlimits[:-1, :], p0[:, :-1], gpt, pq, dir + 'hpt.pdf')
# pf.plot_post_marg(samples[:, :-1], dir + 'posterior.pdf')
# pf.plot_lnprob(sampler.flatlnprobability, dir + 'lnprob.pdf')
# pf.plot_lnp_steps(sampler, nburnin, dir + 'lnprob-vs-step.pdf')
# pf.plot_ept(0.1 * ept_mb, ept_pp, ept_py, dir + 'ept-comparison.pdf')

