import numpy as np
import unfold_input as ui
import plotting_functions as pf
from refold import ept_refold, dca_refold

dtype = 'AuAu200MB'  # 'pp200'
pdfdir = 'pdfs/test/'

# Get unfold results
csv = 'csv/{}/pq_{}.csv'.format(dtype, dtype)
pq = np.loadtxt(csv, delimiter=',')

eptmat = ui.eptmatrix()
dcamat = [ui.dcamatrix(i) for i in range(6)]

ept_dict = {
    'AuAu200MB': ui.eptdata('AuAu200MB'),
    'pp200': ui.eptdata('pp200')}
ept = ept_dict[dtype]
dca = [ui.dcadata(i, dtype) for i in range(6)]

ceptr, beptr, heptr, bfrac_ept = ept_refold(pq, eptmat)

pf.plotept_refold(ept, ceptr, beptr, heptr, pdfdir + 'ept_refold.pdf')
pf.plotbfrac(bfrac_ept, None, pdfdir + 'bfrac-ept.pdf')

cdcar, bdcar, hdcar, bfrac_dca = dca_refold(pq, dcamat, dca, add_bkg=True)
pf.plotdca_fold(dca, cdcar, bdcar, hdcar, pdfdir + 'dca-fold.pdf')
pf.plotbfrac(bfrac_ept, bfrac_dca, pdfdir + 'bfrac.pdf')

# pf.plot_result(parlimits, x0, gpt, pq, pdfdir + 'hpt.pdf')
# pf.plot_post_marg(samples, pdfdir + 'posterior.pdf')
# pf.plot_lnprob(sampler.flatlnprobability, pdfdir + 'lnprob.pdf')
# pf.plot_lnp_steps(sampler, nburnin, pdfdir + 'lnprob-vs-step.pdf')
# pf.plot_ept(0.1 * ept_mb, ept_pp, ept_py, pdfdir + 'ept-comparison.pdf')
