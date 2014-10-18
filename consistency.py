'''
consistency.py: 
1. Plot model and data inputs
2. Check for consistency between models for two different datasets
3. Check for consistency between data and model.
If large inconsistencies are seen, unfolding is unlikely to work.
'''
import os
import numpy as np
from matrixplotter import mmplot
import unfold_input as ui
import plotting_functions as pf

bfrac = 0.007
# Unweighted matrices - matrix elements contain counts
eptmat = ui.eptmatrix(bfrac, False)
dcamat = [ui.dcamatrix(bfrac, i, False) for i in range(6)]
# Weighted matrices - elements are joint probabilities
eptmatw = ui.eptmatrix(bfrac, True)
dcamatw = [ui.dcamatrix(bfrac, i, True) for i in range(6)]
# Generated inclusive hadron pt and ideal data for consistency check
gpt = ui.genpt()
hpt_ideal = eptmat.sum(axis=0)
ept_ideal = eptmat.sum(axis=1)
dca_ideal = [m.sum(axis=1) for m in dcamat]
ept_err = np.sqrt(ept_ideal)  # Whatever

if True:
    z = np.zeros(ui.ndim / 2)
    # B fraction from electron pt spectra
    bfold_ept = np.dot(eptmatw, np.hstack([z, gpt[ui.ndim / 2:]]))
    hfold_ept = np.dot(eptmatw, gpt)
    bfrac_ept = bfold_ept / hfold_ept
    # B fraction from DCA distributions at different pt values
    hfold_dca = [np.dot(m, gpt) for m in dcamatw]
    cfold_dca = [np.dot(m, np.hstack([gpt[:ui.ndim / 2], z])) for m in dcamatw]
    bfold_dca = [np.dot(m, np.hstack([z, gpt[ui.ndim / 2:]])) for m in dcamatw]
    bfrac_dca = np.array([np.sum(b) / np.sum(h)
                          for b, h in zip(bfold_dca, hfold_dca)])
    pf.plotbfrac(bfrac_ept, bfrac_dca, 'pdfs/bfrac-fold.pdf')

if True:
    z = np.zeros(ui.ndim / 2)
    hfold = [np.dot(m, gpt) for m in dcamatw]
    cfold = [np.dot(m, np.hstack([gpt[:ui.ndim / 2], z])) for m in dcamatw]
    bfold = [np.dot(m, np.hstack([z, gpt[ui.ndim / 2:]])) for m in dcamatw]
    pf.plotdca_fold(dca_ideal, cfold, bfold, hfold, 'pdfs/dca-fold.pdf')

if True:
    hptd = [m.sum(axis=0) for m in dcamat]
    pf.plothpt(gpt, hpt_ideal, hptd, 'pdfs/hpt-gen.pdf')

if True:
    z = np.zeros(ui.ndim / 2)
    cept = eptmat[:, :ui.ndim / 2].sum(axis=1)
    bept = eptmat[:, ui.ndim / 2:].sum(axis=1)
    cfold = np.dot(eptmatw, np.hstack([gpt[:ui.ndim / 2], z]))
    bfold = np.dot(eptmatw, np.hstack([z, gpt[ui.ndim / 2:]]))
    hfold = np.dot(eptmatw, gpt)
    pf.plotept_fold(ept_ideal, ept_err, cept, bept, cfold, bfold, hfold,
                    'pdfs/ept-fold.pdf')

if True:
    xl = r'c hadron $p_T$ [GeV/c] $\qquad$ b hadron $p_T$ [GeV/c]'
    # Plot hpt -> ept matrices (unweighted and weighted)
    for mat, w in zip([eptmat,eptmatw],['counts','probs']):
        print "ept mmplot()", w
        mmplot(mat, ui.hptx, ui.eptx, ui.hptbins, ui.eptbins,
               xlabel=xl,
               ylabel=r'$e^{\pm}$ $p_T$ [GeV/c]',
               desc=[r'$h_{c,b}\, p_T\, \to\, e^{\pm}\, p_T$'],
               figname='pdfs/eptmat_' + w + '.pdf')
    # Plot hpt -> dca,ept matrices (unweighted and weighted)
    for mat, w in zip([dcamat,dcamatw],['counts','probs']):
        for i, m in enumerate(mat):
            print "DCA mmplot()", w, i
            pt = (ui.dcaeptbins[i], ui.dcaeptbins[i + 1])
            desc1 = r'$h_{c,b}\, p_T\, \to\, e^{\pm}$ DCA'
            desc2 = r'$e\, p_T \in\, [{:.1f},{:.1f}]\, GeV/c$'.format(*pt)
            mmplot(m, ui.hptx, ui.dcax, ui.hptbins, ui.dcabins,
                   xlabel=xl,
                   ylabel=r'$e^{\pm}$ DCA [cm]',
                   desc=[desc1,desc2],
                   figname='pdfs/dcamat_{}_{}.pdf'.format(w,i))    

    # os.system("pdftk pdfs/*.pdf cat output all.pdf")
