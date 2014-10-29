'''
consistency.py: 
1. Plot model and data inputs
2. Check for consistency between models for two different datasets
3. Check for consistency between data and model.
If large inconsistencies are seen, unfolding is unlikely to work.
'''
import os
import sys
import numpy as np
from matrixplotter import mmplot
import unfold_input as ui
import plotting_functions as pf

np.set_printoptions(precision=3)

bfrac = 0.007
# Decay matrices - elements are joint probabilities
eptmat = ui.eptmatrix()
dcamat = [ui.dcamatrix(i) for i in range(6)]

# Generated inclusive hadron pt and ideal data for consistency check
gpt = ui.genpt(bfrac)
ept_ideal = ui.eptmat_proj(bfrac, axis=1)
dca_ideal = [ui.dcamat_proj(i, bfrac, axis=1) for i in range(6)]
dca = [ui.dcadata_sim(i, bfrac) for i in range(6)]


def dca_refold(gpt, dcamat, dca):
    c, b = ui.idx['c'], ui.idx['b']
    cfold, bfold, hfold = [], [], []
    bfrac = np.zeros((len(dca),3))
    for i, m in enumerate(dcamat):
        cf = np.dot(m[:, c], gpt[c])
        bf = np.dot(m[:, b], gpt[b])
        hf = cf + bf
        scf = dca[i][:, 0].sum() / hf[:, 0].sum()
        cf *= scf
        bf *= scf
        hf *= scf
        cfold.append(cf)
        bfold.append(bf)
        hfold.append(hf)
        bfrac[i,:] = np.sum(bf) / np.sum(hf)

    # Make errors zero until I find time to compute them correctly...
    bfrac[:,1:] *= 0.
    return cfold, bfold, hfold, bfrac


def ept_refold(gpt, eptmat):
    c, b = ui.idx['c'], ui.idx['b']
    cfold = np.dot(eptmat[:, c], gpt[c])
    bfold = np.dot(eptmat[:, b], gpt[b])
    hfold = cfold + bfold
    bfrac = bfold / hfold
    # Make errors zero until I find time to compute them correctly...
    bfrac[:,1:] *= 0.
    return cfold, bfold, hfold, bfrac

if True:
    cfold, bfold, hfold, _ = dca_refold(gpt, dcamat, dca)
    pf.plotdca_fold(dca, cfold, bfold, hfold, 'pdfs/dca-fold.pdf')

if True:
    cfe, bfe, hfe, bfrac_ept = ept_refold(gpt, eptmat)
    cfold, bfold, hfold, bfrac_dca = dca_refold(gpt, dcamat, dca)
    pf.plotbfrac(bfrac_ept, bfrac_dca, 'pdfs/bfrac-fold.pdf')

if True:
    # gpt_counts = ui.genp(None) # No b fraction input here
    hpt_ideal = ui.eptmat_proj(bfrac, axis=0)
    hptd = [ui.dcamat_proj(i, bfrac, axis=0) for i in range(6)]
    pf.plothpt(gpt[:, 0], hpt_ideal[:, 0], hptd, 'pdfs/hpt-gen.pdf')

if True:
    c, b = ui.idx['c'], ui.idx['b']
    m = ui.eptmatrix(weighted=False)
    cept = (1-bfrac)*m[:, c].sum(axis=1)
    bept = bfrac*m[:, b].sum(axis=1)
    cfold = np.dot(eptmat[:, c], gpt[c])
    bfold = np.dot(eptmat[:, b], gpt[b])
    hfold = cfold + bfold
    pf.plotept_fold(ept_ideal, cept, bept, cfold, bfold, hfold,
                    'pdfs/ept-fold.pdf')

if True:
    c, b = ui.idx['c'], ui.idx['b']
    # Plot hpt -> ept matrices
    mmplot(eptmat[:, c], ui.cptx, ui.eptx, ui.cptbins, ui.eptbins,
           xlabel=r'c hadron $p_T$ [GeV/c]',
           ylabel=r'$e^{\pm}$ $p_T$ [GeV/c]',
           desc=[r'$h_{c}\, p_T\, \to\, e^{\pm}\, p_T$'],
           figname='pdfs/eptmat_' + 'c' + '.pdf')
    mmplot(eptmat[:, b], ui.bptx, ui.eptx, ui.bptbins, ui.eptbins,
           xlabel=r'b hadron $p_T$ [GeV/c]',
           ylabel=r'$e^{\pm}$ $p_T$ [GeV/c]',
           desc=[r'$h_{b}\, p_T\, \to\, e^{\pm}\, p_T$'],
           figname='pdfs/eptmat_' + 'b' + '.pdf')

    # Plot hpt -> dca,ept matrices (unweighted and weighted)
    for cb in ['c', 'b']:
        for i, m in enumerate(dcamat):
            print "DCA mmplot()", cb, i
            pt = (ui.dcaeptbins[i], ui.dcaeptbins[i + 1])
            mat = m[:, ui.idx[cb]]
            x = ui.cptx if i == 0 else ui.bptx
            xbins = ui.cptbins if i == 0 else ui.bptbins
            desc1 = r'$h_{:s}\, p_T\, \to\, e$ DCA'.format(cb)
            desc2 = r'$e\, p_T \in\, [{:.1f},{:.1f}]\, GeV/c$'.format(*pt)
            mmplot(mat, x, ui.dcax, xbins, ui.dcabins,
                   xlabel=r'{:s} hadron $p_T$ [GeV/c]'.format(cb),
                   ylabel=r'$e^{\pm}$ DCA [cm]',
                   desc=[desc1, desc2],
                   figname='pdfs/dcamat_{}_{}.pdf'.format(cb, i))

# os.system("pdftk pdfs/*.pdf cat output all.pdf")
