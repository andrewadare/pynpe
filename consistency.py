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
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import raamodel
from matrixplotter import mmplot
import unfold_input as ui


def plotept_fold(ept, ept_err, cept, bept, cfold, bfold, hfold):
    print("plotept_dist()")

    fig, ax = plt.subplots(figsize=(6, 7))
    ax.set_yscale('log')
    ax.set_xlabel(r'$e^{\pm}$ $p_T$ [GeV/c]')
    ax.errorbar(ui.eptx, hfold / ui.eptw,  # yerr= [eptf_lo, eptf_hi],
                lw=2, ls='*', marker='s', ms=10, alpha=0.8, color='crimson',
                label=r'$A_{ept}$*hpt')
    ax.errorbar(ui.eptx, cfold / ui.eptw,  # yerr= [eptf_lo, eptf_hi],
                lw=2, ls='*', marker='s', ms=10, alpha=0.8, color='darkorange',
                label=r'$A_{ept}$*cpt')
    ax.errorbar(ui.eptx, bfold / ui.eptw,  # yerr= [eptf_lo, eptf_hi],
                lw=2, ls='*', marker='s', ms=10, alpha=0.8, color='dodgerblue',
                label=r'$A_{ept}$*bpt')
    # ax.errorbar(ui.eptx, eptmod/ui.eptw, yerr=ept_err,
    #             lw=2, ls='*', marker='o', color='k',
    #             label=r'$A_{ept}$*hptmod')
    ax.errorbar(ui.eptx, ept / ui.eptw, yerr=ept_err,
                lw=2, ls='*', marker='o', color='white',
                label=r'$e^{\pm}$ $p_T$ data')

    ax.plot(ui.eptx, cept / ui.eptw, 'o', color='green',
            label=r'PYTHIA $c \to e^{\pm}$')
    ax.plot(ui.eptx, bept / ui.eptw, 'o', color='yellow',
            label=r'PYTHIA $b \to e^{\pm}$')

    ax.legend()
    fig.savefig('pdfs/ept-fold.pdf')
    return


def plotdca_fold(dca, cfold, bfold, hfold):
    print("plotdca_dists()")
    nr, nc = 2, 3
    fig, axes = plt.subplots(nr, nc)
    for row in range(nr):
        for col in range(nc):
            i = nc * row + col
            a = axes[row, col]
            a.set_yscale('log')
            a.set_ylim([1, 2 * np.max(dca[i])])
            a.tick_params(axis='x', top='off', labelsize=6)
            a.tick_params(axis='y', labelsize=6)
            s = r'{0:.1f}-{1:.1f} GeV/c'.format(
                ui.dcaeptbins[i], ui.dcaeptbins[i + 1])
            a.text(0.55, 0.9, s, fontsize=8, transform=a.transAxes)
            a.step(ui.dcax, hfold[i], lw=1, alpha=0.8, color='crimson')
            a.step(ui.dcax, cfold[i], lw=1, alpha=0.8, color='darkorange')
            a.step(ui.dcax, bfold[i], lw=1, alpha=0.8, color='dodgerblue')
            if False:
                a.step(ui.dcax, bkg[i], color='brown')
            a.step(ui.dcax, dca[i], color='black', alpha=0.6)
            if False:
                a.step(ui.dcax, dcamod[i], color='red', alpha=0.6)
    fig.savefig('pdfs/dca_fold.pdf', bbox_inches='tight')
    return


def plothpt(gpt, hpt, hptd):
    print("plothpt()")
    fig, axes = plt.subplots(1, 2, sharey=True)
    ptx = ui.hptx[:ui.ndim / 2]
    for ax in axes:
        cb = 'c' if ax == axes[0] else 'b'
        r = range(
            ui.ndim / 2) if ax == axes[0] else range(ui.ndim / 2, ui.ndim)
        w = ui.hptw[r]
        ax.set_yscale('log')
        ax.set_xlabel(r'{} hadron $p_T$ [GeV/c]'.format(cb))
        ax.plot(ptx, gpt[r] / w, 'o-', color='gray',
                label=r'Generated HF hadrons')
        ax.plot(ptx, hpt[r] / w, 'o-', label=r'$h\to e \in$ [1,9] GeV/c')
        for i, d in enumerate(hptd):
            ax.plot(ptx, d[r] / w, 'o-',
                    label=r'$h\to e \in$ [{:.1f},{:.1f}] GeV/c'
                    .format(ui.dcaeptbins[i], ui.dcaeptbins[i + 1]))
    axes[0].legend(prop={'size': 10})
    fig.tight_layout()
    fig.savefig('pdfs/hpt-gen.pdf')
    return


def plotbfrac(bfrac_ept, bfrac_dca):
    print("plotbfrac()")
    dcaeptx = ui.dcaeptbins[:-1] + np.diff(ui.dcaeptbins) / 2
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_ylim([0., 1.])
    ax.set_xlabel(r'$e^{\pm}$ $p_T$ [GeV/c]')
    ax.set_ylabel(r'$b \to e / (b \to e + c \to e)$')
    ax.errorbar(ui.eptx, bfrac_ept,  # yerr= [eptf_lo, eptf_hi],
                lw=2, ls='*', marker='s', ms=10, alpha=0.8, color='crimson',
                label=r'$A_{ept}$*bpt / $A_{ept}$*hpt')
    ax.errorbar(dcaeptx, bfrac_dca,
                lw=2, ls='*', marker='s', ms=10, alpha=0.8, color='blue',
                label=r'$A_{dca}$*bpt / $A_{dca}$*hpt')
    ax.legend(loc=2)
    fig.savefig('pdfs/bfrac-fold.pdf')
    return

if __name__ == '__main__':

    # mmplot(eptMat0, hptx, eptx, hptbins, eptbins,
    #        xlabel=r'c hadron $p_T$ [GeV/c] $\qquad$ b hadron $p_T$ [GeV/c]',
    #        ylabel=r'$e^{\pm}$ $p_T$ [GeV/c]',
    #        desc=r'$h\to e\, \in\, [1,9]\, GeV/c$',
    #        figname='pdfs/eptmat0.pdf')

    # for i, m in enumerate(dcaMat0):
    #     mmplot(m, hptx, dcax, hptbins, dcabins,
    #            xlabel=r'c hadron $p_T$ [GeV/c] $\qquad$ b hadron $p_T$ [GeV/c]',
    #            ylabel=r'$e^{\pm}$ DCA [cm]',
    #            desc=r'$h\to e\, \in\, [{:.1f},{:.1f}]\, GeV/c$'\
    #            .format(dcaeptbins[i], dcaeptbins[i + 1]),
    #            figname='pdfs/dcamat0{}.pdf'.format(i))

    # mmplot(eptMat, hptx, eptx, hptbins, eptbins,
    #        xlabel=r'c hadron $p_T$ [GeV/c] $\qquad$ b hadron $p_T$ [GeV/c]',
    #        ylabel=r'$e^{\pm}$ $p_T$ [GeV/c]',
    #        desc=r'$h\to e\, \in\, [1,9]\, GeV/c$',
    #        figname='pdfs/eptmat.pdf')

    # for i, m in enumerate(dcaMat):
    #     mmplot(m, hptx, dcax, hptbins, dcabins,
    #            xlabel=r'c hadron $p_T$ [GeV/c] $\qquad$ b hadron $p_T$ [GeV/c]',
    #            ylabel=r'$e^{\pm}$ DCA [cm]',
    #            desc=r'$h\to e\, \in\, [{:.1f},{:.1f}]\, GeV/c$'\
    #            .format(dcaeptbins[i], dcaeptbins[i + 1]),
    #            figname='pdfs/dcamat{}.pdf'.format(i))

    bfrac = 0.007
    # Unweighted matrices
    eptMat = ui.eptmatrix(bfrac, False)
    dcaMat = [ui.dcamatrix(bfrac, i, False) for i in range(6)]
    # Weighted matrices
    eptMatW = ui.eptmatrix(bfrac, True)
    dcaMatW = [ui.dcamatrix(bfrac, i, True) for i in range(6)]

    gpt = ui.genpt(bfrac)
    hpt_ideal = eptMat.sum(axis=0)
    ept_ideal = eptMat.sum(axis=1)
    ept_err = np.sqrt(ept_ideal)  # Whatever
    dca_ideal = [m.sum(axis=1) for m in dcaMat]

    if True:
        z = np.zeros(ui.ndim / 2)

        # B fraction from electron pt spectra
        bfold_ept = np.dot(eptMatW, np.hstack([z, gpt[ui.ndim / 2:]]))
        hfold_ept = np.dot(eptMatW, gpt)
        bfrac_ept = bfold_ept / hfold_ept

        # B fraction from DCA distributions at different pt values
        hfold_dca = [np.dot(m, gpt) for m in dcaMatW]
        cfold_dca = [np.dot(m, np.hstack([gpt[:ui.ndim / 2], z])) for m in dcaMatW]
        bfold_dca = [np.dot(m, np.hstack([z, gpt[ui.ndim / 2:]])) for m in dcaMatW]
        bfrac_dca = np.array([np.sum(b) / np.sum(h)
                              for b, h in zip(bfold_dca, hfold_dca)])
        plotbfrac(bfrac_ept, bfrac_dca)

    if True:
        z = np.zeros(ui.ndim / 2)
        hfold = [np.dot(m, gpt) for m in dcaMatW]
        cfold = [np.dot(m, np.hstack([gpt[:ui.ndim / 2], z])) for m in dcaMatW]
        bfold = [np.dot(m, np.hstack([z, gpt[ui.ndim / 2:]])) for m in dcaMatW]
        plotdca_fold(dca_ideal, cfold, bfold, hfold)

    if True:
        hptd = [m.sum(axis=0) for m in dcaMat]
        plothpt(gpt, hpt_ideal, hptd)

    if True:
        z = np.zeros(ui.ndim / 2)
        cept = eptMat[:, :ui.ndim / 2].sum(axis=1)
        bept = eptMat[:, ui.ndim / 2:].sum(axis=1)
        cfold = np.dot(eptMatW, np.hstack([gpt[:ui.ndim / 2], z]))
        bfold = np.dot(eptMatW, np.hstack([z, gpt[ui.ndim / 2:]]))
        hfold = np.dot(eptMatW, gpt)
        plotept_fold(ept_ideal, ept_err, cept, bept, cfold, bfold, hfold)

    # os.system("pdftk pdfs/*.pdf cat output all.pdf")
