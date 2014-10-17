import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# import emcee
# import triangle
# import lnpmodels
import unfold_input as ui


def plot_ept(ept_mb, ept_pp, ept_py, figname='ept-comparison.pdf'):
    print("plotept()")

    fig, ax = plt.subplots(figsize=(6, 7))
    ax.set_yscale('log')
    ax.set_xlabel(r'$e^{\pm}$ $p_T$ [GeV/c]')

    # PYTHIA model ept
    ax.errorbar(ui.eptx, ept_py / ui.eptw, yerr=np.sqrt(ept_py),
                lw=2, ls='*', marker='o', ms=10, color='white',
                label=r'PYTHIA HF $e^{\pm}$ $p_T$')

    # PHENIX data
    ax.errorbar(ui.eptx, ept_pp[:, 0] / ui.eptw,  yerr=ept_pp[:, 1],
                lw=2, ls='*', marker='s', ms=8, alpha=0.8, color='green',
                label=r'Run 4 p+p $e^{\pm}$ $p_T$')
    ax.errorbar(ui.eptx, ept_mb[:, 0] / ui.eptw,  yerr=ept_mb[:, 1],
                lw=2, ls='*', marker='s', ms=8, alpha=0.8, color='crimson',
                label=r'Run 4 Au+Au $e^{\pm}$ $p_T$ (/10)')
    ax.legend()
    fig.savefig(figname)
    return

# Draw prior, walkers, initial guess, and result.
def plot_result(ymin, ymax, p0, gpt, pq, figname='hpt.pdf'):
    print("Drawing results...")
    fig, axes = plt.subplots(1, 2)
    ptx = ui.hptx[:ui.ndim / 2]
    for ax in axes:
        cb = 'c' if ax == axes[0] else 'b'
        r = range(
            ui.ndim / 2) if ax == axes[0] else range(ui.ndim / 2, ui.ndim)
        w = ui.hptw[r]
        ax.set_yscale('log')
        ax.set_xlabel(r'{} hadron $p_T$ [GeV/c]'.format(cb))
        ax.fill_between(
            ptx, ymin[r] / w, ymax[r] / w, color='slategray', alpha=0.1)
        for i in range(p0.shape[0]):
            ax.plot(ptx, p0[i, r] / w, ls='*', marker='s',
                    ms=14, color='deepskyblue', alpha=0.01)
        ax.plot(ptx, gpt[r] / w, lw=2, ls='*', marker='o', color='white')
        # ax.plot(ptx, gptmod[r] / w, lw=2, ls='*', marker='s', color='black')
        ax.errorbar(ptx, pq[r, 0] / w, yerr=[pq[r, 2] / w, pq[r, 1] / w],
                    ls='*', fmt='o', color='crimson', ecolor='crimson',
                    capthick=2)
    fig.savefig(figname)


def plot_post_marg(chain, figname='posterior.pdf'):
    # Draw posterior marginal distributions
    # sample.chain has shape (nwalkers, nsteps, ndim)
    print("Marginalizing posterior distribution for summary plot...")
    nr, nc = 4, 5
    fig, axes = plt.subplots(nr, nc)
    for row in range(nr):
        for col in range(nc):
            i = nc * row + col
            a = axes[row, col]
            fc = 'yellow' if i < ui.ndim / 2 else 'lime'
            a.hist(chain[:, i], 100,
                   color='k', facecolor=fc, histtype='stepfilled')
            a.tick_params(axis='x', top='off', labelsize=4)
            a.tick_params(axis='y', left='off', right='off', labelsize=0)
            a.xaxis.get_offset_text().set_size(4)
            a.xaxis.get_major_formatter().set_powerlimits((0, 1))
            s = r'{0:.0f}-{1:.0f} GeV/c'.format(
                ui.hptbins[i % 10], ui.hptbins[i % 10 + 1])
            a.text(0.6, 0.9, s, fontsize=5, transform=a.transAxes)
            if i < ui.ndim / 2:
                a.text(0.05, 0.9, r'$h_{charm}$',
                       fontsize=5, transform=a.transAxes)
            else:
                a.text(0.05, 0.9, r'$h_{beauty}$',
                       fontsize=5, transform=a.transAxes)
    fig.savefig(figname, bbox_inches='tight')


def plot_lnprob(lnp, figname='lnprob.pdf'):
    print("Plotting log probabilities...")
    fig, ax = plt.subplots()
    ax.set_xlabel('log likelihood distribution')
    ax.set_ylabel('samples')
    ax.hist(
        lnp, 100, color='k', facecolor='lightyellow', histtype='stepfilled')
    ax.text(0.05, 0.95, r'mean, std dev: {0:.2f}, {1:.2f}'
            .format(np.mean(lnp), np.std(lnp)), transform=ax.transAxes)
    fig.savefig(figname)


def plot_lnp_steps(sampler, nburnin, figname='lnprob_vs_step.pdf'):
    print("Plotting log probabilities vs step...")
    nwalkers = sampler.chain.shape[0]
    fig, ax = plt.subplots()
    ax.set_xlabel('step')
    ax.set_ylabel(r'$\langle \ln(L) \rangle_{chains}$')
    ax.set_title(r'$\langle \ln(L) \rangle_{chains}$ vs. sample step')
    ax.plot(np.sum(sampler.lnprobability / nwalkers, axis=0), color='k')
    ax.text(0.05, 0.95,
            '{} chains after {} burn-in steps'.format(nwalkers, nburnin),
            transform=ax.transAxes)
    fig.savefig(figname)


def plotept_fold(ept, ept_err, cept, bept, cfold, bfold, hfold, 
                 figname='ept_fold.pdf'):
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
    fig.savefig(figname)
    return


def plotdca_fold(dca, cfold, bfold, hfold, figname='dca_fold.pdf'):
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
    fig.savefig(figname, bbox_inches='tight')
    return


def plothpt(gpt, hpt, hptd, figname='hpt-gen.pdf'):
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
    fig.savefig(figname)
    return


def plotbfrac(bfrac_ept, bfrac_dca, figname='bfrac-fold.pdf'):
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
    fig.savefig(figname)
    return
