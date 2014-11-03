import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import unfold_input as ui


def plot_ept(ept_mb, ept_pp, ept_py, figname='ept-comparison.pdf'):
    print("plotept()")

    fig, ax = plt.subplots(figsize=(6, 7))
    ax.set_yscale('log')
    ax.set_xlabel(r'$e^{\pm}$ $p_T$ [GeV/c]')

    # PYTHIA model ept
    ax.errorbar(ui.eptx, ept_py[:, 0] / ui.eptw, yerr=ept_py[:, 1],
                lw=2, ls='*', marker='o', ms=10, alpha=0.8, color='steelblue',
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


def plot_result(ylims, p0, gpt, pq, figname='hpt.pdf'):
    '''
    Draw prior, walkers, initial guess, and result.
    '''
    print("Drawing results...")
    fig, axes = plt.subplots(1, 2)
    for i, ax in enumerate(axes):
        ptx = ui.cptx if i == 0 else ui.bptx
        cb = 'c' if i == 0 else 'b'
        r = ui.idx[cb]
        w = ui.hptw[r]
        ax.set_yscale('log')
        ax.set_xlabel(r'{} hadron $p_T$ [GeV/c]'.format(cb))
        ax.fill_between(ptx, ylims[r, 0] / w, ylims[r, 1] / w,
                        color='slategray', alpha=0.1)
        for i in range(p0.shape[0]):
            ax.plot(ptx, p0[i, r] / w, ls='*', marker='s',
                    ms=14, color='deepskyblue', alpha=0.01)
        ax.plot(ptx, gpt[r] / w, lw=2, ls='*', marker='o', color='white')
        # ax.plot(ptx, gptmod[r] / w, lw=2, ls='*', marker='s', color='black')
        ax.errorbar(ptx, pq[r, 0] / w, yerr=[pq[r, 2] / w, pq[r, 1] / w],
                    ls='*', fmt='o', color='crimson', ecolor='crimson',
                    capthick=2)
    fig.savefig(figname)


def plot_bfrac_samples(samples, quantiles, figname='bfrac_samples.pdf'):
    print("Plotting log probabilities...")
    fig, ax = plt.subplots()
    ax.set_xlabel('Sampled b/(b+c) values')
    ax.set_ylabel('samples')
    ax.hist(samples, 100, color='k', facecolor='RosyBrown',
            alpha=0.5, histtype='stepfilled')
    ax.text(0.05, 0.95, r'{:.2g} +{:.2g} -{:.2g}'.format(*quantiles),
            transform=ax.transAxes)
    fig.savefig(figname)


def plot_post_marg(chain, xlim, figname='posterior.pdf'):
    # Draw posterior marginal distributions
    print("Marginalizing posterior distribution for summary plot...")
    nr, nc = 4, 5
    fig, axes = plt.subplots(nr, nc)
    for row in range(nr):
        for col in range(nc):
            i = nc * row + col
            a = axes[row, col]
            fc = 'yellow' if i < ui.ncpt else 'lime'
            a.hist(chain[:, i], 100, range=(xlim[i, 0], xlim[i, 1]),
                   color='k', facecolor=fc, histtype='stepfilled')
            a.tick_params(axis='x', top='off', labelsize=4)
            a.tick_params(axis='y', left='off', right='off', labelsize=0)
            a.xaxis.get_offset_text().set_size(4)
            a.xaxis.get_major_formatter().set_powerlimits((0, 1))
            s = r'{0:.0f}-{1:.0f} GeV/c'.format(
                ui.hptbins[i % 10], ui.hptbins[i % 10 + 1])
            a.text(0.6, 0.9, s, fontsize=5, transform=a.transAxes)
            if i < ui.ncpt:
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
    ax.text(0.05, 0.95, r'mean, std dev: {:.3g}, {:.3g}'
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


def plotept_fold(ept, cept, bept, cfold, bfold, hfold,
                 figname='ept_fold.pdf'):
    print("plotept_fold()")

    fig, ax = plt.subplots(figsize=(6, 7))
    ax.set_yscale('log')
    ax.set_xlabel(r'$e^{\pm}$ $p_T$ [GeV/c]')
    ax.errorbar(ui.eptx, hfold[:, 0] / ui.eptw,  # yerr= [eptf_lo, eptf_hi],
                lw=2, ls='*', marker='s', ms=10, alpha=0.8, color='crimson',
                label=r'$A_{ept}$*hpt')
    ax.errorbar(ui.eptx, cfold[:, 0] / ui.eptw,  # yerr= [eptf_lo, eptf_hi],
                lw=2, ls='*', marker='s', ms=10, alpha=0.8, color='darkorange',
                label=r'$A_{ept}$*cpt')
    ax.errorbar(ui.eptx, bfold[:, 0] / ui.eptw,  # yerr= [eptf_lo, eptf_hi],
                lw=2, ls='*', marker='s', ms=10, alpha=0.8, color='dodgerblue',
                label=r'$A_{ept}$*bpt')
    # ax.errorbar(ui.eptx, eptmod/ui.eptw, yerr=ept_err,
    #             lw=2, ls='*', marker='o', color='k',
    #             label=r'$A_{ept}$*hptmod')
    ax.errorbar(ui.eptx, ept[:, 0] / ui.eptw, yerr=ept[:, 1] / ui.eptw,
                lw=2, ls='*', marker='o', color='white',
                label=r'$e^{\pm}$ $p_T$ data')

    ax.plot(ui.eptx, cept / ui.eptw, 'o', color='green',
            label=r'PYTHIA $c \to e^{\pm}$')
    ax.plot(ui.eptx, bept / ui.eptw, 'o', color='yellow',
            label=r'PYTHIA $b \to e^{\pm}$')

    ax.legend()
    fig.savefig(figname)
    return


def plotept_refold(ept, cfold, bfold, hfold, figname='ept_refold.pdf'):
    print("plotept_refold()")

    fig, ax = plt.subplots(figsize=(6, 7))
    ax.set_yscale('log')
    ax.set_xlabel(r'$e^{\pm}$ $p_T$ [GeV/c]')
    ax.errorbar(ui.eptx, hfold[:, 0] / ui.eptw,
                yerr=[hfold[:, 2], hfold[:, 1]] / ui.eptw,
                lw=2, ls='*', marker='s', ms=10, alpha=0.8, color='crimson',
                barsabove=True, label=r'$h_{c+b}$ refold')
    ax.errorbar(ui.eptx, cfold[:, 0] / ui.eptw,
                yerr=[cfold[:, 2], cfold[:, 1]] / ui.eptw,
                lw=2, ls='*', marker='s', ms=10, alpha=0.8, color='darkorange',
                barsabove=True, label=r'$h_{c}$ refold')
    ax.errorbar(ui.eptx, bfold[:, 0] / ui.eptw,
                yerr=[bfold[:, 2], bfold[:, 1]] / ui.eptw,
                lw=2, ls='*', marker='s', ms=10, alpha=0.8, color='dodgerblue',
                barsabove=True, label=r'$h_{b}$ refold')
    ax.errorbar(ui.eptx, ept[:, 0] / ui.eptw,
                yerr=[ept[:, 1], ept[:, 1]] / ui.eptw,
                lw=2, ls='*', marker='o', ms=8, color='limegreen',
                barsabove=True, label=r'$e^{\pm}$ $p_T$ data')
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
            a.set_ylim([0.1, 2 * np.max(dca[i])])
            a.tick_params(axis='x', top='off', labelsize=6)
            a.tick_params(axis='y', labelsize=6)
            s = r'{0:.1f}-{1:.1f} GeV/c'.format(
                ui.dcaeptbins[i], ui.dcaeptbins[i + 1])
            a.text(0.55, 0.9, s, fontsize=8, transform=a.transAxes)
            a.step(ui.dcax, hfold[i][:, 0], lw=2, alpha=0.8, color='crimson')
            a.step(ui.dcax, 
                   cfold[i][:, 0], 
                   lw=2, 
                   alpha=0.8, 
                   color='darkorange')
            a.step(ui.dcax, 
                   bfold[i][:, 0], 
                   lw=2, 
                   alpha=0.8, 
                   color='dodgerblue')
            a.hist(ui.dcax,
                   ui.dcabins,
                   weights=dca[i][:, 0],
                   log=True,
                   color='white', #color='darkseagreen',
                   edgecolor='black',
                   alpha=1.0,
                   linewidth=0.8,
                   histtype='stepfilled')
    fig.savefig(figname, bbox_inches='tight')
    return


def plothpt(gpt, hpt, hptd, figname='hpt-gen.pdf'):
    print("plothpt()")
    fig, axes = plt.subplots(1, 2, sharey=True)
    for i, ax in enumerate(axes):
        ptx = ui.cptx if i == 0 else ui.bptx
        cb = 'c' if i == 0 else 'b'
        r = ui.idx[cb]
        w = ui.hptw[r]
        ax.set_yscale('log')
        ax.set_xlabel(r'{} hadron $p_T$ [GeV/c]'.format(cb))
        ax.plot(ptx, gpt[r] / w, 'o-', color='gray',
                label=r'Generated HF hadrons')
        ax.plot(ptx, hpt[r] / w, 'o-', label=r'$h\to e \in$ [1,9] GeV/c')
        for i, d in enumerate(hptd):
            ax.plot(ptx, d[r, 0] / w, 'o-',
                    label=r'$h\to e \in$ [{:.1f},{:.1f}] GeV/c'
                    .format(ui.dcaeptbins[i], ui.dcaeptbins[i + 1]))
    axes[0].legend(prop={'size': 10})
    fig.tight_layout()
    fig.savefig(figname)
    return


def plotbfrac(bfrac_ept, bfrac_dca=None, figname='bfrac.pdf'):
    print("plotbfrac()")
    dcaeptx = ui.dcaeptbins[:-1] + 0.4 * np.diff(ui.dcaeptbins)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_ylim([0., 1.29])
    ax.set_xlabel(r'$e^{\pm}$ $p_T$ [GeV/c]')
    ax.set_ylabel(r'$b \to e / (b \to e + c \to e)$')
    ax.axhline(1.0, linestyle='--', color='black')
    ax.errorbar(ui.eptx, bfrac_ept[:, 0],
                yerr=[bfrac_ept[:, 2], bfrac_ept[:, 1]],
                lw=2, ls='*', marker='s', ms=10, alpha=0.8, color='crimson',
                label=r'$h_{b}$ refold / $h_{c+b}$ refold')

    if bfrac_dca is not None:
        ax.errorbar(dcaeptx, bfrac_dca[:, 0],
                    yerr=[bfrac_dca[:, 2], bfrac_dca[:, 1]],
                    lw=2, ls='*', marker='s', ms=10, alpha=0.8, color='blue',
                    label=r'DCA unfold')
    ax.legend(loc=2)
    fig.savefig(figname)
    return

# def plotbfrac(bfrac_ept, bfrac_dca, figname='bfrac.pdf'):
#     print("plotbfrac()")
#     dcaeptx = ui.dcaeptbins[:-1] + 0.4 * np.diff(ui.dcaeptbins)
#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.set_ylim([0., 1.])
#     ax.set_xlabel(r'$e^{\pm}$ $p_T$ [GeV/c]')
#     ax.set_ylabel(r'$b \to e / (b \to e + c \to e)$')
#     ax.errorbar(ui.eptx, bfrac_ept[:,0], yerr= [bfrac_ept[:,2],bfrac_ept[:,1]],
#                 lw=2, ls='*', marker='s', ms=10, alpha=0.8, color='crimson',
#                 label=r'$A_{ept}$*bpt / $A_{ept}$*hpt')
#     ax.errorbar(dcaeptx, bfrac_dca[:,0], yerr= [bfrac_dca[:,2],bfrac_dca[:,1]],
#                 lw=2, ls='*', marker='s', ms=10, alpha=0.8, color='blue',
#                 label=r'$A_{dca}$*bpt / $A_{dca}$*hpt')
#     ax.legend(loc=2)
#     fig.savefig(figname)
#     return
