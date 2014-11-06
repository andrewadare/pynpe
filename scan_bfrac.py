import numpy as np
import matplotlib.pyplot as plt
import unfold_input as ui
import plotting_functions as pf
import lnpmodels as lnp

# Don't forget to run project_and_save() if csv files are out of date!!!
dtype = 'pp200' #'pp200' 'AuAu200MB'
dcares = {'AuAu200MB' : 0.007, 'pp200' : 0.01, 'MC' : 0.0}

# Create text files for matrix creation (project_and_save() is idempotent).
ui.project_and_save(dcares[dtype])

c, b = ui.idx['c'], ui.idx['b']
dcamat = [ui.dcamatrix(i, weighted=False) for i in range(6)]
dca = [ui.dcadata(i, dtype) for i in range(6)]

fsteps = np.linspace(0, 1, 101)
ll = np.zeros([len(fsteps),6])

for i in range(6):
    data = dca[i]
    cdca, bdca = dcamat[i][:, c].sum(axis=1), dcamat[i][:, b].sum(axis=1)
    # print np.average(ui.dcax,weights=cdca)
    # print np.average(ui.dcax,weights=bdca)

    for j,f in enumerate(fsteps):
        # Create prediction, initially normalized to 1, then matched to data
        pred = cdca * (1 - f) / cdca.sum() + f * bdca / bdca.sum()
        scf = (np.sum(data[:, 0]) - np.sum(data[:, 1])) / np.sum(pred)
        pred = pred * scf + data[:, 1]

        if np.any(data[:,0] < 0.): print data
        if np.any(pred <= 0.): print pred

        # print lnp.lnpoisson(data[:,0], pred)
        ll[j,i] = lnp.lnpoisson(data[:,0], pred)

#         cd = cdca*scf * (1 - f) / cdca.sum()
#         bd = bdca*scf * f / bdca.sum()
#         fig, ax = plt.subplots()
#         ax.set_yscale('log')
#         ax.set_xlim([ui.dcabins[0], ui.dcabins[-1]])
#         ax.set_ylim([0.1, 2 * np.max(dca[i])])
#         ax.hist(ui.dcax,
#                 ui.dcabins,
#                 weights=dca[i][:, 0],
#                 log=True,
#                 color='white',
#                 edgecolor='black',
#                 alpha=1.0,
#                 linewidth=0.8,
#                 histtype='stepfilled')

#         ax.plot(ui.dcax, dca[i][:, 1], lw=3, color='blue', label='bkg')
#         ax.plot(ui.dcax, cd, lw=3, color='darkorange', label='charm')
#         ax.plot(ui.dcax, bd, lw=3, color='dodgerblue', label='beauty')
#         ax.plot(ui.dcax, pred, lw=3, color='crimson', 
#                 label='pred f={:.2f}'.format(f))
#         ax.legend()

# plt.show()

def plotll(fsteps, ll, figname='ll-{}.pdf'.format(dtype)):
    print("plotdca_dists()")
    nr, nc = 2, 3
    fig, axes = plt.subplots(nr, nc)
    for row in range(nr):
        for col in range(nc):
            i = nc * row + col
            a = axes[row, col]
            a.tick_params(axis='x', top='off', labelsize=6)
            a.tick_params(axis='y', labelsize=6)
            s = r'{0:.1f}-{1:.1f} GeV/c'.format(
                 ui.dcaeptbins[i], ui.dcaeptbins[i + 1])
            a.text(0.55, 0.9, s, fontsize=8, transform=a.transAxes)
            # a.step(ui.dcax, hfold[i][:, 0], lw=2, alpha=0.8, color='crimson')
            a.plot(fsteps, ll[:, i],
                   lw=2, 
                   alpha=0.8, 
                   color='black')
    fig.savefig(figname, bbox_inches='tight')
    return

plotll(fsteps, ll)