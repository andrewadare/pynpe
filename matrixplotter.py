

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.colors import LogNorm

def mmplot(mat, x, y, xbins, ybins, 
           figsize=(8,8), 
           figname='pdfs/noname.pdf',
           desc='',
           xlabel='columns',
           ylabel='rows'):
    '''
    Matrix marginal plot
    Plot a matrix (the "joint distribution") and its marginal histograms.
    '''
    plt.close('all')

    # Sub-plot frame dimensions as a fraction of figsize
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left+width+0.02

    fig = plt.figure(1, figsize=figsize)
    ax_joint = plt.axes([left, bottom, width, height])
    ax_margx = plt.axes([left, bottom_h, width, 0.2] )
    ax_margy = plt.axes([left_h, bottom, 0.2, height])

    # no labels
    nullfmt = NullFormatter()
    ax_margx.xaxis.set_major_formatter(nullfmt)
    ax_margy.yaxis.set_major_formatter(nullfmt)

    xt = np.tile(xbins[:x.shape[0]/2], 2)
    ax_joint.set_xticks(xbins)
    ax_joint.set_xticklabels(["%.0f" % b for b in xt])

    ax_joint.pcolormesh(xbins, ybins, mat, 
                        norm=LogNorm(vmin=mat.min()+1e-8, vmax=mat.max()), 
                        cmap='Blues')
    ax_joint.set_xlabel(xlabel)
    ax_joint.set_ylabel(ylabel)

    ax_margx.set_yscale('log')
    ax_margy.set_xscale('log')

    ax_margx.hist(x, bins=xbins, histtype='step', color='gray', lw=2, weights=np.sum(mat, axis=0))
    ax_margy.hist(y, bins=ybins, histtype='step', color='gray', lw=2, weights=np.sum(mat, axis=1), orientation='horizontal')
    
    fig.text(0.76, 0.9, desc)

    plt.savefig(figname)
