

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.colors import LogNorm

def mmplot(mat, x, y, xbins, ybins, 
           figsize=(8,8), 
           figname='pdfs/noname.pdf',
           desc=[''],
           xlabel='columns',
           ylabel='rows'):
    '''
    Matrix marginal plot
    Plot a matrix (the "joint distribution") and its marginal histograms.
    '''
    plt.close('all')

    # Sub-plot frame dimensions as a fraction of figsize
    left, width = 0.1, 0.64
    bottom, height = 0.1, 0.64
    bottom_h = left_h = left+width+0.04

    fig = plt.figure(1, figsize=figsize)
    ax_joint = plt.axes([left, bottom, width, height])
    ax_margx = plt.axes([left, bottom_h, width, 0.2] ) # Top panel
    ax_margy = plt.axes([left_h, bottom, 0.2, height]) # RHS panel

    # No labels on "bottom" axes of marginal panels
    nullfmt = NullFormatter()
    ax_margx.xaxis.set_major_formatter(nullfmt)
    ax_margy.yaxis.set_major_formatter(nullfmt)

    ax_joint.set_xticks(xbins)
    # xt = np.tile(xbins[:x.shape[0]/2], 2)
    # ax_joint.set_xticklabels(["%.0f" % b for b in xt])

    ax_joint.pcolormesh(xbins, ybins, mat, 
                        norm=LogNorm(vmin=mat.min()+1e-8, vmax=mat.max()), 
                        cmap='Blues')
    ax_joint.set_xlabel(xlabel)
    ax_joint.set_ylabel(ylabel)

     # Deal with buggy MPL limit handling
    ax_joint.set_xlim(xbins[0],xbins[-1])
    ax_joint.set_ylim(ybins[0],ybins[-1])

    # Marginal distributions
    mx = np.sum(mat, axis=0)
    my = np.sum(mat, axis=1)
    ax_margx.hist(x, bins=xbins, histtype='step', color='gray', lw=2, 
                  weights=mx)
    ax_margy.hist(y, bins=ybins, histtype='step', color='gray', lw=2, 
                  weights=my, orientation='horizontal')
    ax_margx.set_yscale('log')
    ax_margy.set_xscale('log')

    # Set abcissa limits
    ax_margx.set_xlim(xbins[0],xbins[-1])
    ax_margy.set_ylim(ybins[0],ybins[-1])

    # Set ordinate limits
    mxmin, mxmax, dx = np.min(mx), np.max(mx), 0.5*np.ptp(mx)
    mymin, mymax, dy = np.min(my), np.max(my), 0.5*np.ptp(my)
    ax_margx.set_ylim(0.99*mxmin, mxmax + dx)
    ax_margy.set_xlim(0.99*mymin, mymax + dy)
    
    # Add description in upper right space. Each list item is a new line.
    for i,d in enumerate(desc):
        fig.text(0.76, 0.95 - i*0.05, d)

    plt.savefig(figname)
