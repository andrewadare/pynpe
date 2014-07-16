'''
consistency.py: 
1. Plot model and data inputs
2. Check for consistency between models for two different datasets
3. Check for consistency between data and model.
If large inconsistencies are seen, unfolding is unlikely to be satisfactory.
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
import npe_io as io
import raamodel
from matplotlib.colors import LogNorm

dtype     = 'MC'
wt        = ''       # or '-weighted'
dcares    = 0.007    # 0.007 cm in MB Au+Au, 0.014 cm in p+p.
bfrac     = 0.03
dcabins   = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)

ept, ept_err, dca, bkg = io.getdata(dtype)

eptMat, dcaMat, hptx, eptx, dcax, hptbins, eptbins = \
io.getmodel(dcares, bfrac, wt)

ndim    = eptMat.shape[1]
hpt     = eptMat.sum(axis=0)
hptmod  = hpt*raamodel.getraa(hpt,10)

def plotdca():
  print("plotdca() matrices")
  nr,nc = 2,3
  fig, axes = plt.subplots(nr,nc)
  for row in range(nr):
    for col in range(nc):
      i = nc*row + col
      a = axes[row,col]
      a.tick_params(axis='x', top='off', labelsize=4)
      a.tick_params(axis='y', left='off', right='off', labelsize=4)
      s = r'{0:.1f}-{1:.1f} GeV/c'.format(dcabins[i], dcabins[i+1])
      a.text(0.6, 0.9, s, fontsize=5, transform=a.transAxes)
      m = dcaMat[i]
      p = a.pcolormesh(m, norm=LogNorm(vmin=m.min()+1e-8, vmax=m.max()),
                       cmap='Spectral_r')
  # fig.colorbar(p)
  fig.savefig('pdfs/dca_matrices.pdf', bbox_inches='tight')

  print("plotdca() distributions")
  fig, axes = plt.subplots(nr,nc)
  for row in range(nr):
    for col in range(nc):
      i = nc*row + col
      a = axes[row,col]
      a.set_yscale('log')
      a.tick_params(axis='x', top='off', labelsize=6)
      a.tick_params(axis='y', labelsize=6)
      s = r'{0:.1f}-{1:.1f} GeV/c'.format(dcabins[i], dcabins[i+1])
      a.text(0.55, 0.9, s, fontsize=8, transform=a.transAxes)
      if False:
        a.step(dcax, bkg[i], color='brown')
      a.step(dcax, dca[i], color='black', alpha = 0.6)
      if False:
        a.step(dcax, dcamod[i], color='red', alpha = 0.6)
  # a.errorbar(eptx, eptrefold, yerr= [eptref_lo, eptref_hi],
  #         lw=2, ls='*', marker='s', ms=10, color='r')
  # a.errorbar(eptx, eptmod, yerr=ept_err, lw=2, ls='*', marker='o', color='k')
  # fig.colorbar(p)
  fig.savefig('pdfs/dca_dists.pdf', bbox_inches='tight')
  return

# Draw matrix
def plotmatrix_ept(aept):
  print("plotmatrix_ept()")
  fig, ax = plt.subplots()
  p = ax.pcolormesh(aept, norm=LogNorm(vmin=aept.min()+1e-8, vmax=aept.max()), cmap='Spectral_r')
  ax.set_xlabel(r'h $p_T$ bin index')
  ax.set_ylabel(r'$e^{\pm}$ $p_T$ bin index')
  ax.set_ylim([0, aept.shape[1]+1])
  fig.colorbar(p)
  fig.savefig('pdfs/aept.pdf')
  return

def plothpt():
  print("plothpt()")
  fig, axes = plt.subplots(1,2)
  ptx = hptx[:ndim/2]
  for ax in axes:
    cb = 'charm' if ax == axes[0] else 'beauty'
    r  = range(ndim/2) if ax == axes[0] else range(ndim/2,ndim)
    ax.set_yscale('log')
    ax.set_xlabel(r'{} hadron $p_T$ [GeV/c]'.format(cb))
    # # ymin and ymax (part of prior)
    # ax.fill_between(ptx, ymin[r], ymax[r], color='slategray', alpha=0.1)
    # # walkers
    # for i in range(nwalkers): 
    #   ax.plot(ptx, p0[i,r], ls='*', marker='s', ms=14, color='deepskyblue', alpha=0.01)
    # gen
    ax.plot(ptx, hpt[r], lw=2, ls='*', marker='o', color='white')
    # mod
    ax.plot(ptx, hptmod[r], lw=2, ls='*', marker='s', color='black')
    # # result
    # ax.errorbar(ptx, pq[r,0], yerr=[pq[r,2], pq[r,1]],
    #           ls='*', fmt='o', color='crimson', ecolor='crimson', capthick=2)
  
  fig.savefig('pdfs/hpt-check.pdf')

if __name__=='__main__':
  # plotmatrix_ept(eptMat)
  # plotdca()
  plothpt()

