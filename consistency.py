'''
consistency.py: 
1. Plot model and data inputs
2. Check for consistency between models for two different datasets
3. Check for consistency between data and model.
If large inconsistencies are seen, unfolding is unlikely to work.
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
import npe_io as io
import raamodel
from matplotlib.colors import LogNorm

dtype     = 'MC'
dcares    = 0.007    # 0.007 cm in MB Au+Au, 0.014 cm in p+p.
bfrac     = 0.007

io.mf = io.modelfile(dcares, bfrac)

eptMat                    = io.eptmatrix()
dcaMat                    = io.dcamatrices()
dcax, dcaeptx, dcaeptbins = io.dcabins()
eptx, eptbins             = io.eptbins()
ept, ept_err              = io.eptdata(dtype)
dca, bkg                  = io.dcadata(dtype)
hpte, hptd, hptx, hptbins = io.hadronpt()
gpt, gptx                 = io.genpt()
dcaweights                = io.dcaweights(bfrac)

# acc_gen_ratio = np.sum(hpt)/np.sum(gpt)

ndim   = eptMat.shape[1]
dca    = [m.sum(axis=1) for m in dcaMat] # !!! Reassignment !!!
ept    = eptMat.sum(axis=1)              # !!! Reassignment !!!
cept   = eptMat[:,:ndim/2].sum(axis=1)
bept   = eptMat[:,ndim/2:].sum(axis=1)
hptmod = hpte*raamodel.getraa(hptx,ndim/2)
hptw   = io.binwidths(hptbins)
eptw   = io.binwidths(eptbins)

z      = np.zeros(ndim/2)
cpte   = np.concatenate([hpte[:ndim/2], z])
bpte   = np.concatenate([z, hpte[ndim/2:]])
cptd   = [np.concatenate([a[:ndim/2], z]) for a in hptd]
bptd   = [np.concatenate([z, a[ndim/2:]]) for a in hptd]

#############################################################################
# Normalize matrices
# correction = 1.0
scale = False
if scale:
  # correction = np.sum(np.dot(eptMat/eptMat[].sum(axis=0), hpt))/\
  # np.sum(np.dot(eptMat/gpt, hpt))
  # eptMat /= gpt
  eptMat /= eptMat.sum(axis=0)
  for (i,m) in enumerate(dcaMat):
    colsum = np.maximum(np.ones_like(m), gpt)
    dcaMat[i] = m/colsum
else:
  eptMat /= eptMat.sum(axis=0)
  for (i,m) in enumerate(dcaMat):
    colsum = np.maximum(np.ones_like(m), m.sum(axis=0))
    dcaMat[i] = m/colsum #* dcaweights[i]
#############################################################################

# Electron pt spectrum from decaying RAA-modified HF hadron spectra
eptmod  = np.dot(eptMat,hptmod)

# "Folded" distributions
hfold_ept = np.dot(eptMat,hpte)
cfold_ept = np.dot(eptMat,cpte)
bfold_ept = np.dot(eptMat,bpte)

# rpt = hpt/gpt;
# print "acc_gen_ratio (ROS):", acc_gen_ratio
# print "SOR (arithmetic mean):", np.sum(rpt)/ndim
# print "SOR (geometric mean):", np.power(np.prod(rpt), 1./ndim)
# print "1/correction:", 1./correction 
# print "refold/data:", np.sum(hfold_ept) / np.sum(ept)
# sys.exit()
# for item in [hfold_ept, cfold_ept, bfold_ept]:
#   item *= correction

hfold_dca = [np.dot(m,v) for m,v in zip(dcaMat,hptd)]
cfold_dca = [np.dot(m,v) for m,v in zip(dcaMat,cptd)]
bfold_dca = [np.dot(m,v) for m,v in zip(dcaMat,bptd)]

# scf = [np.sum(x)/np.sum(y) for x,y in zip(hfold_dca, dca)]
# print scf
# hfold_dca = [hfold_dca[i]/scf[i] for i in range(len(hfold_dca))]
# cfold_dca = [cfold_dca[i]/scf[i] for i in range(len(cfold_dca))]
# bfold_dca = [bfold_dca[i]/scf[i] for i in range(len(bfold_dca))]

# for l in [hfold_dca, cfold_dca, bfold_dca]:
#   l = [x*y for x,y in zip(l,scf)]

bfrac_dca = np.array([np.sum(b)/np.sum(h) for b,h in zip(bfold_dca, hfold_dca)])

def plotdca_matrices():
  print("plotdca_matrices()")
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
  return

def plotdca_dists():
  print("plotdca_dists()")
  nr,nc = 2,3
  fig, axes = plt.subplots(nr,nc)
  for row in range(nr):
    for col in range(nc):
      i = nc*row + col
      a = axes[row,col]
      a.set_yscale('log')
      a.set_ylim([1, 1.2*np.max(hptd[i])])
      a.tick_params(axis='x', top='off', labelsize=6)
      a.tick_params(axis='y', labelsize=6)
      s = r'{0:.1f}-{1:.1f} GeV/c'.format(dcaeptbins[i], dcaeptbins[i+1])
      a.text(0.55, 0.9, s, fontsize=8, transform=a.transAxes)
      a.step(dcax, hfold_dca[i], lw=1, alpha = 0.8, color='crimson')
      a.step(dcax, cfold_dca[i], lw=1, alpha = 0.8, color='darkorange')
      a.step(dcax, bfold_dca[i], lw=1, alpha = 0.8, color='dodgerblue')
      if False:
        a.step(dcax, bkg[i], color='brown')
      a.step(dcax, dca[i], color='black', alpha = 0.6)
      if False:
        a.step(dcax, dcamod[i], color='red', alpha = 0.6)
  fig.savefig('pdfs/dca_dists.pdf', bbox_inches='tight')
  return

# Draw matrix
def plotept_matrix(aept):
  print("plotept_matrix()")
  fig, ax = plt.subplots()
  p = ax.pcolormesh(aept, norm=LogNorm(vmin=aept.min()+1e-8, vmax=aept.max()), 
                    cmap='Spectral_r')
  ax.set_xlabel(r'h $p_T$ bin index')
  ax.set_ylabel(r'$e^{\pm}$ $p_T$ bin index')
  ax.set_ylim([0, aept.shape[1]+1])
  fig.colorbar(p)
  fig.savefig('pdfs/aept.pdf')
  return

def plotept_dist():  
  print("plotept_dist()")
  fig, ax = plt.subplots(figsize=(6,7))
  ax.set_yscale('log')
  ax.set_xlabel(r'$e^{\pm}$ $p_T$ [GeV/c]')
  ax.errorbar(eptx, hfold_ept/eptw, #yerr= [eptf_lo, eptf_hi],
              lw=2, ls='*', marker='s', ms=10, alpha=0.8, color='crimson',
              label=r'$A_{ept}$*hpt')
  ax.errorbar(eptx, cfold_ept/eptw, #yerr= [eptf_lo, eptf_hi],
              lw=2, ls='*', marker='s', ms=10, alpha=0.8, color='darkorange',
              label=r'$A_{ept}$*cpt')
  ax.errorbar(eptx, bfold_ept/eptw, #yerr= [eptf_lo, eptf_hi],
              lw=2, ls='*', marker='s', ms=10, alpha=0.8, color='dodgerblue',
              label=r'$A_{ept}$*bpt')
  ax.errorbar(eptx, eptmod/eptw, yerr=ept_err, 
              lw=2, ls='*', marker='o', color='k',
              label=r'$A_{ept}$*hptmod')
  ax.errorbar(eptx, ept/eptw, yerr=ept_err, 
              lw=2, ls='*', marker='o', color='white',
              label=r'$e^{\pm}$ $p_T$ data')

  ax.plot(eptx, cept/eptw, 'o', color='green', label=r'PYTHIA $c \to e^{\pm}$')
  ax.plot(eptx, bept/eptw, 'o', color='yellow', label=r'PYTHIA $b \to e^{\pm}$')

  ax.legend()
  fig.savefig('pdfs/ept-check.pdf')

def plothpt():
  print("plothpt()")
  fig, axes = plt.subplots(1,2,sharey=True)
  ptx = hptx[:ndim/2]
  for ax in axes:
    cb = 'c' if ax == axes[0] else 'b'
    r  = range(ndim/2) if ax == axes[0] else range(ndim/2,ndim)
    w  = hptw[r]
    ax.set_yscale('log')
    ax.set_xlabel(r'{} hadron $p_T$ [GeV/c]'.format(cb))
    ax.plot(ptx, hpte[r]/w, 'o-',label=r'$h\to e \in$ [1,9] GeV/c')
    for i,d in enumerate(hptd):
      ax.plot(ptx, d[r]/w, 'o-',
              label=r'$h\to e \in$ [{:.1f},{:.1f}] GeV/c'.format(dcaeptbins[i],
                                                         dcaeptbins[i+1]))

    # ax.plot(ptx, hptmod[r]/w, lw=2, ls='*', marker='s', color='black')
  axes[1].legend(prop={'size':10})
  fig.tight_layout()
  fig.savefig('pdfs/hpt-check.pdf')
  return

def plotbfrac():
  print("plotbfrac()")
  fig, ax = plt.subplots(figsize=(8,6))
  ax.set_ylim([0.,1.])
  ax.set_xlabel(r'$e^{\pm}$ $p_T$ [GeV/c]')
  ax.set_ylabel(r'$b \to e / (b \to e + c \to e)$')
  ax.errorbar(eptx, bfold_ept/hfold_ept, #yerr= [eptf_lo, eptf_hi],
              lw=2, ls='*', marker='s', ms=10, alpha=0.8, color='crimson',
              label=r'$A_{ept}$*bpt / $A_{ept}$*hpt')
  ax.errorbar(dcaeptx, bfrac_dca, 
              lw=2, ls='*', marker='s', ms=10, alpha=0.8, color='blue',
              label=r'$A_{dca}$*bpt / $A_{dca}$*hpt')
  ax.legend(loc=2)
  fig.savefig('pdfs/bfrac-check.pdf')
  return

if __name__=='__main__':
  plotept_matrix(eptMat)
  plotept_dist()
  # plotdca_matrices()
  plotdca_dists()
  plothpt()
  plotbfrac()


