'''
npe_io.py
Input module for electron unfolding analysis

Example usage:
  import npe_io as io
  # Assign mf before calling functions:
  io.mf = modelfile(0.007, 0.01) # TFile object returned by modelfile
  eptMat = io.eptmatrix()
'''

import numpy as np
from ROOT import TFile, TH1
from h2np import h2a, binedges, binctrs

mf = None
dcaeptbins = np.array((1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0))

def modelfile(dcares, bfrac): # Args: DCA resolution [cm], b/(b+c) fraction
  s = 'bfrac{0:.3f}-dcares{1:.0f}um.root'.format(bfrac, dcares*1e4)
  return TFile('rootfiles/' + s)

def checkobjs(objs, source):
  for (i,obj) in enumerate(objs):
    if obj==None:
      print("Error: Object {} not retrieved from {}.".format(i, source))
      print("Exit")
      sys.exit()

def binwidths(bins):
  return np.array([j-i for i,j in zip(bins[:-1], bins[1:])])

def eptmatrix():
  eptHist = mf.Get('hAept')
  eptMat = h2a(eptHist)
  return eptMat

def dcamatrices():
  dcaHists = [None]*6
  for i in range(6):
    h = mf.Get('hdca{}'.format(i))
    dcaHists[i] = h 
  dcaMat = [h2a(h) for h in dcaHists]
  return dcaMat

def dcabins():
  dcaHist = mf.Get('hdca0')
  dcax    = binctrs(dcaHist,'x')
  dcabins = binedges(dcaHist,'x')
  dcaeptx = dcaeptbins[:-1] + 0.4*binwidths(dcaeptbins) # ~ <pt> in bin TODO: do it right
  return dcax, dcabins, dcaeptx, dcaeptbins

def eptbins():
  eptHist = mf.Get('hAept')
  eptx    = binctrs(eptHist,'x')
  eptbins = binedges(eptHist,'x')
  return eptx, eptbins

def eptdata(dtype='MC'):
  de = {'MC':'simspectra', 'PP':'ppg077spectra', 'MB':'ppg077spectra'}
  fe = TFile('rootfiles/{}.root'.format(de[dtype]))
  hept = fe.Get('hEpt' + dtype)
  ept     = h2a(hept)
  ept_err = h2a(hept, 'e')
  return ept, ept_err

def dcadata(dtype='MC'):
  hdca   = [None]*6
  hbkg   = [None]*6
  dd = {'MC':'simdca', 'PP':'qm12dca', 'MB':'qm12dca'}
  fd = TFile('rootfiles/{}.root'.format(dd[dtype]))
  for i in range(6):
    prefix = '' if dtype=='MC' else 'qm12'
    hdca[i] = fd.Get('{}{}dca{}'.format(prefix, dtype, i))
    hbkg[i] = fd.Get('{}{}bkg{}'.format(prefix, dtype, i))
  dca = [h2a(h) for h in hdca]
  bkg = [h2a(h) for h in hbkg]
  return dca, bkg

def dcaweights(bfrac):
  hDept = mf.Get('hDept')
  hBept = mf.Get('hBept')
  ce = h2a(hDept.ProjectionX())
  be = h2a(hBept.ProjectionX())
  he = (1-bfrac)*ce + bfrac*be
  wts, bins = np.histogram(binctrs(hDept,'x'), bins=dcaeptbins, weights=he)
  wts /= np.sum(wts)
  return wts

def hadronpt():
  '''
  Return pt distribution of HF hadrons in detector acceptance.
  '''
  eptHist = mf.Get('hAept')
  hptx    = binctrs(eptHist,'y')
  hptbins = binedges(eptHist,'y')
  eptMat  = h2a(eptHist)
  hpte    = eptMat.sum(axis=0)
  hptd    = [m.sum(axis=0) for m in dcamatrices()]
  return hpte, hptd, hptx, hptbins

def genpt(opt=''):
  '''
  Return pt distribution of HF hadrons generated into all phase space.
  Note that gptx = hptx by definition (returned here for convenience).
  '''
  hpte, hptd, gptx, hptbins = hadronpt()
  N = hpte.shape[0]/2 + 1
  h4 = mf.Get('hadron_pt_4')
  h5 = mf.Get('hadron_pt_5')
  c = h2a(h4)
  b = h2a(h5)
  cx = binctrs(h4)
  ch, bins = np.histogram(cx, bins=hptbins[:N], weights=c)
  bh, bins = np.histogram(cx, bins=hptbins[:N], weights=b)
  gpt = np.concatenate([ch,bh])
  if opt=='all':
    return c, b, cx, ch, bh, gpt, gptx
  else:
    return gpt, gptx

if __name__=='__main__':
  import matplotlib.pyplot as plt

  mf = modelfile(0.007, 0.007)
  hpte, hptd, hptx, hptbins = hadronpt()
  c, b, cx, ch, bh, gpt, gptx = genpt('all')
  ndim = hpte.shape[0]

  fig, ax = plt.subplots()
  ax.set_yscale('log')
  ax.set_xlabel(r'hadron $p_T$ [GeV/c]')
  ax.plot(cx, c, 'o', color='white', label='generated c hadrons')
  ax.plot(cx, b, 'o', color='yellow',label='generated b hadrons')
  ax.legend()
  fig.savefig('pdfs/hpt-gen-finebins.pdf')
