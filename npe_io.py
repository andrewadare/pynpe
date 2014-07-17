'''
npe_io.py
'''

import numpy as np
from ROOT import TFile, TH1
from h2np import h2a, binedges, binctrs

def checkobjs(objs, source):
  for (i,obj) in enumerate(objs):
    if obj==None:
      print("Error: Object {} not retrieved from {}.".format(i, source))
      print("Exit")
      sys.exit()

def binwidths(bins):
  return np.array([j-i for i,j in zip(bins[:-1], bins[1:])])

def modelfile(dcares=0.007, bfrac=0.03, wt=''):
  s = 'bfrac{0:.3f}-dcares{1:.0f}um{2}.root'.format(bfrac, dcares*1e4, wt)
  return TFile('rootfiles/' + s)

def eptmatrix(dcares=0.007, bfrac=0.03, wt=''):
  f = modelfile(dcares, bfrac, wt)
  eptHist = f.Get('hAept')
  eptMat = h2a(eptHist)
  return eptMat

def dcamatrices(dcares=0.007, bfrac=0.03, wt=''):
  dcaHists = [None]*6
  f = modelfile(dcares, bfrac, wt)
  for i in range(6):
    h = f.Get('hdca{}'.format(i))
    dcaHists[i] = h 
  dcaMat = [h2a(h) for h in dcaHists]
  return dcaMat

def hadronpt(dcares=0.007, bfrac=0.03, wt=''):
  f = modelfile(dcares, bfrac, wt)
  eptHist = f.Get('hAept')
  eptMat  = h2a(eptHist)
  hpt     = eptMat.sum(axis=0)
  hptx    = binctrs(eptHist,'y')
  hptbins = binedges(eptHist,'y')
  return hpt, hptx, hptbins

def dcabins(dcares=0.007, bfrac=0.03, wt=''):
  f = modelfile(dcares, bfrac, wt)
  dcaHist = f.Get('hdca0')
  dcax    = binctrs(dcaHist,'x')
  dcaeptbins = np.array((1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0))
  dcaeptx    = dcaeptbins[:-1] + binwidths(dcaeptbins)
  return dcax, dcaeptx, dcaeptbins

def eptbins(dcares=0.007, bfrac=0.03, wt=''):
  f = modelfile(dcares, bfrac, wt)
  eptHist = f.Get('hAept')
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
