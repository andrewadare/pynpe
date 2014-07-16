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

def getmodelhists(dcares=0.007, bfrac=0.03, wt=''):
  dcaHists = [None]*6
  s = 'bfrac{0:.3f}-dcares{1:.0f}um{2}.root'.format(bfrac, dcares*1e4, wt)
  f = TFile('rootfiles/' + s)
  eptHist = f.Get('hAept')
  # hptGen = fe.Get('hptGen')
  for i in range(6):
    h = f.Get('hdca{}'.format(i))
    dcaHists[i] = h 
    print dcaHists[i].GetName(), dcaHists[i].GetEntries()
  checkobjs([item for sublist in [eptHist,dcaHists] for item in sublist],
            f.GetName())
  return eptHist, dcaHists

def getmodel(dcares=0.007, bfrac=0.03, wt=''):
  dcaHists = [None]*6
  s = 'bfrac{0:.3f}-dcares{1:.0f}um{2}.root'.format(bfrac, dcares*1e4, wt)
  f = TFile('rootfiles/' + s)
  eptHist = f.Get('hAept')
  # hptGen = fe.Get('hptGen')
  for i in range(6):
    h = f.Get('hdca{}'.format(i))
    dcaHists[i] = h 

  eptMat = h2a(eptHist)
  dcaMat = [h2a(h) for h in dcaHists]

  hptx = binctrs(eptHist,'y')
  eptx = binctrs(eptHist,'x')
  dcax = binctrs(dcaHists[0],'x')

  hptbins = binedges(eptHist,'y')
  eptbins = binedges(eptHist,'x')

  return eptMat, dcaMat, hptx, eptx, dcax, hptbins, eptbins

def getdatahists(dtype='MC'):
  hdca   = [None]*6
  hbkg   = [None]*6
  de = {'MC':'simspectra', 'PP': 'ppg077spectra', 'MB': 'ppg077spectra'}
  dd = {'MC':'simdca',     'PP': 'qm12dca',       'MB': 'qm12dca'}
  fe = TFile('rootfiles/{}.root'.format(de[dtype]))
  fd = TFile('rootfiles/{}.root'.format(dd[dtype]))

  hept = fe.Get('hEpt' + dtype)

  for i in range(6):
    prefix = '' if dtype=='MC' else 'qm12'
    hdca[i] = fd.Get('{}{}dca{}'.format(prefix, dtype, i))
    hbkg[i] = fd.Get('{}{}bkg{}'.format(prefix, dtype, i))

  return hept, hdca, hbkg

def getdata(dtype='MC'):
  hdca   = [None]*6
  hbkg   = [None]*6
  de = {'MC':'simspectra', 'PP': 'ppg077spectra', 'MB': 'ppg077spectra'}
  dd = {'MC':'simdca',     'PP': 'qm12dca',       'MB': 'qm12dca'}
  fe = TFile('rootfiles/{}.root'.format(de[dtype]))
  fd = TFile('rootfiles/{}.root'.format(dd[dtype]))

  hept = fe.Get('hEpt' + dtype)

  for i in range(6):
    prefix = '' if dtype=='MC' else 'qm12'
    hdca[i] = fd.Get('{}{}dca{}'.format(prefix, dtype, i))
    hbkg[i] = fd.Get('{}{}bkg{}'.format(prefix, dtype, i))

  ept     = h2a(hept)
  ept_err = h2a(hept, 'e')
  dca = [h2a(h) for h in hdca]
  bkg = [h2a(h) for h in hbkg]
  return ept, ept_err, dca, bkg
