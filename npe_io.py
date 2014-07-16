'''
npe_io.py
'''

import numpy as np
from ROOT import TFile, TH1
from h2np import h2a, binedges, binctrs

def checkobjs(objs, source):
  for (i,obj) in enumerate(objs):
    if obj==None:
      print("Error: TObject {} not retrieved from {}.".format(i, source))
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
  # checkobjs([item for sublist in [eptHist,dcaHists] for item in sublist],
  #           f.GetName())
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

  eptx = binctrs(eptHist,'x')
  hptx = binctrs(eptHist,'y')
  dcax = binctrs(dcaHists[0],'x')

  return eptMat, dcaMat, hptx, eptx, dcax

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

# def getbinning(dcares=0.007, bfrac=0.03, wt=''):
#   eptHist, dcaHists = getmodelhists(dcares, bfrac, wt)
#   eptx = binctrs(eptHist,'x')
#   hptx = binctrs(eptHist,'y')
#   dcax = binctrs(dcaHists[0],'x')
#   return hptx, eptx, dcax

def getraa(ptx, k):
  
  def draa(x):
    return 1.3*np.sqrt(2*np.pi*1.1*1.1)*norm.pdf(x, loc=1.5, scale=1.1) + \
    0.2/(1 + np.exp(-x+3))

  def braa(x):
    return 0.6*np.exp(-x/3) + \
    1.1*np.sqrt(2*np.pi*1.5*1.5)*norm.pdf(x, loc=3.4, scale=1.5) + \
    0.3/(1 + np.exp(-x+7))

  return np.concatenate((draa(ptx[:k]), braa(ptx[:k])))
