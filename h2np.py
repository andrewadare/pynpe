
from ROOT import TH1
import numpy as np

def h2a(h, opt=''):
  '''
  Returns a numpy array from a ROOT histogram. If opt is 'e', return errors.
  '''
  if not isinstance(h, TH1):
    print("Error: Object is not a valid ROOT histogram")
    return

  dims = [h.GetNbinsX(), h.GetNbinsY(), h.GetNbinsZ()]
  n = np.prod(dims)
  if dims[2] == 1: dims.pop()
  if dims[1] == 1: dims.pop()
  a = np.zeros(tuple(dims))

  if opt=='e':
    if a.ndim==1:
      for i in range(dims[0]):
        a[i] = h.GetBinError(i+1)
    if a.ndim==2:
      for j in range(dims[1]):
        for i in range(dims[0]):
          a[i,j] = h.GetBinError(i+1, j+1)
    if a.ndim==3:
      for k in range(dims[2]):
        for j in range(dims[1]):
          for i in range(dims[0]):
            a[i,j,k] = h.GetBinError(i+1, j+1, k+1)
  else:
    if a.ndim==1:
      for i in range(dims[0]):
        a[i] = h.GetBinContent(i+1)
    if a.ndim==2:
      for j in range(dims[1]):
        for i in range(dims[0]):
          a[i,j] = h.GetBinContent(i+1, j+1)
    if a.ndim==3:
      for k in range(dims[2]):
        for j in range(dims[1]):
          for i in range(dims[0]):
            a[i,j,k] = h.GetBinContent(i+1, j+1, k+1)
  return a

def binedges(h, opt='x'):
  '''
  Returns a numpy array of bins from a ROOT histogram along opt=x,y,z.
  '''
  if not isinstance(h, TH1):
    print("Error: Object is not a valid ROOT histogram")
    return

  axis = h.GetXaxis()
  if opt=='y':
    axis = h.GetYaxis()
  if opt=='z':
    axis = h.GetZaxis()

  bins = [axis.GetXbins().GetArray()[i] for i in range(axis.GetNbins()+1)]
  return np.array(bins)

def binctrs(h, opt='x'):
  '''
  Returns a numpy array of bin centers from a ROOT histogram along opt=x,y,z.
  '''
  if not isinstance(h, TH1):
    print("Error: Object is not a valid ROOT histogram")
    return

  axis = h.GetXaxis()
  if opt=='y':
    axis = h.GetYaxis()
  if opt=='z':
    axis = h.GetZaxis()

  bins = [axis.GetBinCenter(i+1) for i in range(axis.GetNbins())]
  return np.array(bins)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if __name__ == '__main__':
  from ROOT import TH2D
  print('Test example: 5 x 3 TH2D with content = bin index')
  h = TH2D('h', 'htest', 5,0.,1., 3,0.,1.)
  for i in range(1,16):
    h.SetBinContent(i,i)

  a = h2a(h)
  print(a)



