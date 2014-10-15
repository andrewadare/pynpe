import numpy as np
from ROOT import TFile, TH1, TH2D, TCanvas, gStyle
from h2np import h2a, binedges, binctrs
from os import path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matrixplotter import mmplot


dcaeptbins = np.array((1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0))
eptbins = np.array([1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3,
                    3.2, 3.4, 3.6, 3.8, 4, 4.5, 5, 6, 7, 8, 9])
dptbins = np.array([0., 1, 2, 3, 4, 5, 7, 9, 12, 15, 20])
bptbins = dptbins
hptbins = np.hstack((dptbins[:-1], dptbins[-1] + bptbins))
dcabins = np.linspace(-0.2, 0.2, 101)


eptx = eptbins[:-1] + np.diff(eptbins)/2
dptx = dptbins[:-1] + np.diff(dptbins)/2
bptx = bptbins[:-1] + np.diff(bptbins)/2
hptx = hptbins[:-1] + np.diff(hptbins)/2
dcax = dcabins[:-1] + np.diff(dcabins)/2


def project_and_save():
    fc = TFile('rootfiles/gen_c.root')
    fb = TFile('rootfiles/gen_b.root')
    hcb = [fc.Get('DCA_pt_template_4'), fb.Get('DCA_pt_template_5')]
    labels = ['c', 'b']
    gStyle.SetOptStat(0)
    for i, h in enumerate(hcb):
        # h.RebinX(10)  # HF hadron pt (200, 0, 20)
        # # h.RebinY(10) # electron pt (100, 0, 10) No rebinning
        # # TODO: apply convolution/shift before z rebin
        # h.RebinZ(2)  # electron dca (700, -0.7, +0.7)

        h.GetXaxis().SetRangeUser(dptbins[0], dptbins[-1] - 1e-4)
        h.GetYaxis().SetRangeUser(eptbins[0], eptbins[-1] - 1e-4)
        h.GetZaxis().SetRangeUser(-0.7 + 1e-4, 0.7 - 1e-4)
        haept = h.Project3D('NUF_NOF_xy')
        haept.SetName('haept_{}'.format(labels[i]))
        haept.SetTitle('{} hadron pt vs. electron pt;electron pt;{} hadron pt'
                       .format(labels[i], labels[i]))
        c = TCanvas('{}_ept'.format(labels[i]), 'c', 500, 500)
        haept.Draw('col')
        c.SetLogz()
        c.SaveAs('pdfs/{}_to_ept.pdf'.format(labels[i]))
        # write_csv(h2a(haept), 'csv/{}_to_ept.csv'.format(labels[i]))
        write_csv(h2a_rebin2d(haept, eptbins, dptbins),
                  'csv/{}_to_ept_rebin.csv'.format(labels[i]))
        # np.savetxt('csv/{}_to_ept.csv'.format(labels[i]), h2a(haept),
        #            fmt='%.0f', delimiter=',')

        for j, lowedge in enumerate(dcaeptbins[:-1]):
            h.GetXaxis().SetRangeUser(dptbins[0], dptbins[-1] - 1e-4)
            h.GetYaxis().SetRangeUser(dcaeptbins[j], dcaeptbins[j + 1] - 1e-4)
            h.GetZaxis().SetRangeUser(dcabins[0] + 1e-4, dcabins[-1] - 1e-4)
            hadca = h.Project3D('NUF_NOF_xz')
            hadca.SetName('hadca_{}_{}'.format(labels[i], j))
            # print hadca.GetNbinsX(),
            # hadca.GetXaxis().GetXmin(),hadca.GetXaxis().GetXmax()
            hadca.SetTitle('{} hadron pt vs. electron DCA at e pt {}-{} GeV/c;DCA [cm];{} hadron pt'
                           .format(labels[i], dcaeptbins[j], dcaeptbins[j + 1], labels[i]))
            c = TCanvas('{}_dca_{}'.format(labels[i], j), 'c', 500, 500)
            hadca.Draw('col')
            c.SetLogz()
            c.SaveAs('pdfs/{}_to_dca_{}.pdf'.format(labels[i], j))
            # write_csv(h2a(hadca), 'csv/{}_to_dca_{}.csv'.format(labels[i], j))
            write_csv(h2a_rebin2d(hadca, dcabins, dptbins),
                      'csv/{}_to_dca_rebin.csv'.format(labels[i]))


def h2a_rebin2d(h, newxbins, newybins, eps=1e-6):
    a = np.zeros([len(newxbins) - 1, len(newybins) - 1])
    for j, ylo in enumerate(newybins[:-1]):
        for i, xlo in enumerate(newxbins[:-1]):
            xhi = newxbins[i + 1] - eps
            yhi = newybins[j + 1] - eps
            ilo = h.GetXaxis().FindBin(xlo + eps)
            ihi = h.GetXaxis().FindBin(xhi)
            jlo = h.GetYaxis().FindBin(ylo + eps)
            jhi = h.GetYaxis().FindBin(yhi)

            a[i, j] = h.Integral(ilo, ihi, jlo, jhi)

            if False and h.GetName() == 'haept_c': # Debug / cross check
                xa = h.GetXaxis().GetBinLowEdge(ilo)
                xb = h.GetXaxis().GetBinUpEdge(ihi)
                ya = h.GetYaxis().GetBinLowEdge(jlo)
                yb = h.GetYaxis().GetBinUpEdge(jhi)
                print("ix: {}-{} jy: {}-{} ept: {}-{} hpt: {}-{}"
                      .format(ilo, ihi, jlo, jhi, xa, xb, ya, yb))
    return a


def write_csv(a, filename):
    '''
    Write a 1- or 2-dimensional numpy array to csv file.
    Does not work with arrays of rank > 2 (a savetxt() limitation). 
    Includes a header with dimension info, which can be parsed by get_dims().
    '''
    with file(filename, 'w') as outfile:
        print('Writing array with shape {} to {}...'.format(a.shape, filename))
        outfile.write('# Array shape: {}\n'.format(a.shape))
        np.savetxt(outfile, a, fmt='%.0f', delimiter=',')


def get_dims(filename):
    '''
    Parse first line of csv file written above to return shape.
    '''
    with open(filename, 'r') as f:
        l = f.readline()
        dims = l[l.find("(") + 1:l.find(")")].split(', ')
        return [int(d) for d in dims]


def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def eptmatrix(bfrac):
    # Load a CSV to array and plot
    cfile = 'csv/c_to_ept_rebin.csv'
    bfile = 'csv/b_to_ept_rebin.csv'
    cmat = np.loadtxt(cfile, delimiter=',').reshape(get_dims(cfile))
    bmat = np.loadtxt(bfile, delimiter=',').reshape(get_dims(bfile))
    return np.hstack(((1.-bfrac)*cmat, bfrac*bmat))

if __name__ == '__main__':

    # Open two 3-D ROOT histograms, rebin, and write to csv file
    # if find('dmeson_decays.csv', 'csv') == '' \
    # or find('bmeson_decays.csv', 'csv') == '':
