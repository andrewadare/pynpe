import numpy as np
from ROOT import TFile, TH1, TH2D, TCanvas, gStyle
from h2np import h2a

# Bin edge arrays
dcaeptbins = np.array((1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0))
eptbins = np.array([1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3,
                    3.2, 3.4, 3.6, 3.8, 4, 4.5, 5, 6, 7, 8, 9])
dptbins = np.array([0., 1, 2, 3, 4, 5, 7, 9, 12, 15, 20])
bptbins = dptbins
hptbins = np.hstack((dptbins[:-1], dptbins[-1] + bptbins))
dcabins = np.linspace(-0.2, 0.2, 101)

# Bin center arrays
eptx = eptbins[:-1] + np.diff(eptbins) / 2
dptx = dptbins[:-1] + np.diff(dptbins) / 2
bptx = bptbins[:-1] + np.diff(bptbins) / 2
hptx = hptbins[:-1] + np.diff(hptbins) / 2
dcax = dcabins[:-1] + np.diff(dcabins) / 2


def project_and_save(draw=False):
    '''
    How DCA_pt_template TH3Fs are binned
    ------------------------------------
     x: 200 bins, 0-20 GeV/c (HF hadron pt)
     y: 100 bins, 0-10 GeV/c (decay electron pt)
     z: 700 bins, -0.7 to 0.7 cm (electron DCA)
    '''
    fc = TFile('rootfiles/gen_c.root')
    fb = TFile('rootfiles/gen_b.root')
    hcb = [fc.Get('DCA_pt_template_4'), fb.Get('DCA_pt_template_5')]
    hpt = [fc.Get('hadron_pt_4'), fb.Get('hadron_pt_5')]
    cb = ['c', 'b']
    gStyle.SetOptStat(0)
    for i, h in enumerate(hcb):
        hbins = dptbins if i == 0 else bptbins

        # Rebin inclusive generated hadron pt histogram and write to csv.
        # This is used for weighting of the decay matrices.
        hr = hpt[i].Rebin(len(hbins)-1, cb[i] + 'pt', hbins)
        np.savetxt('csv/{}_pt.csv'.format(cb[i]),
                   h2a(hr), fmt='%.0f', delimiter=',',
                   header='Inclusive {} hadron pt binning: \n{}'
                   .format(cb[i], hbins))

        # Project TH3 to hadron pt vs electron pt
        h.GetXaxis().SetRangeUser(hbins[0], hbins[-1] - 1e-4)
        h.GetYaxis().SetRangeUser(eptbins[0], eptbins[-1] - 1e-4)
        h.GetZaxis().SetRangeUser(-0.7 + 1e-4, 0.7 - 1e-4)
        haept = h.Project3D('NUF_NOF_xy')
        haept.SetName('haept_{}'.format(cb[i]))
        haept.SetTitle('{} hadron pt vs. electron pt;electron pt;{} hadron pt'
                       .format(cb[i], cb[i]))
        np.savetxt('csv/{}_to_ept.csv'.format(cb[i]),
                   h2a_rebin2d(haept, eptbins, hbins),
                   fmt='%.0f', delimiter=',',
                   header='e pt (row) binning: \n{}\n'
                   '{} hadron pt (column) binning: \n{}'
                   .format(eptbins, cb[i], hbins))
        if draw == True:
            c = TCanvas('{}_ept'.format(cb[i]), 'c', 500, 500)
            haept.Draw('col')
            c.SetLogz()
            c.SaveAs('pdfs/{}_to_ept.pdf'.format(cb[i]))

        # Project TH3 to hadron pt vs electron DCA in each electron pt bin
        for j, lowedge in enumerate(dcaeptbins[:-1]):
            h.GetXaxis().SetRangeUser(hbins[0], hbins[-1] - 1e-4)
            h.GetYaxis().SetRangeUser(dcaeptbins[j], dcaeptbins[j + 1] - 1e-4)
            h.GetZaxis().SetRangeUser(dcabins[0] + 1e-4, dcabins[-1] - 1e-4)
            hadca = h.Project3D('NUF_NOF_xz')
            hadca.SetName('hadca_{}_{}'.format(cb[i], j))
            hadca.SetTitle('{} hadron pt vs. electron DCA at e pt {}-{} GeV/c'
                           .format(cb[i], dcaeptbins[j], dcaeptbins[j + 1]))
            hadca.SetXTitle('electron DCA [cm]')
            hadca.SetYTitle('{} hadron pt'.format(cb[i]))

            # TODO: apply convolution and shift

            np.savetxt('csv/{}_to_dca_{}.csv'.format(cb[i], j),
                       h2a_rebin2d(hadca, dcabins, hbins),
                       fmt='%.1f', delimiter=',',
                       header='e dca (row) binning: \n{}\n'
                       '{} hadron pt (column) binning: \n{}'
                       .format(dcabins, cb[i], hbins))
            if draw == True:
                c = TCanvas('{}_dca_{}'.format(cb[i], j), 'c', 500, 500)
                hadca.Draw('col')
                c.SetLogz()
                c.SaveAs('pdfs/{}_to_dca_{}.pdf'.format(cb[i], j))


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

            if False and h.GetName() == 'haept_c':  # Debug / cross check
                xa = h.GetXaxis().GetBinLowEdge(ilo)
                xb = h.GetXaxis().GetBinUpEdge(ihi)
                ya = h.GetYaxis().GetBinLowEdge(jlo)
                yb = h.GetYaxis().GetBinUpEdge(jhi)
                print("ix: {}-{} jy: {}-{} ept: {}-{} hpt: {}-{}"
                      .format(ilo, ihi, jlo, jhi, xa, xb, ya, yb))
    return a


def eptmatrix(bfrac):
    cfile = 'csv/c_to_ept.csv'
    bfile = 'csv/b_to_ept.csv'
    cmat = np.loadtxt(cfile, delimiter=',')
    bmat = np.loadtxt(bfile, delimiter=',')
    return np.hstack(((1. - bfrac) * cmat, bfrac * bmat))


def dcamatrix(bfrac, eptbin):
    cfile = 'csv/c_to_dca_{}.csv'.format(eptbin)
    bfile = 'csv/b_to_dca_{}.csv'.format(eptbin)
    cmat = np.loadtxt(cfile, delimiter=',')
    bmat = np.loadtxt(bfile, delimiter=',')
    return np.hstack(((1. - bfrac) * cmat, bfrac * bmat))


if __name__ == '__main__':
    # Generate csv files and plots with current settings
    project_and_save()
