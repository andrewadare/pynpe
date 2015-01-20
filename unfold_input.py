import numpy as np
from scipy.stats import norm
from ROOT import TFile, TH1, TH2D, TCanvas, gStyle
from h2np import h2a
import ppg077data

# Bin edge arrays
dcaeptbins = np.array((1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0))
eptbins = np.array([1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3,
                    3.2, 3.4, 3.6, 3.8, 4, 4.5, 5, 6, 7, 8, 9])
cptbins = np.array([0., 1, 2, 3, 4, 5, 7, 9, 12, 15, 20])
bptbins = cptbins
hptbins = np.hstack((cptbins[:-1], cptbins[-1] + bptbins))
# dcabins = np.linspace(-0.2, 0.2, 101) # for QM12
dcabins = np.linspace(-0.7, 0.7, 701) # for Run 11

# Bin width arrays
eptw = np.diff(eptbins)
cptw = np.diff(cptbins)
bptw = np.diff(bptbins)
hptw = np.diff(hptbins)
dcaw = np.diff(dcabins)

# Bin center arrays
eptx = eptbins[:-1] + eptw / 2
cptx = cptbins[:-1] + cptw / 2
bptx = bptbins[:-1] + bptw / 2
hptx = hptbins[:-1] + hptw / 2
dcax = dcabins[:-1] + dcaw / 2

# Number of "dimensions" = free parameters = bins in unfolding result.
# nfb is currently 7: 6 dca + 1 ept. Indices follow that order.
ncpt = len(cptx)
nbpt = len(bptx)
nhpt = len(hptx)
# nfb = len(dcaeptbins)

# Indices of c or b hadron points within parameter array.
# idx['f'] = 0..6, where 0-5 are the dca ept indices and 6 is for ept.
idx = {'c': np.arange(0, ncpt),
       'b': np.arange(ncpt, ncpt + nbpt)}
# 'f': np.arange(ncpt + nbpt, ncpt + nbpt + nfb)}

maskranges_mb = np.array([
    [0.04, 0.03, -0.15, -0.15, -0.15, -0.15],
    [0.15, 0.15, -0.02, -0.01, -0.01, -0.01],
    [9999, 9999, +0.02, +0.01, +0.01, +0.01],
    [9999, 9999, +0.15, +0.15, +0.15, +0.15]])

maskranges_pp = np.array([
    [-0.15, -0.15, -0.15, -0.15, -0.15, -0.15],
    [-0.01, -0.01, -0.01, -0.01, -0.01, -0.01],
    [+0.01, +0.01, +0.01, +0.01, +0.01, +0.01],
    [+0.15, +0.15, +0.15, +0.15, +0.15, +0.15]])

# Columns: electron pt, mid, high, low.
fonll = np.array([[1.014536231, 0.0381860323, 0.1041675777, 0.0104167578],
                  [1.240923602, 0.0588883544, 0.1422486553, 0.0172563082],
                  [1.436874098, 0.0934971793, 0.1907639830, 0.0378974067],
                  [1.647938077, 0.1280972581, 0.2531595749, 0.0551449687],
                  [1.798600603, 0.1592600691, 0.3397997114, 0.0758385446],
                  [1.964376613, 0.1904141339, 0.4160405825, 0.1000306118],
                  [2.175230682, 0.2389032230, 0.4818559496, 0.1276599466],
                  [2.401250711, 0.2839113132, 0.5407093191, 0.1552980277],
                  [2.657497704, 0.3289019111, 0.5787729042, 0.1898193904],
                  [3.004320637, 0.3807845367, 0.6168015044, 0.2312502733],
                  [3.321074037, 0.4222678970, 0.6478506144, 0.2657366511],
                  [3.773796300, 0.4671447938, 0.6788734858, 0.2898325097],
                  [4.287077448, 0.5050422006, 0.6960685704, 0.3242576639],
                  [4.830690515, 0.5359776097, 0.7200507281, 0.3414177636],
                  [5.253553155, 0.5565662308, 0.7337035903, 0.3620326234],
                  [5.721756243, 0.5771286133, 0.7472864827, 0.3757029781],
                  [5.993641492, 0.5873879390, 0.7575458084, 0.3859972887]])


def dca_subset(dcalist, dcamatlist, dtype):
    '''
    Create subarrays excluding masked data - no gaps, dimensions are smaller.
    Numpy masked arrays were tried, but resulted in ~5x slower runtimes 
    than this approach.
    '''
    subdca = []
    subdcamat = []
    for i, dfull in enumerate(dcalist):
        mfull = dcamatlist[i]
        r = maskranges_mb if dtype == 'AuAu200MB' else maskranges_pp
        a = np.zeros((0, dfull.shape[1]))
        m = np.zeros((0, mfull.shape[1]))
        for j, x in enumerate(dcabins[:-1]):
            if (x > r[0, i] and x < r[1, i]) or (x > r[2, i] and x < r[3, i]):
                a = np.vstack([a, dfull[j, :]])
                m = np.vstack([m, mfull[j, :]])
        subdca.append(a)
        subdcamat.append(m)
    return subdca, subdcamat


def project_and_save(dcares=np.full(len(dcaeptbins), 0.007),
					 dcamean=np.full(len(dcaeptbins), 0.0),
					 draw=False):
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
        print 'Creating', cb[i], 'hadron --> electron pt matrix'
        hbins = cptbins if i == 0 else bptbins

        # Rebin inclusive generated hadron pt histogram and write to csv.
        # This is used for weighting of the decay matrices.
        # hr = hpt[i].Rebin(len(hbins) - 1, cb[i] + 'pt', hbins)
        np.savetxt('csv/{}_pt.csv'.format(cb[i]),
                   h2a_rebin1d(hpt[i], hbins), fmt='%.0f', delimiter=',',
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
            print 'Creating', cb[i], 'hadron --> electron DCA matrix', j
            h.GetXaxis().SetRangeUser(hbins[0], hbins[-1] - 1e-4)
            h.GetYaxis().SetRangeUser(dcaeptbins[j], dcaeptbins[j + 1] - 1e-4)
            h.GetZaxis().SetRangeUser(dcabins[0] + 1e-4, dcabins[-1] - 1e-4)
            hadca = h.Project3D('NUF_NOF_xz')
            hadca.SetName('hadca_{}_{}'.format(cb[i], j))
            hadca.SetTitle('{} hadron pt vs. electron DCA at e pt {}-{} GeV/c'
                           .format(cb[i], dcaeptbins[j], dcaeptbins[j + 1]))
            hadca.SetXTitle('electron DCA [cm]')
            hadca.SetYTitle('{} hadron pt'.format(cb[i]))

            # I think this is the ~20 micron shift. remove
            # a = h2a_rebin2d(hadca, dcabins, hbins, xbinshift=2)
            a = h2a_rebin2d(hadca, dcabins, hbins, xbinshift=0)

            if dcares[j] > 0:
                # Apply convolution to columns of a, each column being the DCA
                # distribution at hadron pt bin `col'.
                ker = norm(loc=dcamean[j], scale=dcares[j]).pdf(dcax)
                ker *= 1. / ker.sum()
                for col in range(a.shape[1]):
                    a[:, col] = np.convolve(a[:, col], ker, 'same')

            np.savetxt('csv/{}_to_dca_{}.csv'.format(cb[i], j),
                       a,
                       fmt='%.1f', delimiter=',',
                       header='e dca (row) binning: \n{}\n'
                       '{} hadron pt (column) binning: \n{}'
                       .format(dcabins, cb[i], hbins))
            if draw == True:
                c = TCanvas('{}_dca_{}'.format(cb[i], j), 'c', 500, 500)
                hadca.Draw('col')
                c.SetLogz()
                c.SaveAs('pdfs/{}_to_dca_{}.pdf'.format(cb[i], j))


def h2a_rebin1d(h, newxbins, eps=1e-6):
    a = np.zeros([len(newxbins) - 1])
    for i, xlo in enumerate(newxbins[:-1]):
        xhi = newxbins[i + 1]
        ilo = h.GetXaxis().FindBin(xlo + eps)
        ihi = h.GetXaxis().FindBin(xhi - eps)
        a[i] = h.Integral(ilo, ihi)
    return a


def h2a_rebin2d(h, newxbins, newybins, xbinshift=0, eps=1e-6):
    a = np.zeros([len(newxbins) - 1, len(newybins) - 1])
    for j, ylo in enumerate(newybins[:-1]):
        for i, xlo in enumerate(newxbins[:-1]):
            xhi = newxbins[i + 1] - eps
            yhi = newybins[j + 1] - eps
            ilo = np.max((h.GetXaxis().FindBin(xlo + eps) + xbinshift, 1))
            ihi = np.min(
                (h.GetXaxis().FindBin(xhi) + xbinshift, h.GetNbinsX()))
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


def genpt(bfrac=None):
    chpt = np.loadtxt('csv/c_pt.csv', delimiter=',')
    bhpt = np.loadtxt('csv/b_pt.csv', delimiter=',')
    if bfrac is not None:
        chpt *= 1 - bfrac
        bhpt *= bfrac
    gpt = np.hstack((chpt, bhpt))
    e = np.sqrt(gpt)
    gpt = np.vstack((gpt, e, e)).T
    return gpt


def eptmatrix(weighted=True):
    cmat = np.loadtxt('csv/c_to_ept.csv', delimiter=',')
    bmat = np.loadtxt('csv/b_to_ept.csv', delimiter=',')
    if weighted == True:
        chpt = np.loadtxt('csv/c_pt.csv', delimiter=',')
        bhpt = np.loadtxt('csv/b_pt.csv', delimiter=',')
        cmat /= chpt
        bmat /= bhpt
    return np.hstack((cmat, bmat))


def dcamatrix(dca_ept_bin, weighted=True):
    cfile = 'csv/c_to_dca_{}.csv'.format(dca_ept_bin)
    bfile = 'csv/b_to_dca_{}.csv'.format(dca_ept_bin)
    cmat = np.loadtxt(cfile, delimiter=',')
    bmat = np.loadtxt(bfile, delimiter=',')
    if weighted == True:
        chpt = np.loadtxt('csv/c_pt.csv', delimiter=',')
        bhpt = np.loadtxt('csv/b_pt.csv', delimiter=',')
        cmat /= chpt
        bmat /= bhpt
    return np.hstack((cmat, bmat))


def eptmat_proj(bfrac, axis):
    '''
    Returns a projection from the unweighted decay matrix.
    Useful for unfolding tests on a fully self-consistent system.
    axis 0: hadron pt
    axis 1: electron pt
    '''
    cmat = np.loadtxt('csv/c_to_ept.csv', delimiter=',')
    bmat = np.loadtxt('csv/b_to_ept.csv', delimiter=',')
    m = np.hstack(((1. - bfrac) * cmat, bfrac * bmat))
    proj = m.sum(axis)
    # Add error column.
    proj = np.vstack((proj, np.sqrt(proj))).T
    return proj


def dcamat_proj(dca_ept_bin, bfrac, axis):
    '''
    Returns a projection from the unweighted decay matrix.
    Useful for unfolding tests on a fully self-consistent system.
    axis 0: hadron pt
    axis 1: electron DCA
    '''
    cfile = 'csv/c_to_dca_{}.csv'.format(dca_ept_bin)
    bfile = 'csv/b_to_dca_{}.csv'.format(dca_ept_bin)
    cmat = np.loadtxt(cfile, delimiter=',')
    bmat = np.loadtxt(bfile, delimiter=',')
    m = np.hstack(((1. - bfrac) * cmat, bfrac * bmat))
    proj = m.sum(axis)
    # Add error column.
    proj = np.vstack((proj, np.sqrt(proj))).T
    return proj


def eptdata(data_type, rand_syserr=False, err_type='SysStat'):
    '''
    Return 2-D numpy array of electron spectra.
    Column 0 contains data, column 1 contains stat error.
    data_type can be 'AuAu200MB' or 'pp200'.
    '''
    d = reload(ppg077data)
    if data_type == 'AuAu200MB':
        pts = d.yinv_mb[7:]
        stat_err = 0.5 * (d.statlo_mb[7:] + d.stathi_mb[7:])
        syst_err = 0.5 * (d.syslo_mb[7:] + d.syshi_mb[7:])
        if err_type == 'Stat':
        	print("Using statistical errors on electron spectra only")
        	err = stat_err	
        else:
        	print("Using statistical+systematic errors on electron spectra")
        	err = np.sqrt(stat_err ** 2 + syst_err ** 2)

        # randomly sample the systematic error if desired
        if rand_syserr:
            # Only use statistical error when shifting systematics
            err = stat_err

            # Generate the sigma at the given pt
            # based on an overall global shift
            # and a tilt about the center of the
            # distribution
            sig = (2 * eptx - (eptx[0] + eptx[-1])) / (eptx[0] - eptx[-1]) \
                * np.random.normal() + np.random.normal()

            # Shift the data by the systematic
            pts = pts + sig * syst_err

        # Multiply by bin width
        pts *= np.diff(d.eptbins[7:])
        err *= np.diff(d.eptbins[7:])
        return np.vstack((pts, err)).T

    elif data_type == 'pp200':
        pts = d.xsec_pp[7:]
        err = np.sqrt(d.stat_pp[7:] ** 2 + d.syst_pp[7:] ** 2)

        # Multiply by bin width
        pts *= np.diff(d.eptbins[7:])
        err *= np.diff(d.eptbins[7:])
        return np.vstack((pts, err)).T

    else:
        print('Error: data_type "{}" not recognized'.format(data_type))
        return


def dcadata(dca_ept_bin, data_type, filename='rootfiles/run11DCA.root'):
    '''
    Return 1-D numpy array of electron DCA yields.
    data_type can be 'AuAu200MB' or 'pp200'.
    Column 0 is data, column 1 is background.
    '''
    dtypes = {'AuAu200MB': 'MB', 'pp200': 'PP'}
    f = TFile(filename)
    dcaname = 'run11{}dca{}'.format(dtypes[data_type], dca_ept_bin)
    bkgname = 'run11{}bkg{}'.format(dtypes[data_type], dca_ept_bin)
    # dcaname = 'qm12{}dca{}'.format(dtypes[data_type], dca_ept_bin)
    # bkgname = 'qm12{}bkg{}'.format(dtypes[data_type], dca_ept_bin)
    hdca = f.Get(dcaname)
    hbkg = f.Get(bkgname)

    assert isinstance(hdca, TH1)
    assert isinstance(hbkg, TH1)

    a = np.vstack((h2a(hdca), h2a(hbkg))).T
    return a


def dcadata_sim(dca_ept_bin, bfrac, dtype='MB'):
    # statistics of QM12 DCA distributions:
    qm12ppbkg = [17948, 3386,  776, 212,  87, 13]
    qm12pptot = [29492, 7317, 2221, 811, 482, 98]
    qm12mbbkg = [91148, 11388, 2148, 556, 191, 34]
    qm12mbtot = [181649, 31795, 7172, 2171, 1010, 170]

    tot = qm12mbtot[dca_ept_bin]  # TODO: generalize

    dproj = dcamat_proj(dca_ept_bin, bfrac, axis=1)
    dproj *= tot / dproj.sum()
    for j, mu in enumerate(dproj[:, 0]):
        dproj[j, 0] = np.random.poisson(mu)

    # Background is zeros for now in simulation
    dproj[:, 1] = np.zeros_like(dproj[:, 0])
    return dproj


def eptdata_sim(bfrac, integral, dtype,):
    ept_rd = eptdata(dtype)
    ept = eptmat_proj(bfrac, axis=1)
    ept *= integral / ept.sum()

    for j, err in enumerate(ept_rd[:, 1]):
        fe = err / ept_rd[j, 0]
        e = ept[j, 0] * fe
        ept[j, 0] += err * np.random.randn()
        ept[j, 1] = err

    return ept


if __name__ == '__main__':
    np.set_printoptions(precision=3)

    # Generate csv files and plots with current settings
    project_and_save()
