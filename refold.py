import numpy as np
import unfold_input as ui
import scipy.stats as stats


def dca_refold(gpt, dcamat, dca, dtype='AuAu200MB', add_bkg=False):
    c, b = ui.idx['c'], ui.idx['b']
    cfold, bfold, hfold, rfold, pfold = [], [], [], [], []
    subdca, subdcamat = ui.dca_subset(dca, dcamat, dtype)
    bfrac = np.zeros((len(dca), 3))
    for i, m in enumerate(dcamat):
        cf = np.dot(m[:, c], gpt[c])
        bf = np.dot(m[:, b], gpt[b])
        hf = cf + bf
        subhf = np.dot(subdcamat[i][:, c], gpt[c]) + np.dot(subdcamat[i][:, b], gpt[b])
        # scf = (dca[i][:, 0].sum() - dca[i][:, 1].sum()) / hf[:, 0].sum()
        scf = (subdca[i][:, 0].sum() - subdca[i][:, 1].sum()) / subhf[:, 0].sum()
        cf *= scf
        bf *= scf
        hf *= scf

        if add_bkg == True:
            hf[:, 0] += dca[i][:, 1]

        rf = hf[:, 0] / dca[i][:, 0] 
        pf = stats.poisson.pmf(dca[i][:, 0], hf[:, 0])

        cfold.append(cf)
        bfold.append(bf)
        hfold.append(hf)
        rfold.append(rf)
        pfold.append(pf)
        bfrac[i, :] = np.sum(bf[:, 0]) / (np.sum(bf[:, 0]) + np.sum(cf[:, 0]))

        # Error propagation
        bs = bf.sum(axis=0)[0]
        cs = cf.sum(axis=0)[0]
        df = 1. / (cs + bs) ** 2 * np.sqrt(cs * cs * bs + bs * bs * cs)
        
        bfrac[i, 1] = df
        bfrac[i, 2] = df



    return cfold, bfold, hfold, bfrac, rfold, pfold


def ept_refold(gpt, eptmat):
    c, b = ui.idx['c'], ui.idx['b']
    cfold = np.dot(eptmat[:, c], gpt[c])
    bfold = np.dot(eptmat[:, b], gpt[b])
    hfold = cfold + bfold
    bfrac = bfold / hfold
    cf, bf, hf = cfold[:, 0], bfold[:, 0], hfold[:, 0] # center points (y)
    cl, bl, hl = cfold[:, 2], bfold[:, 2], hfold[:, 2] # low error (dy)

    # Don't allow lower limit of error bars to reach <= zero
    r = cl >= cf # offending rows
    cfold[r,2] = 0.999*cfold[r,0] 
    r = bl >= bf
    bfold[r,2] = 0.999*bfold[r,0] 
    r = hl >= hf
    hfold[r,2] = 0.999*hfold[r,0] 

    # Propagate errors to b fraction
    # TODO: limit upper error extent <= 1
    dc_hi, dc_lo = cfold[:, 1], cfold[:, 2]
    db_hi, db_lo = bfold[:, 1], bfold[:, 2]
    df_hi = 1. / (bf + cf) ** 2 * \
        np.sqrt(cf * cf * db_hi * db_hi + bf * bf * dc_hi * dc_hi)
    df_lo = 1. / (bf + cf) ** 2 * \
        np.sqrt(cf * cf * db_lo * db_lo + bf * bf * dc_lo * dc_lo)
    bfrac[:, 1] = df_hi
    bfrac[:, 2] = df_lo
    
    return cfold, bfold, hfold, bfrac


# if __name__ == '__main__':
#     pass
    # pq =
