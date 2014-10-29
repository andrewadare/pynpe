import numpy as np
import unfold_input as ui

def dca_refold(gpt, dcamat, dca):
    c, b = ui.idx['c'], ui.idx['b']
    cfold, bfold, hfold = [], [], []
    bfrac = np.zeros((len(dca),3))
    for i, m in enumerate(dcamat):
        cf = np.dot(m[:, c], gpt[c])
        bf = np.dot(m[:, b], gpt[b])
        hf = cf + bf
        scf = dca[i][:, 0].sum() / hf[:, 0].sum()
        cf *= scf
        bf *= scf
        hf *= scf
        cfold.append(cf)
        bfold.append(bf)
        hfold.append(hf)
        bfrac[i,:] = np.sum(bf) / np.sum(hf)

    # Make errors zero until I find time to compute them correctly...
    bfrac[:,1:] *= 0.
    return cfold, bfold, hfold, bfrac


def ept_refold(gpt, eptmat):
    c, b = ui.idx['c'], ui.idx['b']
    cfold = np.dot(eptmat[:, c], gpt[c])
    bfold = np.dot(eptmat[:, b], gpt[b])
    hfold = cfold + bfold
    bfrac = bfold / hfold
    # Make errors zero until I find time to compute them correctly...
    bfrac[:,1:] *= 0.
    return cfold, bfold, hfold, bfrac
