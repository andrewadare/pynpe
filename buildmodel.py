import os, sys
from ROOT import gSystem, gROOT, TFile, TH3F
from npe_io import checkobjs

wt = ''           # or "-weighted"
bfrac = 0.0072    # FONLL predicts approx. 0.0072
dcares = 0.007    # 0.007 cm in MB Au+Au, 0.014 cm in p+p.
norm = '_normalized' if wt=='-weighted' else ''
gSystem.Load('../dcagen/DcaGen_C.so')
from ROOT import DcaGen

cfile = TFile('../dcagen/rootfiles/gen_c.root')
bfile = TFile('../dcagen/rootfiles/gen_b.root')
hc = cfile.Get('DCA_pt_template{}_4'.format(norm))
hb = bfile.Get('DCA_pt_template{}_5'.format(norm))
h4 = cfile.Get("hadron_pt_4")
h5 = bfile.Get("hadron_pt_5")

ofile = 'rootfiles/bfrac{0:.3f}-dcares{1:.0f}um{2}.root'.format(bfrac, 
                                                                dcares*1e4, 
                                                                wt)
checkobjs([hc,hb,h4,h5], 'gen_{c,b}.root')

gen = DcaGen(hc, hb)
gen.SetBFrac(bfrac)
gen.SetDcaResolution(dcares)
gen.BookHistos()
gen.WriteToRootFile(ofile)

outfile = TFile(ofile, 'update')
h4.Write()
h5.Write()
