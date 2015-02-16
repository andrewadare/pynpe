
import unfold_input as ui
import plotting_functions as pf
import numpy as np
from refold import ept_refold, dca_refold

basedir = "output/Run11AuAu200MB_randdca/"
rangelo = 0
rangehi = 20

#---------------------------------------------
# Plot the electron pT samples
#---------------------------------------------

# ept_mb = ui.eptdata('AuAu200MB',False,'Stat')

# ept_sys = list()
# for samp in range(rangelo, rangehi):
# 	outdir='{}/samp{}'.format(basedir, samp)
# 	readept = "{}/csv/1/ept.csv".format(outdir)
# 	ept_samp = np.genfromtxt(readept, delimiter=",") 
# 	ept_sys.append(ept_samp)

# pf.plot_ept_syssample(ept_mb, ept_sys)

#---------------------------------------------
# Plot the b-fraction variation
#---------------------------------------------

# Get the central value
# pq = np.genfromtxt("output/Run11AuAu200MB_rebin2/csv/2/pq.csv", delimiter=',')
# eptmat = ui.eptmatrix()
# ceptr, beptr, heptr, bfrac_ept = ept_refold(pq, eptmat)

# bfrac_sys = list()
# pq_sys = list()
# for samp in range(rangelo, rangehi):
# 	outdir = "{}/samp{}".format(basedir, samp)
# 	readpq = "{}/csv/2/pq.csv".format(outdir)
# 	pq_samp = np.genfromtxt(readpq, delimiter=',')
# 	ceptr_samp, beptr_samp, heptr_samp, bfrac_samp = ept_refold(pq_samp, eptmat)
# 	bfrac_sys.append(bfrac_samp)
# 	pq_sys.append(pq_samp)

# pf.plotbfrac_syssample(bfrac_ept, bfrac_sys, None, ui.fonll)
# pf.plot_result_sysample(pq, pq_sys)


#---------------------------------------------
# Plot the b-fraction variation
# for a set of files
#---------------------------------------------

# Get the central value
# pq = np.genfromtxt("output/Run11AuAu200MB_fixside/csv/3/pq.csv", delimiter=',')
pq = np.genfromtxt("output/Run11AuAu200MB_fitside/csv/3/pq.csv", delimiter=',')
# pq = np.genfromtxt("output/Run11AuAu200MB_eptonly/csv/3/pq.csv", delimiter=',')
#Temporarily set everything to 0 except the 6-8 hpt bins
# pq[0:6,:] = (0, 0, 0)
# pq[7:10,:] = (0, 0, 0)
# pq[10:16,:] = (0, 0, 0)
# pq[17:20,:] = (0, 0, 0)
eptmat = ui.eptmatrix()
ceptr, beptr, heptr, bfrac_ept = ept_refold(pq, eptmat)

# Get the pythia hadron yields
bfrac = 0.007
gpt_full = ui.genpt(bfrac)
gpt = gpt_full[:, 0]  # Extract column with data, leaving out errors.
ept = ui.eptdata('AuAu200MB')
norm_factor = np.sum(ept[:, 0]) / np.sum(ui.eptmat_proj(bfrac, axis=1)[:, 0])
gpt *= norm_factor
gpt_full[:,1] *= norm_factor
gpt_full[:,2] *= norm_factor

# Take the ratio of the hadron yields to pythia
print(pq[:,0]/gpt)

pf.plot_result_rat(gpt, pq)
pf.plotbfrac(bfrac_ept, None, ui.fonll, 'bfrac-ept.pdf')
pf.plotept_refold(ept, ceptr, beptr, heptr, 'ept-refold.pdf')


filelist = [
"output/Run11AuAu200MB_runfold/csv/3/pq.csv",
"output/Run11AuAu200MB_lunfold/csv/3/pq.csv",
]
bfrac_sys = list()
pq_sys = list()
for f in filelist:
	pq_samp = np.genfromtxt(f, delimiter=',')
	ceptr_samp, beptr_samp, heptr_samp, bfrac_samp = ept_refold(pq_samp, eptmat)
	bfrac_sys.append(bfrac_samp)
	pq_sys.append(pq_samp)

pf.plotbfrac_syssample(bfrac_ept, bfrac_sys, None, ui.fonll)
pf.plot_result_sysample(pq, pq_sys)

