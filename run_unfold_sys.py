import unfold as uf
import sys

#----------------------------------------------------
# Get the command line arguments
#----------------------------------------------------
# rangelo = int(sys.argv[1])
# rangehi = int(sys.argv[2])


#----------------------------------------------------
# Run the unfold over the Run 11 AuAu 200 Final Data
#----------------------------------------------------

for step in range(0, 3):
    print("\n Running Step {} \n".format(step))
    uf.unfold(step=step, alpha=0.2, dcaweight=0.5,
              dtype='AuAu200MB',
              outdir='output/Run11AuAu200MB_centweight', 
              dca_filename='rootfiles/data_jan22_centweight/run11DCA_centweight.root',
              ept_err='Stat', rand_ept=False)

#----------------------------------------------------
# Run the unfold sampling the ept systematics
#----------------------------------------------------

# for samp in range(rangelo, rangehi):
#     for step in range(2):
#         print("\n Running Step {} for Sample {} \n".format(step, samp))
#         uf.unfold(step=step, dtype='AuAu200MB',
#                   outdir='output/Run11AuAu200MB_randept/samp{}'.format(samp), 
#                   ept_err='Stat', rand_ept=True)

# uf.unfold(step=0, dtype='AuAu200MB', outdir='Run11AuAu200MB_randept', rand_ept=True)

#----------------------------------------------------
# Run the unfold sampling the DCA systematics
#----------------------------------------------------

# for samp in range(rangelo, rangehi):
#     for step in range(2):
#         print("\n Running Step {} for Sample {} \n".format(step, samp))
#         uf.unfold(step=step, dtype='AuAu200MB',
#                   outdir='output/Run11AuAu200MB_randdca/samp{}'.format(samp), 
#                   dca_filename='rootfiles/dcasys/run11DCA_syssample_{}.root'.format(samp),
#                   ept_err='Stat', rand_ept=False)


#----------------------------------------------------
# Run the unfold sampling the DCA & ept systematics
#----------------------------------------------------

# for samp in range(rangelo, rangehi):
# 	for esys in range(0, 5):
# 	    for step in range(3):
# 	        print("\n Running Step {} for DCA sys {} ept sys {} \n".format(step, samp, esys))
# 	        uf.unfold(step=step, dtype='AuAu200MB',
# 	                  outdir='output/Run11AuAu200MB_randdca_randept/samp{}_{}'.format(samp, esys), 
# 	                  dca_filename='rootfiles/dcasys/run11DCA_syssample_{}.root'.format(samp),
# 	                  ept_err='Stat', rand_ept=True)
