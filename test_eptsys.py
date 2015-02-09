
import unfold_input as ui
import plotting_functions as pf

ept_mb = ui.eptdata('AuAu200MB',False)
print(ept_mb[0,0])

ept_sys = list()
for i in range(0,1000):
    ept_sys.append(ui.eptdata('AuAu200MB',True,'Stat'))
    # print(ept_sys[i][0,0])

pf.plot_ept_syssample(ept_mb, ept_sys)
