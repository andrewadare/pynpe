import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import unfold_input as ui

dtype = 'MC'
bfrac = 0.0073
dcamat = [ui.dcamatrix(i) for i in range(6)]
dca = [ui.dcadata_sim(i, bfrac) for i in range(6)]

# subdca is a list of 6 arrays, each shape (70,2)
subdca, subdcamat = ui.dca_subset(dca, dcamat, dtype)

# list of 6 arrays, each shape (500,70)
preds = [np.loadtxt('csv/preds{}.csv'.format(i),delimiter=',') for i in range(6)]

class DcaAnimation(animation.TimedAnimation):
    def __init__(self, data, pred):
        self.t = np.arange(pred[0].shape[0])
        self.x = np.arange(pred[0].shape[1])
        self.data = data
        self.pred = pred
        self.datalines = [Line2D([],[], color='black', lw=2) for i in range(6)]
        self.predlines = [Line2D([],[], color='red',   lw=2) for i in range(6)]
        nr, nc = 2, 3
        fig, axes = plt.subplots(nr, nc)
        for row in range(nr):
            for col in range(nc):
                i = nc * row + col
                d = self.data[i][:,0]
                a = axes[row, col]
                a.set_yscale('log')
                a.set_xlim([0, 70])
                a.set_ylim([1, 2*np.max(d)])
                a.tick_params(axis='x', top='off', labelsize=6)
                a.tick_params(axis='y', labelsize=6)
                s = r'{0:.1f}-{1:.1f} GeV/c'.format(
                    ui.dcaeptbins[i], ui.dcaeptbins[i + 1])
                a.text(0.55, 0.9, s, fontsize=8, transform=a.transAxes)
                a.add_line(self.datalines[i])
                a.add_line(self.predlines[i])
        animation.TimedAnimation.__init__(self, fig, interval=50, blit=True)

    def _draw_frame(self, framedata):
        i = framedata
        self._drawn_artists = []
        for k in range(6):
            self.datalines[k].set_data(self.x, self.data[k][:,0]) # Move to init
            self.predlines[k].set_data(self.x, self.pred[k][i,:])
            self._drawn_artists.append(self.datalines[k])
            self._drawn_artists.append(self.predlines[k])

    def new_frame_seq(self):
        return iter(range(self.t.size))

    def _init_draw(self):
        for k in range(6):
            self.datalines[k].set_data([],[])
            self.predlines[k].set_data([],[])

ani = DcaAnimation(subdca, preds)
#ani.save('test_sub.mp4')
plt.show()
