import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LogNorm
from scipy.stats import norm

# Draw ept
x = np.linspace(0,10,num=100)

def draa(x):
  return 1.3*np.sqrt(2*np.pi*1.1*1.1)*norm.pdf(x, loc=1.5, scale=1.1) + \
  0.2/(1 + np.exp(-x+3))

def braa(x):
  return 0.6*np.exp(-x/3) + \
  1.1*np.sqrt(2*np.pi*1.5*1.5)*norm.pdf(x, loc=3.4, scale=1.5) + \
  0.3/(1 + np.exp(-x+7))

fig, ax = plt.subplots()
ax.set_xlabel(r'$p_T$ [GeV/c]')
ax.set_ylabel(r'$R_{AA}$')
ax.plot(x, draa(x), lw=4, color='r', alpha=0.6, label=r'charmed hadrons')
ax.plot(x, braa(x), lw=4, color='b', alpha=0.6, label=r'beauty hadrons')
ax.legend()
fig.savefig('pdfs/db-raa.pdf')
