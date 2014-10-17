import numpy as np
# Data points are from Table VIII of PHENIX PPG077
# http://link.aps.org/doi/10.1103/PhysRevC.84.044905

# length 29
eptbins = np.array([
0.3,
0.4,
0.5,
0.6,
0.7,
0.8,
0.9,
1,
1.2,
1.4,
1.6,
1.8,
2,
2.2,
2.4,
2.6,
2.8,
3,
3.2,
3.4,
3.6,
3.8,
4,
4.5,
5,
6,
7,
8,
9])

# The rest of these arrays are all length 28
eptx = np.array([
0.35,
0.45,
0.55,
0.65,
0.75,
0.85,
0.95,
1.1,
1.3,
1.5,
1.7,
1.9,
2.1,
2.3,
2.5,
2.7,
2.9,
3.1,
3.3,
3.5,
3.7,
3.9,
4.25,
4.75,
5.5,
6.5,
7.5,
8.5])

yinv_mb = np.array([
7.96e-2,
4.03e-2,
2.24e-2,
1.30e-2,
7.96e-3,
5.33e-3,
3.47e-3,
1.95e-3,
9.51e-4,
4.45e-4,
1.98e-4,
1.00e-4,
5.21e-5,
2.93e-5,
1.66e-5,
9.63e-6,
5.79e-6,
3.61e-6,
2.25e-6,
1.47e-6,
1.13e-6,
6.24e-7,
3.24e-7,
1.16e-7,
2.87e-8,
8.40e-9,
5.04e-9,
1.66e-9])

statlo_mb = np.array([
2.51e-3,
1.09e-3,
5.66e-4,
3.44e-4,
2.12e-4,
1.42e-4,
1.00e-4,
4.56e-5,
1.14e-5,
5.08e-6,
1.77e-6,
1.00e-6,
6.16e-7,
4.08e-7,
2.78e-7,
1.97e-7,
1.44e-7,
1.08e-7,
8.23e-8,
6.42e-8,
5.28e-8,
4.01e-8,
1.75e-8,
1.05e-8,
4.62e-9,
2.42e-9,
2.07e-9,
1.29e-9])

stathi_mb = np.array([
2.47e-3,
1.08e-3,
5.59e-4,
3.41e-4,
2.10e-4,
1.41e-4,
9.93e-5,
4.43e-5,
1.12e-5,
4.96e-6,
1.77e-6,
1.00e-6,
6.16e-7,
4.08e-7,
2.78e-7,
1.97e-7,
1.43e-7,
1.08e-7,
8.14e-8,
6.33e-8,
5.20e-8,
3.93e-8,
1.72e-8,
1.03e-8,
4.42e-9,
2.07e-9,
1.29e-9,
6.17e-10])

syslo_mb = np.array([
2.02e-2,
8.03e-3,
3.81e-3,
2.03e-3,
1.09e-3,
6.38e-4,
3.90e-4,
2.01e-4,
1.08e-4,
5.17e-5,
2.99e-5,
1.45e-5,
7.42e-6,
4.03e-6,
2.25e-6,
1.30e-6,
7.73e-7,
4.77e-7,
3.00e-7,
1.95e-7,
1.4e-7,
8.64e-8,
4.61e-8,
2.09e-8,
9.98e-9,
3.92e-9,
1.78e-9,
8.22e-10])

syshi_mb = np.array([
2.03e-2,
8.13e-3,
3.86e-3,
2.05e-3,
1.10e-3,
6.44e-4,
3.93e-4,
2.07e-4,
1.11e-4,
5.29e-5,
2.98e-5,
1.46e-5,
7.42e-6,
4.03e-6,
2.25e-6,
1.30e-6,
7.73e-7,
4.76e-7,
3.00e-7,
1.95e-7,
1.40e-7,
8.54e-8,
4.48e-8,
1.80e-8,
5.89e-9,
1.69e-9,
7.60e-10,
3.67e-10])

xsec_pp = np.array([
0.0135962,
0.00604719,
0.00335997,
0.00180748,
0.00130069,
0.000749943,
0.000560876,
0.000318049,
0.000125812,
6.57837e-05,
3.82693e-05,
1.89306e-05,
1.04344e-05,
6.08104e-06,
3.4231e-06,
2.06418e-06,
1.40079e-06,
8.64167e-07,
5.90605e-07,
4.1125e-07,
2.82597e-07,
1.91474e-07,
1.07843e-07,
4.83793e-08,
1.5932e-08,
5.30132e-09,
1.27112e-09,
8.18851e-10])

stat_pp = np.array([
0.0044487,
0.00170029,
0.000756314,
0.000398009,
0.000227974,
0.00013467,
9.14154e-05,
3.77206e-05,
2.1017e-05,
1.24581e-05,
2.06567e-06,
1.17996e-06,
7.43731e-07,
4.96175e-07,
3.62979e-07,
6.49475e-08,
4.77352e-08,
3.53262e-08,
2.72772e-08,
2.14723e-08,
1.70031e-08,
1.35628e-08,
5.94956e-09,
3.73171e-09,
1.78946e-09,
9.25843e-10,
5.7317e-10,
5.16379e-10])

syst_pp = np.array([
0.00595164,
0.00239096,
0.00104295,
0.000511658,
0.000259141,
0.000134524,
7.84702e-05,
3.64048e-05,
1.49814e-05,
6.55504e-06,
3.33202e-06,
1.64895e-06,
8.95727e-07,
5.09253e-07,
3.05295e-07,
2.8018e-07,
1.77134e-07,
1.1035e-07,
7.31717e-08,
4.90272e-08,
3.3108e-08,
2.25572e-08,
1.2463e-08,
5.52552e-09,
1.92929e-09,
6.03104e-10,
1.77741e-10,
8.85088e-11])