import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from functions import get_plasma_time_parameters, get_U_loop_data

from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression



def weighted_moving_average(x, N):
    w = np.hanning(N)
    #w = sp.signal.windows.boxcar(N)
    w = w / w.sum()
    return np.convolve(x, w, mode='same')



saveFig = 1
fileFormat = 'png'

shot_no = 50117
filename = f'data/TektrMSO64_ALL.csv'

# load data
t_start, t_end = get_plasma_time_parameters(shot_no=shot_no)
U_loop = get_U_loop_data(shot_no=shot_no)
MHD_data = pd.read_csv('data/MHDring_rawData.csv', skiprows=1, sep=',', decimal='.')
MHD_data.columns = ['t', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'diamagnet1', 'diamagnet2', 'rogowski', 'tor1', 'tor2']
raw_data = pd.read_csv(filename, skiprows=12, sep=',', decimal='.', names=['t', 'dia in', 'dia out', 'rog in', 'U HFS'])

# polarity inversion
raw_data['rog in'] = - raw_data['rog in']

# offsettrend removal
trend = np.mean(MHD_data['rogowski'])
MHD_data['rogowski'] = MHD_data['rogowski'] - trend

# trend = np.mean(raw_data['rog in'])
# raw_data['rog in'] = raw_data['rog in'] - trend

# smoothing
N = 50
raw_data['rog in'] = weighted_moving_average(raw_data['rog in'], N)
MHD_data['rogowski'] = weighted_moving_average(MHD_data['rogowski'], N)

# plot raw signal
c1 = 11
plt.figure(figsize=(10,6))
# plt.plot(raw_data['t'], raw_data['dia in'], label=f'inner diamagnetic loop')
# plt.plot(raw_data['t'], raw_data['dia out'], label=f'outer diamagnetic loop')
plt.plot(raw_data['t'], raw_data['rog in'], label=f'inner Rogowski coil')
# plt.plot(raw_data['t'], raw_data['U HFS'], label=f'HFS Mirnov coil')
plt.plot(MHD_data['t'], MHD_data['rogowski'], label=f'hardware noise')
plt.plot(MHD_data['t'], MHD_data['rogowski']*c1, label=f'{c1}*hardware noise')

plt.xlabel('t (s)')
plt.ylabel('U (V)')
plt.title(f'raw signal #{shot_no}')
plt.legend()
plt.grid(True)
if saveFig == 1:
    plt.savefig(f'results/RogCoilTest_RawSignal.{fileFormat}', dpi=300)
plt.show()

# # noise offset
# u_interp = np.interp(raw_data['t'].values, MHD_data['t'].values, MHD_data['rogowski'].values)
# K0 = abs(np.min(raw_data['rog in']))/abs(np.min(u_interp))
# K0=11
# K0=0
# raw_data['rog in'] = raw_data['rog in'] - K0*u_interp

# # plot raw signal
# plt.figure(figsize=(10,6))
# plt.plot(raw_data['t'], raw_data['dia in'], label=f'inner diamagnetic loop')
# plt.plot(raw_data['t'], raw_data['dia out'], label=f'outer diamagnetic loop')
# plt.plot(raw_data['t'], raw_data['rog in'], label=f'inner Rogowski coil')
# plt.plot(raw_data['t'], raw_data['rog in'] - 11*u_interp, label=f'diff')
# plt.plot(raw_data['t'], raw_data['U HFS'], label=f'HFS Mirnov coil')
# plt.plot(MHD_data['t'], MHD_data['rogowski']*10, label='hardware noise')

# plt.xlabel('t (s)')
# plt.ylabel('U (V)')
# plt.legend()
# plt.grid(True)
# plt.show()

# calibration constant
#K = 15.15e6
K_cal = 2*1e4
# integration of Rogowski coil signal
u_interp1 = np.interp(raw_data['t'].values, MHD_data['t'].values, MHD_data['rogowski'].values)
u_interp2 = np.interp(raw_data['t'].values, U_loop['t'].values, U_loop['U_loop'].values)

integral_1 = K_cal*cumulative_trapezoid(raw_data['rog in'], raw_data['t'], initial=0)
integral_2 = K_cal*cumulative_trapezoid(u_interp1, raw_data['t'], initial=0)

int_data = pd.DataFrame({'t': raw_data['t'], 'rog': integral_1, 'noise': integral_2, 'U_loop': u_interp2})

# U_loop offset
mask1 = int_data['t'] <= 4*1e-3
K1 = np.max(int_data['rog'][mask1])/np.max(int_data['U_loop'][mask1])
# harware noise
mask2 = int_data['t'] >= 11.5*1e-3
K2 = np.max(int_data.loc[mask2,'rog'])/np.max(int_data['noise'][mask2])
# U_loop offset and hardware noise
sig = int_data['rog']-K1*u_interp1
mask3 = int_data['t'] >= 11.5*1e-3
K3 = abs(np.max(sig[mask3])/np.max(int_data['noise'][mask3]))

# linear regression
mask = (int_data['t'] >= 11.5*1e-3) | (int_data['t'] <= 4*1e-3)
y = int_data['rog'][mask]
X = np.column_stack([int_data['U_loop'][mask], int_data['noise'][mask]])  # (N,2)

model = LinearRegression(fit_intercept=True)
model.fit(X, y)

k1, k2 = model.coef_
k3 = model.intercept_
r2 = model.score(X, y)   # R^2

denoisedSignal = int_data['rog'] - k1*int_data['U_loop'] - k2*int_data['noise'] - k3

print(f"k1={k1}, k2={k2}, k3={k3}, R2={r2:.4f}")
offsetFree = pd.DataFrame({'t': int_data['t'], 'denoised': denoisedSignal})

# predikce
y_pred = model.predict(X)
residuals = y - y_pred

# offsets removal
detrended_data = pd.DataFrame({'t': int_data['t'], 'U_loop free': int_data['rog']-K1*int_data['U_loop'], 'noise free': int_data['rog']-K2*int_data['noise'], 'both offsets free': sig - K3*int_data['noise']})



# plot integrated data
c2 = 100

plt.figure(figsize=(10,6))
plt.plot(int_data['t'], int_data['rog'], label='inner Rogowski')
plt.plot(int_data['t'], int_data['U_loop'], label='U_loop')
plt.plot(int_data['t'], int_data['noise']*K2, label='hardware noise')

plt.plot(detrended_data['t'], detrended_data['U_loop free'], label=r'$U_{l}$ free')
# plt.plot(detrended_data['t'], detrended_data['noise free'], label='HN free')
# plt.plot(detrended_data['t'], K3*int_data['noise'], label='offset')
# plt.plot(detrended_data['t'], detrended_data['both offsets free'], label='offsets free')

plt.plot(offsetFree['t'], offsetFree['denoised'], label='denoised')
# plt.plot(offsetFree['t'], int_data['rog'] - offsetFree['denoised'], label='lin. reg.')

# plt.plot(MHD_data['t'], MHD_data['rogowski']*c2, label=f'{c2}*raw hardware noise')

plt.axvline(x=t_start, color='black', linestyle='--', linewidth=1) # show duration of plasma
plt.axvline(x=t_end, color='black', linestyle='--', linewidth=1)

plt.xlabel('t (s)')
plt.ylabel('I (a.u.)')
plt.title(f'integrated signal #{shot_no}')
plt.legend()
plt.grid(True)
if saveFig == 1:
    plt.savefig(f'results/RogCoilTest_IntSignal.{fileFormat}', dpi=300)
plt.show()