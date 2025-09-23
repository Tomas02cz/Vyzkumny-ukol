from functions import *
from input_parameters import *
from scan_database import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import cumulative_trapezoid




t_start, t_end = get_plasma_time_parameters(47768)
print(t_start)
print(t_end)

# shot parameters
shot_no_vacuum = 49074
shot_no = 49077 #47768 #47768 #48531 44800

# load data
t_start, t_end = get_plasma_time_parameters(shot_no)
raw_data = get_rogowski_data(shot_no)
raw_data_vacuum = get_rogowski_data(shot_no_vacuum)

save=0
show=1
        
# fit for off-set subtruction
n = 0 # degree of fit polynomial

# computing fit coefficients and the offset polynomial
coeffs1 = np.polyfit(raw_data['t'].values, raw_data['rogowski1'].values, n)
coeffs2 = np.polyfit(raw_data['t'].values, raw_data['rogowski2'].values, n)
trend1 = np.polyval(coeffs1, raw_data['t'].values)
trend2 = np.polyval(coeffs2, raw_data['t'].values)

coeffs1_vacuum = np.polyfit(raw_data_vacuum['t'].values, raw_data_vacuum['rogowski1'].values, n)
coeffs2_vacuum = np.polyfit(raw_data_vacuum['t'].values, raw_data_vacuum['rogowski2'].values, n)
trend1_vacuum = np.polyval(coeffs1_vacuum, raw_data_vacuum['t'].values)
trend2_vacuum = np.polyval(coeffs2_vacuum, raw_data_vacuum['t'].values)

# substruction of offset from raw signal
raw_data_detrended = raw_data.copy()
raw_data_detrended['rogowski1'] = raw_data['rogowski1'] - trend1
raw_data_detrended['rogowski2'] = raw_data['rogowski2'] - trend2

raw_data_detrended_vacuum = raw_data.copy()
raw_data_detrended_vacuum['rogowski1'] = raw_data_vacuum['rogowski1'] - trend1_vacuum
raw_data_detrended_vacuum['rogowski2'] = raw_data_vacuum['rogowski2'] - trend2_vacuum
    
# vacuum signal subtruction
raw_data_subtructed = raw_data_detrended.copy()
raw_data_subtructed['rogowski1'] -= raw_data_detrended_vacuum['rogowski1']
raw_data_subtructed['rogowski2'] -= raw_data_detrended_vacuum['rogowski2']

plt.figure(figsize=(10, 6))
plt.plot(raw_data_subtructed['t'], raw_data_subtructed['rogowski1'], label='corrected inner Rogowski coil')
plt.plot(raw_data_detrended['t'], raw_data_detrended['rogowski1'], label='inner without off-set')
# plt.plot(raw_data['t'], raw_data['rogowski2'], label='outer Rogowski coil')
# plt.plot(raw_data_detrended['t'], raw_data_detrended['rogowski2'], label='outer without off-set')
# show duration of plasma
plt.axvline(x=t_start, color='black', linestyle='--', linewidth=1) 
plt.axvline(x=t_end, color='black', linestyle='--', linewidth=1)
plt.xlabel('time [ms]')
plt.ylabel('voltage [V]')
plt.title(f'Raw signal from inner Rogowski coil during shot #{shot_no}')
plt.legend()
plt.grid(True)
if save == 1:
    plt.savefig(f'results/{shot_no}_RogowskiCoil_FittedRawSignal.png')
if show == 1:
    plt.show()



# integration of raw data
integral_U1 = -cumulative_trapezoid(raw_data_detrended['rogowski1'], raw_data_detrended['t'], initial=0)
integral_U2 = -cumulative_trapezoid(raw_data_detrended['rogowski2'], raw_data_detrended['t'], initial=0)

integral_S1 = -cumulative_trapezoid(raw_data_subtructed['rogowski1'], raw_data_subtructed['t'], initial=0)
integral_S2 = -cumulative_trapezoid(raw_data_subtructed['rogowski2'], raw_data_subtructed['t'], initial=0)

# make new pd.DataFrame of integrated data
integrated_data = pd.DataFrame({'t': raw_data_detrended['t'], 'rogowski1': integral_U1, 'rogowski2': integral_U2})

integrated_data_corrected = pd.DataFrame({'t': raw_data_subtructed['t'], 'rogowski1': integral_S1, 'rogowski2': integral_S2})




# graph of integrated data
plt.figure(figsize=(10, 6))
plt.plot(integrated_data['t'], integrated_data['rogowski1'], label='inner Rogowski coil')
#plt.plot(integrated_data['t'], integrated_data['rogowski2'], label='outer Rogowski coil')
plt.plot(integrated_data_corrected['t'], integrated_data_corrected['rogowski1'], label='corrected inner Rogowski coil')
#plt.plot(integrated_data_corrected['t'], integrated_data_corrected['rogowski2'], label='corrected outer Rogowski coil')

# show plasma duration
plt.axvline(x=t_start, color='black', linestyle='--', linewidth=1)
plt.axvline(x=t_end, color='black', linestyle='--', linewidth=1)

plt.xlabel('time [ms]')
plt.ylabel(f'$I_p$ [kA]')
plt.title(f'Corrected integrated signal from Rogowski coils during shot #{shot_no}')
plt.legend()
plt.grid(True)

plt.savefig(f'results/{shot_no}_RogowskiCoil_IntegratedSignal.png')

plt.show()    