from functions import *

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.integrate import cumulative_trapezoid



shot_no = 47768

# load data
t_start, t_end = get_plasma_time_parameters(shot_no)

raw_data = get_Bt_data(shot_no)

Bt = get_integrated_Bt_data(shot_no)

# graph of Bt from plasma discharge
plt.figure(figsize=(10, 6))
plt.plot(Bt['t'], Bt['Bt'], label='Bt')

plt.axvline(x=t_start, color='black', linestyle='--', linewidth=1) # show duration of plasma
plt.axvline(x=t_end, color='black', linestyle='--', linewidth=1)

plt.xlabel('time [ms]')
plt.ylabel('Bt [T]')
plt.title(f'Toroidal magnetic field during shot #{shot_no}')
plt.legend()
plt.grid(True)

# graph of raw data from plasma discharge
plt.figure(figsize=(10, 6))
plt.plot(raw_data['t'], raw_data['Bt'], label='Bt')

plt.axvline(x=t_start, color='black', linestyle='--', linewidth=1) # show duration of plasma
plt.axvline(x=t_end, color='black', linestyle='--', linewidth=1)

plt.xlabel('time [ms]')
plt.ylabel('voltage [V]')
plt.title(f'Raw signal from TF coil during shot #{shot_no}')
plt.legend()
plt.grid(True)
#plt.savefig(f'results/{shot_no}_TFCoil_RawSignal.png')

plt.show()


# integration of raw data
K_BtCoil = 70.42 # calibration constant T/(Vs)
# raw_data['t'] /= 1000 #TODO
integral_U1 = cumulative_trapezoid(raw_data['Bt'], raw_data['t'], initial=0) * K_BtCoil

# new DataFrame of integrated data
integrated_data = pd.DataFrame({'t': raw_data['t'], 'Bt': integral_U1})



# graph of integrated data
plt.figure(figsize=(10, 6))
plt.plot(integrated_data['t'], integrated_data['Bt'], label='toroidal magnetic field')

# plt.axvline(x=t_start, color='black', linestyle='--', linewidth=1)
# plt.axvline(x=t_end, color='black', linestyle='--', linewidth=1)

plt.xlabel('time [ms]')
plt.ylabel(f'$B_t$ [T]')
plt.title(f'Integrated signal from TF coils during shot #{shot_no}')
plt.legend()
plt.grid(True)