from functions import *
from rogowski_coil import integrated_data as I_all # integrated signal from Rogowski coils
from TF_coil import integrated_data as Bt # integrated signal from TF coil

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import scipy.stats as stats

from scipy.integrate import cumulative_trapezoid




''' FUNCTION'S DEFINITIONS '''

def plot_raw_data(shot_no_vacuum, shot_no, raw_data_vacuum, raw_data, save):
    # graph of raw data from vacuum shot
    plt.figure(figsize=(10, 6))
    plt.plot(raw_data_vacuum['t'], raw_data_vacuum['diamagnet1'], label='inner diamagnetic loop')
    plt.plot(raw_data_vacuum['t'], raw_data_vacuum['diamagnet2'], label='outer diamagnetic loop')
    
    plt.xlabel('time [ms]')
    plt.ylabel('voltage [V]')
    plt.title(f'Raw signal from diamagnetic loops during vacuum shot #{shot_no_vacuum}')
    plt.legend()
    plt.grid(True)
    if save == 1:
        plt.savefig(f'results/{shot_no_vacuum}_DiamagneticLoop_RawSignal_VacuumShot.png')
    
    # graph of raw data from plasma discharge
    plt.figure(figsize=(10, 6))
    plt.plot(raw_data['t'], raw_data['diamagnet1'], label='inner diamagnetic loop')
    plt.plot(raw_data['t'], raw_data['diamagnet2'], label='outer diamagnetic loop')
    
    plt.axvline(x=t_start, color='black', linestyle='--', linewidth=1) # show duration of plasma
    plt.axvline(x=t_end, color='black', linestyle='--', linewidth=1)
    
    plt.xlabel('time [ms]')
    plt.ylabel('voltage [V]')
    plt.title(f'Raw signal from diamagnetic loops during plasma discharge #{shot_no}')
    plt.legend()
    plt.grid(True)
    if save == 1:
        plt.savefig(f'results/{shot_no}_DiamagneticLoop_RawSignal_PlasmaShot.png')
    
    plt.show()







# shot parameters
# shot_no_vacuum = 47769
# shot_no = 47768

shot_no_vacuum = 49092
shot_no = 49093

#diamagnet_diagnostics(shot_no_vacuum, shot_no)

#def diamagnet_diagnostics(shot_no_vacuum, shot_no):

# diamagnetic loops parameters
D_1 = 17.5 # diameter of inner diamagnetic loop in cm
D_2 = 20.4 # diameter of outer diamagnetic loop in cm

# effective surface of loop
A_1 = np.pi * D_1**2 / 4
A_2 = np.pi * D_2**2 / 4

# load data
t_start, t_end = get_plasma_time_parameters(shot_no)
raw_data_vacuum = get_diamagnet_data(shot_no_vacuum)
raw_data = get_diamagnet_data(shot_no)

# plot raw signal
plot_raw_data(shot_no_vacuum, shot_no, raw_data_vacuum, raw_data, save=0)

# plot raw data
#plot_raw_data(shot_no_vacuum, shot_no, raw_data_vacuum, raw_data)
# graph of raw data from vacuum shot
plt.figure(figsize=(10, 6))
plt.plot(raw_data_vacuum['t'], raw_data_vacuum['diamagnet1'], label='inner diamagnetic loop')
plt.plot(raw_data_vacuum['t'], raw_data_vacuum['diamagnet2'], label='outer diamagnetic loop')

plt.xlabel('time [ms]')
plt.ylabel('voltage [V]')
plt.title(f'Raw signal from diamagnetic loops during vacuum shot #{shot_no_vacuum}')
plt.legend()
plt.grid(True)
plt.savefig(f'results/{shot_no_vacuum}_DiamagneticLoop_RawSignal_VacuumShot.png')

# graph of raw data from plasma discharge
plt.figure(figsize=(10, 6))
plt.plot(raw_data['t'], raw_data['diamagnet1'], label='inner diamagnetic loop')
plt.plot(raw_data['t'], raw_data['diamagnet2'], label='outer diamagnetic loop')

plt.axvline(x=t_start, color='black', linestyle='--', linewidth=1) # show duration of plasma
plt.axvline(x=t_end, color='black', linestyle='--', linewidth=1)

plt.xlabel('time [ms]')
plt.ylabel('voltage [V]')
plt.title(f'Raw signal from diamagnetic loops during plasma discharge #{shot_no}')
plt.legend()
plt.grid(True)
plt.savefig(f'results/{shot_no}_DiamagneticLoop_RawSignal_PlasmaShot.png')

plt.show()






# kalibration of diamagnetic loops

# calibration: U1 = m * U2
mask = (raw_data['t'] >= t_start) & (raw_data['t'] <= t_end)
x = raw_data_vacuum['diamagnet2'][mask].values # columns as numpy array
y = raw_data_vacuum['diamagnet1'][mask].values
m = np.dot(x, y) / np.dot(x, x) # computing optimal m
print(f"kalibration number m: {m}")
# computin difference raw signal
raw_data_vacuum['difference'] = raw_data_vacuum['diamagnet1'] - m * raw_data_vacuum['diamagnet2']
raw_data['difference'] = raw_data['diamagnet1'] - m * raw_data['diamagnet2']



# calibration: U1 = k * U2 + q ; U1 = raw_data_vacuum['diamagnet1'], U2 = raw_data_vacuum['diamagnet2']
# in scipy
slope, intercept, r_value, p_value, std_err = stats.linregress(raw_data_vacuum['diamagnet2'], raw_data_vacuum['diamagnet1'])

# in numpy
#k, q = np.polyfit(raw_data_vacuum['diamagnet2'], raw_data_vacuum['diamagnet1'], deg=1)

# calibrated data
raw_data_vacuum_calibrated = raw_data_vacuum.copy()
raw_data_calibrated = raw_data.copy()
raw_data_vacuum_calibrated['diamagnet2'] = slope * raw_data_vacuum['diamagnet2'] + intercept
raw_data_calibrated['diamagnet2'] = slope * raw_data['diamagnet2'] + intercept
#raw_data_vacuum['diamagnet2_calibrated_numpy'] = k * raw_data_vacuum['diamagnet2'] + q


plt.plot(raw_data_vacuum['t'], raw_data_vacuum['diamagnet1'], label='vacuum shot (reference)')
plt.plot(raw_data_vacuum_calibrated['t'], raw_data_vacuum_calibrated['diamagnet2'], label='vacuum shot (calibrated)', linestyle='--')
plt.plot(raw_data['t'], raw_data['diamagnet1'], label='plasma discharge (reference)')
plt.plot(raw_data_calibrated['t'], raw_data_calibrated['diamagnet2'], label='plasma discharge (calibrated)', linestyle='--')
plt.xlabel('Time [s]')
plt.ylabel('Induced voltage [V]')
plt.legend()
plt.title('Calibration of Diamagnet Diagnostics')
plt.grid(True)
plt.show()






# plot difference signal for vacuum shot
plt.figure(figsize=(10, 6))
plt.plot(raw_data_vacuum['t'], raw_data_vacuum['diamagnet1'], label='inner diamagnetic loop')
plt.plot(raw_data_vacuum['t'], raw_data_vacuum['diamagnet2'], label='outer diamagnetic loop')
plt.plot(raw_data_vacuum['t'], raw_data_vacuum['difference'], label='difference between loops')

plt.xlabel('time [ms]')
plt.ylabel('voltage [V]')
plt.title(f'Raw signal from diamagnetic loops during vacuum shot #{shot_no_vacuum}')
plt.legend()
plt.grid(True)

# plot difference signal for plasma discharge
plt.figure(figsize=(10, 6))
plt.plot(raw_data['t'], raw_data['diamagnet1'], label='inner diamagnetic loop')
plt.plot(raw_data['t'], raw_data['diamagnet2'], label='outer diamagnetic loop')
plt.plot(raw_data['t'], raw_data['difference'], label='difference between loops')

plt.axvline(x=t_start, color='black', linestyle='--', linewidth=1)
plt.axvline(x=t_end, color='black', linestyle='--', linewidth=1)

plt.xlabel('time [ms]')
plt.ylabel('voltage [V]')
plt.title(f'Raw signal from diamagnetic loops during plasma discharge #{shot_no}')
plt.legend()
plt.grid(True)





# integration of raw data
integral_U1_vacuum = cumulative_trapezoid(raw_data_vacuum['diamagnet1'], raw_data_vacuum['t'], initial=0)
integral_U2_vacuum = cumulative_trapezoid(raw_data_vacuum['diamagnet2'], raw_data_vacuum['t'], initial=0)

integral_U1 = cumulative_trapezoid(raw_data['diamagnet1'], raw_data['t'], initial=0)
integral_U2 = cumulative_trapezoid(raw_data['diamagnet2'], raw_data['t'], initial=0)

integral_difference_vacuum = cumulative_trapezoid(raw_data_vacuum['difference'], raw_data_vacuum['t'], initial=0) # this should be constant at 0
integral_difference = cumulative_trapezoid(raw_data['difference'], raw_data['t'], initial=0)

# new DataFrame of integrated data
integrated_data_vacuum = pd.DataFrame({'t': raw_data_vacuum['t'], 'diamagnet1': integral_U1_vacuum, 'diamagnet2': integral_U2_vacuum, 'difference': integral_difference_vacuum})
integrated_data = pd.DataFrame({'t': raw_data['t'], 'diamagnet1': integral_U1, 'diamagnet2': integral_U2, 'difference': integral_difference})




#def plot_integrated_data(integrated_data, integrated_data_vacuum, calibration_const):
# graph of integrated data
plt.figure(figsize=(10, 6))
plt.plot(integrated_data_vacuum['t'], integrated_data_vacuum['diamagnet1'], label='inner diamagnetic loop')
plt.plot(integrated_data_vacuum['t'], integrated_data_vacuum['diamagnet2'], label='outer diamagnetic loop')
plt.plot(integrated_data_vacuum['t'], integrated_data_vacuum['difference'], label='difference') # this should be constant at 0

plt.xlabel('time [ms]')
plt.ylabel('flux [Wb]')
plt.title(f'Integrated signal from diamagnetic loops during vacuum shot #{shot_no_vacuum}')
plt.legend()
plt.grid(True)


plt.figure(figsize=(10, 6))
plt.plot(integrated_data['t'], integrated_data['diamagnet1'], label='inner diamagnetic loop')
plt.plot(integrated_data['t'], integrated_data['diamagnet2'], label='outer diamagnetic loop')
plt.plot(integrated_data['t'], integrated_data['difference'], label='difference')

plt.axvline(x=t_start, color='black', linestyle='--', linewidth=1)
plt.axvline(x=t_end, color='black', linestyle='--', linewidth=1)

plt.xlabel('time [ms]')
plt.ylabel('flux [Wb]')
plt.title(f'Integrated signal from diamagnetic loops during plasma discharge #{shot_no}')
plt.legend()
plt.grid(True)

# plot only during plasma discharge
mask = (raw_data['t'] >= t_start) & (raw_data['t'] <= t_end)

plt.figure(figsize=(10, 6))
plt.plot(integrated_data_vacuum['t'][mask], integrated_data_vacuum['diamagnet1'][mask], label='inner diamagnetic loop')
plt.plot(integrated_data_vacuum['t'][mask], integrated_data_vacuum['diamagnet2'][mask], label='outer diamagnetic loop')
plt.plot(integrated_data_vacuum['t'][mask], integrated_data_vacuum['difference'][mask], label='difference')
plt.plot(integrated_data_vacuum['t'][mask], m * integrated_data_vacuum['diamagnet2'][mask], label='calibrated outer diamagnetic loop')

plt.axvline(x=t_start, color='black', linestyle='--', linewidth=1)
plt.axvline(x=t_end, color='black', linestyle='--', linewidth=1)

plt.xlabel('time [ms]')
plt.ylabel('flux [Wb]')
plt.title(f'Integrated signal from diamagnetic loops during vacuum shot #{shot_no}')
plt.legend()
plt.grid(True)


plt.figure(figsize=(10, 6))
plt.plot(integrated_data['t'][mask], integrated_data['diamagnet1'][mask], label='inner diamagnetic loop')
plt.plot(integrated_data['t'][mask], integrated_data['diamagnet2'][mask], label='outer diamagnetic loop')
plt.plot(integrated_data['t'][mask], integrated_data['difference'][mask], label='difference')

plt.axvline(x=t_start, color='black', linestyle='--', linewidth=1)
plt.axvline(x=t_end, color='black', linestyle='--', linewidth=1)

plt.xlabel('time [ms]')
plt.ylabel('flux [Wb]')
plt.title(f'Integrated signal from diamagnetic loops only during plasma discharge #{shot_no}')
plt.legend()
plt.grid(True)

plt.show()



# calibrated
mask = (raw_data['t'] >= t_start) & (raw_data['t'] <= t_end)

plt.figure(figsize=(10, 6))
plt.plot(integrated_data['t'][mask], integrated_data['diamagnet1'][mask], label='inner diamagnetic loop')
plt.plot(integrated_data['t'][mask], m * integrated_data['diamagnet2'][mask], label='outer diamagnetic loop')
plt.plot(integrated_data['t'][mask], integrated_data['difference'][mask], label='difference')

plt.axvline(x=t_start, color='black', linestyle='--', linewidth=1)
plt.axvline(x=t_end, color='black', linestyle='--', linewidth=1)

plt.xlabel('time [ms]')
plt.ylabel('flux [Wb]')
plt.title(f'Integrated and calibrated signal from diamagnetic loops only during plasma discharge #{shot_no}')
plt.legend()
plt.grid(True)

plt.show()





# compute total magnetic flux
magnetic_flux = pd.DataFrame({'t': integrated_data['t'], 'uncorrected total flux difference': - 1 / m / (A_2 / A_1 - 1) * integrated_data['difference'], 'vacuum flux difference': - integrated_data_vacuum['difference'], 'total flux difference': - 1 / m / (A_2 / A_1 - 1) * integrated_data['difference'] + integrated_data_vacuum['difference']})




# graph of total magnetic flux difference
mask = (raw_data['t'] >= t_start) & (raw_data['t'] <= t_end)

plt.figure(figsize=(10, 6))
plt.plot(magnetic_flux['t'][mask], magnetic_flux['uncorrected total flux difference'][mask], label='total toroidal magnetic flux')
plt.plot(magnetic_flux['t'][mask], magnetic_flux['total flux difference'][mask], label='corrected total toroidal magnetic flux')
plt.plot(magnetic_flux['t'][mask], magnetic_flux['vacuum flux difference'][mask], label='vacuum magnetic flux difference')

plt.axvline(x=t_start, color='black', linestyle='--', linewidth=1)
plt.axvline(x=t_end, color='black', linestyle='--', linewidth=1)

plt.xlabel('time [ms]')
plt.ylabel('toroidal magnetic flux [Wb]')
plt.title(f'Toroidal magnetic flux during plasma discharge #{shot_no}')
plt.legend()
plt.grid(True)

plt.show()



# compute paramagnetic effect
I_p = pd.DataFrame({'t': I_all['t'], 'Ip': I_all['rogowski2']}) # TODO implement I_p computation from inner Rogowski coil

interpolated_Bt = np.interp(magnetic_flux['t'], Bt['t'], Bt['Bt']) # interpolate Bt data to match sampling time of diamagnetic loops 
B_0 = Bt.copy()
B_0['Bt'] = interpolated_Bt # TODO B_0 should be on axis (Is it right now?)

magnetic_flux['paramagnetic flux'] = (mu_0 * I_p['Ip'])**2 / (8 * np.pi * B_0['Bt'])

# compute diamagnetic flux
magnetic_flux['diamagnetic flux'] = magnetic_flux['total flux difference'] - magnetic_flux['paramagnetic flux']



# plot total toroidal magnetic flux, paramagnetic and diamagnetic toroidal magnetic flux
plt.figure(figsize=(10, 6))
plt.plot(magnetic_flux['t'], magnetic_flux['total flux difference'], label='total toroidal magnetic flux')
plt.plot(magnetic_flux['t'], magnetic_flux['paramagnetic flux'], label='paramagnetic effect contribution')
plt.plot(magnetic_flux['t'], magnetic_flux['diamagnetic flux'], label='diamagnetic effect contribution')

plt.axvline(x=t_start, color='black', linestyle='--', linewidth=1)
plt.axvline(x=t_end, color='black', linestyle='--', linewidth=1)

plt.xlabel('time [ms]')
plt.ylabel('toroidal magnetic flux [Wb]')
plt.title(f'Change in toroidal magnetic flux during plasma discharge #{shot_no}')
plt.legend()
plt.grid(True)

plt.show()



# compute perp. energy, thermal energy and energy confinement time
U_loop = get_U_loop_data(shot_no)

P_ohm = pd.DataFrame({'t': magnetic_flux['t'], 'P_ohm': I_p['Ip'] * U_loop['U_loop']})

W_perp = pd.DataFrame({'t': magnetic_flux['t'], 'W_perp': - B_0['Bt'] / mu_0 * magnetic_flux['diamagnetic flux']})
W_th = pd.DataFrame({'t': magnetic_flux['t'], 'W_th': 3 * W_perp['W_perp'] * np.pi * R})
tau_E = pd.DataFrame({'t': magnetic_flux['t'], 'tau_E': W_th['W_th'] / P_ohm['P_ohm']})


# plot perp. energy
plt.figure(figsize=(10, 6))
plt.plot(W_perp['t'], W_perp['W_perp'], label='perpendicular energy')

plt.axvline(x=t_start, color='black', linestyle='--', linewidth=1)
plt.axvline(x=t_end, color='black', linestyle='--', linewidth=1)

plt.xlabel('time [ms]')
plt.ylabel(r'W_{perp} [J/m]')
plt.title(f'Perpendicular energy during plasma discharge #{shot_no}')
plt.legend()
plt.grid(True)

# plot ohmic heating power
plt.figure(figsize=(10, 6))
plt.plot(P_ohm['t'], P_ohm['P_ohm'], label='ohmic heating power')

plt.axvline(x=t_start, color='black', linestyle='--', linewidth=1)
plt.axvline(x=t_end, color='black', linestyle='--', linewidth=1)

plt.xlabel('time [ms]')
plt.ylabel(r'P_{\Omega} [W]')
plt.title(f'Ohmic heating power during plasma discharge #{shot_no}')
plt.legend()
plt.grid(True)

# plot thermal energy
plt.figure(figsize=(10, 6))
plt.plot(W_th['t'], W_th['W_th'], label='thermal energy')

plt.axvline(x=t_start, color='black', linestyle='--', linewidth=1)
plt.axvline(x=t_end, color='black', linestyle='--', linewidth=1)

plt.xlabel('time [ms]')
plt.ylabel(r'W_{th} [J]')
plt.title(f'Thermal energy during plasma discharge #{shot_no}')
plt.legend()
plt.grid(True)

# plot energy confinement time
plt.figure(figsize=(10, 6))
plt.plot(tau_E['t'], tau_E['tau_E'], label='energy confinement time')

plt.axvline(x=t_start, color='black', linestyle='--', linewidth=1)
plt.axvline(x=t_end, color='black', linestyle='--', linewidth=1)

plt.xlabel('time [ms]')
plt.ylabel(r'\tau_{E} [s]')
plt.title(f'Energy confinement time during plasma discharge #{shot_no}')
plt.legend()
plt.grid(True)

plt.show()
