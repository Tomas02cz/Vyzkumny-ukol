from functions import *
from scan_database import shot_no_with_diagnostics

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.integrate import cumulative_trapezoid



''' FUNCTION'S DEFINITIONS ''' 

def rogowski_coil_analysis(shot_no, save, show):     
    ''' this is the main function of Rogowski coils data analysis
        
        shot_no = number of shot in Golem database
    '''
    # load data
    t_start, t_end = get_plasma_time_parameters(shot_no)
    raw_data = get_rogowski_data(shot_no)
    
    # plot raw signal
    plot_raw_signal(shot_no, raw_data, t_start, t_end, save=save, show=show) # set save=1 for saving the fig
    
    # remove offset from raw signal and plot the result
    raw_data_detrended = remove_offset(shot_no, raw_data)
    show_offset_fit(shot_no, raw_data, raw_data_detrended, t_start, t_end, save=save, show=show) # set save=1 for saving the fig
    # raw_data_detrended = raw_data.copy()
    
    # integrate raw signal and plot integrated data
    integrated_data = integrate_signal(raw_data_detrended)
    plot_integrated_signal(shot_no, integrated_data, t_start, t_end, save=save, show=show) # set save=1 for saving the fig
    
    return integrated_data

def plot_raw_signal(shot_no, raw_data, t_start, t_end, save, show):
    # graph of raw data from plasma discharge
    plt.figure(figsize=(10, 6))
    plt.plot(raw_data['t'], raw_data['rogowski1'], label='inner Rogowski coil')
    plt.plot(raw_data['t'], raw_data['rogowski2'], label='outer Rogowski coil')
    
    plt.axvline(x=t_start, color='black', linestyle='--', linewidth=1) # show duration of plasma
    plt.axvline(x=t_end, color='black', linestyle='--', linewidth=1)
    
    plt.xlabel('time [ms]')
    plt.ylabel('voltage [V]')
    plt.title(f'Raw signal from Rogowski coils during shot #{shot_no}')
    plt.legend()
    plt.grid(True)
    
    if save == 1:
        plt.savefig(f'results/{shot_no}_RogowskiCoil_RawSignal.png')
    if show == 1:
        plt.show()
        
def remove_offset(shot_no, raw_data):
    ''' computes offset fit and substructs it from raw signal
    
        shot_no = number of the shot in Golem database
        raw_data = pd.DataFrame of raw signal from Rogowski coils
    '''
    # fit for off-set subtruction
    n = 0 # degree of fit polynomial
    
    # computing fit coefficients and the offset polynomial
    coeffs1 = np.polyfit(raw_data['t'].values, raw_data['rogowski1'].values, n)
    coeffs2 = np.polyfit(raw_data['t'].values, raw_data['rogowski2'].values, n)
    trend1 = np.polyval(coeffs1, raw_data['t'].values)
    trend2 = np.polyval(coeffs2, raw_data['t'].values)
    
    # substruction of offset from raw signal
    raw_data_detrended = raw_data.copy()
    raw_data_detrended['rogowski1'] = raw_data['rogowski1'] - trend1
    raw_data_detrended['rogowski2'] = raw_data['rogowski2'] - trend2
    
    return raw_data_detrended

def show_offset_fit(shot_no, raw_data, raw_data_detrended, t_start, t_end, save, show):
    ''' shows offset fit substructed from raw signal 
    
        shot_no = number of the shot in Golem database
        raw_data = pd.DataFrame of raw signal from Rogowski coils
        raw_data_detrended = pd.DataFrame of raw signal from Rogowski coils without offset
        t_start, t_end = time points of plasma duration
        save = for optinal saving of graph if save == 1
    '''
    plt.figure(figsize=(10, 6))
    plt.plot(raw_data['t'], raw_data['rogowski1'], label='inner Rogowski coil')
    plt.plot(raw_data_detrended['t'], raw_data_detrended['rogowski1'], label='inner without off-set')
    plt.plot(raw_data['t'], raw_data['rogowski2'], label='outer Rogowski coil')
    plt.plot(raw_data_detrended['t'], raw_data_detrended['rogowski2'], label='outer without off-set')
    
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
def integrate_signal(raw_data_detrended):
    # integration of raw data
    integral_U1 = -cumulative_trapezoid(raw_data_detrended['rogowski1'], raw_data_detrended['t'], initial=0)
    integral_U2 = -cumulative_trapezoid(raw_data_detrended['rogowski2'], raw_data_detrended['t'], initial=0)
    
    # make new pd.DataFrame of integrated data
    integrated_data = pd.DataFrame({'t': raw_data_detrended['t'], 'rogowski1': integral_U1, 'rogowski2': integral_U2})
    
    return integrated_data

def plot_integrated_signal(shot_no, integrated_data, t_start, t_end, save, show):
    # graph of integrated data
    plt.figure(figsize=(10, 6))
    plt.plot(integrated_data['t'], integrated_data['rogowski1'], label='inner Rogowski coil')
    plt.plot(integrated_data['t'], integrated_data['rogowski2'], label='outer Rogowski coil')
    
    # show plasma duration
    plt.axvline(x=t_start, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=t_end, color='black', linestyle='--', linewidth=1)
    
    plt.xlabel('time [ms]')
    plt.ylabel(f'$I_p$ [kA]')
    plt.title(f'Integrated signal from Rogowski coils during shot #{shot_no}')
    plt.legend()
    plt.grid(True)
    
    if save == 1:
        plt.savefig(f'results/{shot_no}_RogowskiCoil_IntegratedSignal.png')
    if show == 1:
        plt.show()    
        
        
    plt.figure(figsize=(10, 6)) # TODO delete this figure after inner Rogowski coil calibration
    plt.plot(integrated_data['t']*1000, integrated_data['rogowski1'], label='inner Rogowski coil')
    plt.axvline(x=t_start, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=t_end, color='black', linestyle='--', linewidth=1)
    plt.xlabel('time [ms]')
    plt.ylabel(f'$I_p$ [kA]')
    plt.title(f'Integrated signal from inner Rogowski coil during shot #{shot_no}')
    plt.legend()
    plt.grid(True)
    
    if save == 1:
        plt.savefig(f'results/{shot_no}_InnerRogowskiCoil_IntegratedSignal_without1000.png')
    if show == 1:
        plt.show()
   
def plot_all_shots(shots_list, shot_type_name, save, show):
    plt.figure(figsize=(10, 6))
    
    for shot_no, data in shots_list:
        plt.plot(data['t'], data['rogowski1'], label=f'#{shot_no}')
    
    plt.title(f'Integrated signal – {shot_type_name} shots (inner Rogowski coil)')
    plt.xlabel('time [ms]')
    plt.ylabel('$I_p$ [kA]')
    plt.grid(True)
    plt.legend()
    if save == 1:
        plt.savefig(f'results/{shot_type_name}_shots_between#{shots_list}_InnerRogowskiCoil_IntegratedSignal.png')
    if show == 1:
        plt.show()


''' MAIN '''    
 
# shot parameters
shot_no_vacuum = 49092
shot_no = 49093 #47768 #47768 #48531 44800

saveFormat: str = 'png' # TODO insert save format option


# load data
t_start, t_end = get_plasma_time_parameters(shot_no)
raw_data = get_rogowski_data(shot_no)
raw_data_vacuum = get_rogowski_data(shot_no_vacuum)



vacuum_shots = []
plasma_shots = []

for _, row in shot_no_with_diagnostics.iterrows():
    shot_no = row['shot number']
    shot_type = row['shot type']
    
    integrated_data = rogowski_coil_analysis(shot_no, save=0, show=0)
    
    if shot_type == 0:
        vacuum_shots.append((shot_no, integrated_data))
    else:
        plasma_shots.append((shot_no, integrated_data))

plot_all_shots(vacuum_shots, 'vacuum', save=1, show=1)
plot_all_shots(plasma_shots, 'plasma', save=1, show=1)