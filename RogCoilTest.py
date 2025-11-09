import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from functions import get_plasma_time_parameters, get_U_loop_data, get_Ip_data, get_integrated_Bt_data, get_MHD_ring_data

from scipy.integrate import cumulative_trapezoid
from sklearn.linear_model import LinearRegression





# ---- společné funkce ----
def weighted_moving_average(x, N):
    w = np.hanning(N)
    w /= w.sum()
    return np.convolve(x, w, mode='same')


# ---- funkce pro jeden výstřel ----
def process_shot(shot_no, saveFig=True, fileFormat='png'):
    '''Zpracuje jeden výstřel a vrátí výsledky jako slovník s DataFrames.'''

    # --- načtení dat ---
    try:
        t_start, t_end = get_plasma_time_parameters(shot_no=shot_no)
    except Exception:
        print(f'Missing plasma time parameters for shot #{shot_no}, using zeros instead.')
        t_start = 0
        t_end = 0
    U_loop = get_U_loop_data(shot_no=shot_no)
    try:
        I_p = get_Ip_data(shot_no)
    except Exception:
        print(f'No plasma current data found for shot #{shot_no}, using zeros instead.')
        I_p = pd.DataFrame({'t': U_loop['t'], 'Ip': np.zeros(len(U_loop['t']))})
    Bt = get_integrated_Bt_data(shot_no)
    MHD_data = get_MHD_ring_data(shot_no)

    filename = f'http://golem.fjfi.cvut.cz/shots/{shot_no}/Devices/Oscilloscopes/TektrMSO64-a/TektrMSO64_ALL.csv'
    if shot_no <= 50260:
        raw_data = pd.read_csv(filename, skiprows=12, sep=',', decimal='.', names=['t', 'dia in', 'dia out', 'rog in', 'Bt HFS'])
    elif shot_no < 50299:
        raw_data = pd.read_csv(filename, skiprows=12, sep=',', decimal='.', names=['t', 'rog in', 'dia in', 'dia out', 'Bt HFS', 'none1', 't_int', 'rog int'])

    # --- korekce polarity ---
    if shot_no <= 50260:
        raw_data['rog in'] *= -1
    if shot_no < 50299:
        raw_data['Bt HFS'] *= -1

    # --- korekce offsetů ---
    # cols_to_offset = [col for col in raw_data.columns if col != 't']
    try:
        raw_data['dia in'] -= np.mean(raw_data['dia in'][:100])
        raw_data['dia out'] -= np.mean(raw_data['dia out'][:100])
        raw_data['rog in'] -= np.mean(raw_data['rog in'][-100:])
        raw_data['Bt HFS'] -= np.mean(raw_data['Bt HFS'][:100])
        for i in range(1, 17):
            MHD_data[f'f{i}'] -= np.mean(MHD_data[f'f{i}'][-5000:])
        for i in range(1, 5):
            MHD_data[f'lim{i}'] -= np.mean(MHD_data[f'lim{i}'][-5000:])
        for i in range(1, 3):
            MHD_data[f'diamagnet{i}'] -= np.mean(MHD_data[f'diamagnet{i}'][:100])
            MHD_data[f'rogowski{i}'] -= np.mean(MHD_data[f'rogowski{i}'][-100:])
            MHD_data[f'tor{i}'] -= np.mean(MHD_data[f'tor{i}'][:100])
        MHD_data['noise'] -= np.mean(MHD_data['noise'][-5000:])
    except Exception:
        print('Data not measured for shot #{shot_no}.')

    # --- vyhlazení ---
    N = 50
    if shot_no < 50299:
        cols_to_smooth = [col for col in raw_data.columns if col != 't']
        raw_data[cols_to_smooth] = raw_data[cols_to_smooth].apply(lambda x: weighted_moving_average(x, N))
    cols_to_smooth = [col for col in MHD_data.columns if col != 't']
    MHD_data[cols_to_smooth] = MHD_data[cols_to_smooth].apply(lambda x: weighted_moving_average(x, N))

    # --- integrace ---
    K_cal = 15.15e6
    interp_raw_data = {}
    integral_raw_data = {}
    integral_MHD_data = {}
    
    # --- interpolační přepočet časů ---
    I_p_interp = np.interp(MHD_data['t'], I_p['t'], I_p['Ip'])
    Bt_interp = np.interp(MHD_data['t'], Bt['t'], Bt['Bt'])
    U_loop_interp = np.interp(MHD_data['t'], U_loop['t'], U_loop['U_loop'])
    if shot_no < 50299:
        for col in [col for col in raw_data.columns if col != 't']:
            interp_raw_data[col] = np.interp(MHD_data['t'], raw_data['t'], raw_data[col])
    
    if shot_no <= 50260:
        MHD_data['noise'] = MHD_data['rogowski1']
        MHD_data['rogowski1'] = interp_raw_data['rog in']
        MHD_data['diamagnet1'] = interp_raw_data['dia in']
        MHD_data['diamagnet2'] = interp_raw_data['dia out']
        MHD_data['tor2'] = interp_raw_data['Bt HFS']
        MHD_data['U_loop'] = U_loop_interp
    elif shot_no < 50299:
        MHD_data['noise'] = MHD_data['rogowski1']
        MHD_data['rogowski1'] = interp_raw_data['rog in']
        MHD_data['U_loop'] = U_loop_interp
    else:
        MHD_data['U_loop MHD'] = 10*MHD_data['U_loop']  # kvůli děliči napětí
        MHD_data['U_loop'] = U_loop_interp
        
    for col in [c for c in MHD_data.columns if c != 't']:
        integral_MHD_data[col] = cumulative_trapezoid(MHD_data[col], MHD_data['t'], initial=0)
    
    # --- calibration constants ---
    integral_MHD_data['rogowski1'] *= K_cal
    
    # --- sestavení výsledného DataFrame ---
    int_data = pd.DataFrame({
        't': MHD_data['t'],
        **{col: integral_MHD_data[col] for col in integral_MHD_data},
        'Ip': I_p_interp,
        'Bt': Bt_interp
    })

    # ---- data fit ----
    
    # mask for offset fitting
    mask = (int_data['t'] >= 13e-3) | (int_data['t'] <= 3e-3)
    mask = int_data['t'] <= 40e-3
    
    # fit models
    fits = {
        'U_loop': ['U_loop'],
        'Bt': ['Bt'],
        'noise': ['noise'],
        'all': ['U_loop', 'Bt', 'noise']
    }
    
    # fit computation
    denoised_dict = {}
    coeffs_dict = {}
    
    for key, features in fits.items():
        
        X = np.column_stack([int_data[f][mask] for f in features])
        y = int_data['rogowski1'][mask] - int_data['Ip'][mask]
        model = LinearRegression(fit_intercept=True).fit(X, y)
        coefs = model.coef_
        intercept = model.intercept_
        R2 = model.score(X, y)
        
        # 'denoised' signal reconstruction
        pred = sum(coefs[i] * int_data[features[i]] for i in range(len(features)))
        denoised = int_data['rogowski1'] - pred - intercept
        
        # saving results
        denoised_dict[key] = denoised
        coeffs_dict[key] = {
            'features': features,
            'coefficients': coefs.tolist(),
            'intercept': intercept,
            'R2': R2
        }
    
        # Krátký výpis
        coef_str = ', '.join([f'k{i+1}={c:.3e}' for i, c in enumerate(coefs)])
        print(f'  → Regression ({key}): {coef_str}, k0={intercept:.3e}, R²={R2:.4f}')
    
    # --- vytvoření DataFrame se všemi variantami denoised ---
    denoised_df = pd.DataFrame({
        't': int_data['t'],
        **{f'denoised_{key}': val for key, val in denoised_dict.items()}
    })
    
    # --- finální slovník výsledků ---
    results = {
        'shot_no': shot_no,
        't_start': t_start,
        't_end': t_end,
        'MHD_data': MHD_data,
        'int_data': int_data,
        'denoised': denoised_df,
        'coefficients': coeffs_dict
    }

    return results


# ---- hlavní smyčka pro více výstřelů ----

#(vac, plas) = (50181/50182, 50180/50183), (50184/50185/50186/50188/50189, 50187), (50190, 50191/50192/50193/50194), (50208, 50209)

# shot_list = [50305]
# shot_list = [50302]
# shot_list = [50304]
# shot_list = list(range(50247, 50252))
# shot_list = list(range(50253, 50256))
# shot_list = list(range(50245, 50260+1))
shot_list = list(range(50299, 50304+1))

# vacuum only Bt
shot_no_Bt = [50250, 50258, 50259]
# vacuum only current drive
shot_no_cd = [50248, 50257]
# vacuum all off
shot_no_off = [50249, 50260]
# vacuum all on
shot_no_on = [50247, 50256]
# plasma
shot_no_plas = [50251, 50253, 50254, 50255]


all_results = {}

for shot in shot_list:
    try:
        print(f'\nProcessing shot #{shot} ...')
        all_results[shot] = process_shot(shot, saveFig=False)
    except Exception as e:
        print(f'Error processing shot #{shot}: {e}')
        continue  # pokračuj na další výstřel


'''
GRAFY
'''

# ---- grafy jednotlivých shotů ----
for shot, data in all_results.items():
    plt.figure(figsize=(10, 6))
    plt.plot(data['MHD_data']['t'], data['MHD_data']['U_loop']/np.max(abs(data['MHD_data']['U_loop'])), label=r'$U_{loop}$ basic') if 'U_loop' in data['MHD_data'] else None
    plt.plot(data['MHD_data']['t'], data['MHD_data']['U_loop MHD']/np.max(abs(data['MHD_data']['U_loop MHD'])), label=r'$U_{loop}$') if 'U_loop MHD' in data['MHD_data'] else None
    plt.plot(data['MHD_data']['t'], data['MHD_data']['rogowski1']/np.max(abs(data['MHD_data']['rogowski1'])), label='rog in') 
    plt.plot(data['MHD_data']['t'], data['MHD_data']['rogowski2']/np.max(abs(data['MHD_data']['rogowski2'])), label='rog out')
    plt.plot(data['MHD_data']['t'], data['MHD_data']['tor1'], label='tor1')
    plt.plot(data['MHD_data']['t'], data['MHD_data']['tor2'], label='tor2')
    plt.xlabel('t (s)')
    plt.ylabel('U (a.u.)')
    plt.title(f'#{shot} raw signal')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(10, 6))
    # plt.plot(data['denoised']['t'], data['denoised']['denoised']/np.max(abs(data['denoised']['denoised'])), label=f'denoised #{shot}')
    plt.plot(data['int_data']['t'], data['int_data']['U_loop MHD']/np.max(abs(data['int_data']['U_loop MHD'])), label=r'$U_{loop}$' f' #{shot}') if 'U_loop MHD' in data['MHD_data'] else None
    plt.plot(data['int_data']['t'], data['int_data']['rogowski1']/np.max(abs(data['int_data']['rogowski1'])), label=f'rog in #{shot}')
    plt.plot(data['int_data']['t'], data['int_data']['rogowski2']/np.max(abs(data['int_data']['rogowski2'])), label=f'rog out #{shot}')
    plt.plot(data['int_data']['t'], data['int_data']['tor1']/np.max(abs(data['int_data']['tor1'])), label=f'tor1 #{shot}')
    plt.plot(data['int_data']['t'], data['int_data']['tor2']/np.max(abs(data['int_data']['tor2'])), label=f'tor2 #{shot}')
    plt.plot(data['int_data']['t'], data['int_data']['noise']/np.max(abs(data['int_data']['noise'])), label=f'noise #{shot}')
    plt.xlabel('t (s)')
    plt.ylabel('I (a. u.)')
    plt.title(f'#{shot} integrated signal')
    plt.legend()
    plt.grid(True)
    plt.show()

# ---- srovnávací grafy ----
plt.figure(figsize=(10, 6))
for shot, data in all_results.items():
    plt.plot(data['MHD_data']['t'], data['MHD_data']['rogowski1']/np.max(abs(data['MHD_data']['rogowski1'])), label=f'rog in #{shot}')
    plt.plot(data['MHD_data']['t'], data['MHD_data']['U_loop MHD']/np.max(abs(data['MHD_data']['U_loop MHD'])), label=r'$U_{loop}$' f' #{shot}') if 'U_loop MHD' in data['MHD_data'] else None
    plt.plot(data['MHD_data']['t'], data['MHD_data']['tor2']/np.max(abs(data['MHD_data']['tor2'])), label=f'tor2 #{shot}')
    plt.plot(data['MHD_data']['t'], data['MHD_data']['noise']/np.max(abs(data['MHD_data']['noise'])), label=f'noise #{shot}')
    
plt.xlabel('t (s)')
plt.ylabel('U (a.u.)')
plt.title('Comparison of raw signals')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
for shot, data in all_results.items():
    plt.plot(data['MHD_data']['t'], data['MHD_data']['rogowski1'], label=f'rog in #{shot}')
    
plt.xlabel('t (s)')
plt.ylabel('U (a.u.)')
plt.title('Comparison of raw Rogowski signals')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
for shot, data in all_results.items():
    # plt.plot(data['denoised']['t'], data['denoised']['denoised_Bt']/np.max(abs(data['denoised']['denoised_Bt'])), label=f'rog-Bt #{shot}')
    # plt.plot(data['denoised']['t'], data['denoised']['denoised_all']/np.max(abs(data['denoised']['denoised_all'])), label=f'rog-all #{shot}')
    plt.plot(data['int_data']['t'], -data['int_data']['rogowski1']/np.max(abs(data['int_data']['rogowski1'])), label=f'rog in #{shot}')
    plt.plot(data['int_data']['t'], data['int_data']['Bt']/np.max(abs(data['int_data']['Bt'])), label=f'Bt #{shot}')
    # plt.plot(data['int_data']['t'], data['int_data']['tor2']/np.max(abs(data['int_data']['tor2'])), label=f'tor2 #{shot}')
    plt.plot(data['int_data']['t'], -data['int_data']['rogowski1']/np.max(abs(data['int_data']['rogowski1']))-data['int_data']['Bt']/np.max(abs(data['int_data']['Bt'])), label=f'diff #{shot}')
# plt.plot(all_results[shot_no_plas]['int_data']['t'], all_results[shot_no_plas]['int_data']['rogowski1']-all_results[shot_no_on]['int_data']['rogowski1'], label=f'Ip+? #{shot}') if shot_no_plas in all_results else None

plt.xlabel('t (s)')
plt.ylabel('I (a.u.)')
plt.title('Comparison of integrated Rogowski signals')
plt.legend()
plt.grid(True)
plt.show()

# ---- grafy jednotlivých efektů ----
# Bt effect
if shot_no_Bt in all_results:
    subset = {s: all_results[s] for s in [shot_no_Bt]}
    plt.figure(figsize=(10, 6))
    for shot, data in subset.items():
        plt.plot(data['int_data']['t'], data['int_data']['rogowski1']/np.max(abs(data['int_data']['rogowski1'])), label=f'rog in #{shot}')
        plt.plot(data['int_data']['t'], data['int_data']['tor2']/np.max(abs(data['int_data']['tor2'])), label=f'tor2 #{shot}')
        plt.plot(data['int_data']['t'], data['int_data']['Bt']/np.max(abs(data['int_data']['Bt'])), label=f'Bt #{shot}')
    
    plt.xlabel('t (s)')
    plt.ylabel('I (a.u.)')
    plt.title('Bt field effect')
    plt.legend()
    plt.grid(True)
    plt.show()

# U_loop effect
if shot_no_cd in all_results:
    subset = {s: all_results[s] for s in [shot_no_cd]}
    plt.figure(figsize=(10, 6))
    for shot, data in subset.items():
        plt.plot(data['MHD_data']['t'], data['MHD_data']['rogowski1']/np.max(abs(data['MHD_data']['rogowski1'])), label=f'rog in #{shot}')
        plt.plot(data['MHD_data']['t'], data['MHD_data']['U_loop']/np.max(abs(data['MHD_data']['U_loop'])), label=r'$U_{loop}$ basic' f' #{shot}')
        plt.plot(data['MHD_data']['t'], data['MHD_data']['noise']/np.max(abs(data['MHD_data']['noise'])), label=f'noise #{shot}')
    
    plt.xlabel('t (s)')
    plt.ylabel('U (a.u.)')
    plt.title(r'$U_{loop}$ field effect')
    plt.legend()
    plt.grid(True)
    plt.show()

# Bt and U_loop effect
if shot_no_on in all_results:
    subset = {s: all_results[s] for s in [shot_no_on]}
    plt.figure(figsize=(10, 6))
    for shot, data in subset.items():
        plt.plot(data['int_data']['t'], data['int_data']['rogowski1']/np.max(abs(data['int_data']['rogowski1'])), label=f'rog in #{shot}')
        plt.plot(data['int_data']['t'], data['int_data']['U_loop']/np.max(abs(data['int_data']['U_loop'])), label=r'$U_{loop}$ basic' f' #{shot}')
        plt.plot(data['int_data']['t'], data['int_data']['tor2']/np.max(abs(data['int_data']['tor2'])), label=f'tor2 #{shot}')
        plt.plot(data['int_data']['t'], data['int_data']['Bt']/np.max(abs(data['int_data']['Bt'])), label=f'Bt #{shot}')
        plt.plot(data['int_data']['t'], data['int_data']['noise']/np.max(abs(data['int_data']['noise'])), label=f'noise #{shot}')
    
    plt.xlabel('t (s)')
    plt.ylabel('I (a.u.)')
    plt.title(r'Bt and $U_{loop}$ field effect')
    plt.legend()
    plt.grid(True)
    plt.show()

# all off
if shot_no_off in all_results:
    subset = {s: all_results[s] for s in [shot_no_off]}
    plt.figure(figsize=(10, 6))
    for shot, data in subset.items():
        plt.plot(data['MHD_data']['t'], data['MHD_data']['rogowski1']/np.max(abs(data['MHD_data']['rogowski1'])), label=f'rog in #{shot}')
        plt.plot(data['MHD_data']['t'], data['MHD_data']['U_loop']/np.max(abs(data['MHD_data']['U_loop'])), label=r'$U_{loop}$ basic' f' #{shot}')
        plt.plot(data['MHD_data']['t'], data['MHD_data']['tor2']/np.max(abs(data['MHD_data']['tor2'])), label=f'tor2 #{shot}')
        plt.plot(data['MHD_data']['t'], data['MHD_data']['noise']/np.max(abs(data['MHD_data']['noise'])), label=f'noise #{shot}')
        
    plt.xlabel('t (s)')
    plt.ylabel('U (a.u.)')
    plt.title('Vacuum, all off, raw signals')
    plt.legend()
    plt.grid(True)
    plt.show()

# effects
plt.figure(figsize=(10, 6))
plt.plot(data['MHD_data']['t'], data['MHD_data']['rogowski1']/np.max(abs(data['MHD_data']['rogowski1'])), label=f'rog in #{shot}')
plt.plot(data['MHD_data']['t'], data['MHD_data']['U_loop']/np.max(abs(data['MHD_data']['U_loop'])), label=r'$U_{loop}$ basic' f' #{shot}')
plt.plot(data['MHD_data']['t'], data['MHD_data']['tor2']/np.max(abs(data['MHD_data']['tor2'])), label=f'tor2 #{shot}')
plt.plot(data['MHD_data']['t'], data['MHD_data']['noise']/np.max(abs(data['MHD_data']['noise'])), label=f'noise #{shot}')
    
plt.xlabel('t (s)')
plt.ylabel('U (a.u.)')
plt.title('Effects')
plt.legend()
plt.grid(True)
plt.show()

