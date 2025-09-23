import requests
import io

import pandas as pd
import numpy as np

from scipy.signal import ShortTimeFFT
from scipy.signal.windows import boxcar

from scipy.stats import pearsonr
from functools import partial
from multiprocessing import Pool

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk



FONT: tuple[str, int] = ('Courier New', 13)

# URLs
URL_B_PLASMA: str = 'http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/PlasmaDetection/Results/b_plasma'
URL_R: str = 'http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/FastCameras/Camera_Radial/CameraRadialPosition'
URL_T_START: str = 'http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/PlasmaDetection/Results/t_plasma_start'
URL_T_END: str = 'http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/PlasmaDetection/Results/t_plasma_end'
URL_BT: str = 'http://golem.fjfi.cvut.cz/shotdir/{shot_no}/Diagnostics/BasicDiagnostics/Results/Bt.csv'
URL_Bt_coil: str = 'http://golem.fjfi.cvut.cz/shotdir/{shot_no}/Diagnostics/PlasmaDetection/U_BtCoil.csv'
URL_IP: str = 'http://golem.fjfi.cvut.cz/shotdir/{shot_no}/Diagnostics/BasicDiagnostics/Results/Ip.csv'
URL_Uloop: str = 'http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/BasicDiagnostics/Results/U_loop.csv'
# Mirnov coils
URL_MIRNOV1: str = "http://golem.fjfi.cvut.cz/shots/{shot_no}/Devices/DASs/2NI_PC-VoSv/NI.lvm"
URL_MIRNOV2: str = 'http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/MHDring-TM/DAS_raw_data_dir/rawData.csv'
# Rogowski coils
URL_ROGOWSKI_OUTER: str = 'http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/PlasmaDetection/U_RogCoil.csv'
URL_ROGOWSKI_OLD_INNER: str = 'http://golem.fjfi.cvut.cz/shotdir/{shot_no}/Diagnostics/MHDring-TM/U_rogowski.csv'
# diamagnetic loops
URL_DIAMAGNET_OLD_INNER: str = 'http://golem.fjfi.cvut.cz/shotdir/{shot_no}/Diagnostics/MHDring-TM/U_diam_inner.csv'
URL_DIAMAGNET_OLD_OUTER: str = 'http://golem.fjfi.cvut.cz/shotdir/{shot_no}/Diagnostics/MHDring-TM/U_diam_outer.csv'


# sampling period in seconds
T_SAMPLE: float = 10**(-6)
# shortest allow time interval
L_MIN = 50 * T_SAMPLE

# radii of GOLEM tokamak in meters
a = 0.085
R = 0.4

mu_0 = 4 * np.pi * 10**(-7)


# Good
def check_plasma_presence(shot_no: int) -> bool:
    content = requests.get(URL_B_PLASMA.format(shot_no=shot_no)).content
    b_plasma = pd.read_csv(io.StringIO(content.decode('iso-8859-1')), names = ['b_plasma'])
    if b_plasma['b_plasma'][0] == 1:
        return True
    else:
        return False


def check_camera(shot_no) -> bool:
    status_code = requests.get(URL_R.format(shot_no=shot_no)).status_code
    if status_code == 200:
        return True
    else:
        return False

# Good
def get_plasma_time_parameters(shot_no: int):
    # t_start
    content = requests.get(URL_T_START.format(shot_no=shot_no)).content
    # reads t_start
    t_start = pd.read_csv(io.StringIO(content.decode('iso-8859-1')), names = ['t_start'])
    t_start = float(t_start.to_numpy())
    # t_end
    content = requests.get(URL_T_END.format(shot_no=shot_no)).content
    # reads t_end
    t_end = pd.read_csv(io.StringIO(content.decode('iso-8859-1')), names = ['t_end'])
    t_end = float(t_end.to_numpy())
    return t_start/1000, t_end/1000

# selecting time interval from pd.DataFrame based on column labeled 't'
def select_time_interval(raw_data: pd.DataFrame, t_start: float, t_end: float) -> pd.DataFrame:
    t_start *= 1000
    t_end *= 1000   # to be in ms
    
    if t_start < 0 or t_end < 0:
        messagebox.showerror(title='Invalid time interval.', message='One of the end points of the time interval is negative.')
    if t_end - t_start < L_MIN:
        messagebox.showerror(title='Invalid time interval.', message=f'The length of the time interval must be at least {L_MIN*10**6} ns.')
        return raw_data
    # choosing apropriate time interval for Mirnov coil plot
    raw_data['t'] = pd.to_numeric(raw_data['t'], errors='coerce').fillna(0)
    cut_data = raw_data[(raw_data['t'] > t_start) & (raw_data['t'] < t_end)].reset_index(drop = True)
    return cut_data


# loads data from mirnov coils and saves them as csv file localy
def get_Bt_selected_data(shot_no: int, t_start: float = None, t_end: float = None) -> pd.DataFrame:
    # data path
    content = requests.get(URL_BT.format(shot_no=shot_no)).content
    # reads data
    raw_data = pd.read_csv(io.StringIO(content.decode('iso-8859-1')), sep = ',', names = ['t', 'Bt'])
    if t_start is None or t_end is None:
        t_start, t_end = get_plasma_time_parameters(shot_no)
    return select_time_interval(raw_data, t_start, t_end)


# loads data from mirnov coils and saves them as csv file localy
def get_Ip_selected_data(shot_no: int, t_start: float = None, t_end: float = None) -> pd.DataFrame:
    # data path
    content = requests.get(URL_IP.format(shot_no=shot_no)).content
    # reads data
    raw_data = pd.read_csv(io.StringIO(content.decode('iso-8859-1')), sep = ',', names = ['t', 'Ip'])  
    if t_start is None or t_end is None:
        t_start, t_end = get_plasma_time_parameters(shot_no)
    # cuts data
    return select_time_interval(raw_data, t_start, t_end)

# loads data from camera radial position
def get_R_data(shot_no: int) -> pd.DataFrame:
    # data path
    content = requests.get(URL_R.format(shot_no=shot_no)).content
    # reads data
    raw_data = pd.read_csv(io.StringIO(content.decode('iso-8859-1')), sep = ',', names = ['t', 'R'])
    return raw_data

# loads Bt data and saves them as csv file localy
def get_Bt_data(shot_no: int) -> pd.DataFrame:
    # data path
    if shot_no < 41524:
        URL: str = 'http://golem.fjfi.cvut.cz/shotdir/{shot_no}/Diagnostics/BasicDiagnostics/DetectPlasma/U_BtCoil.csv'
        content = requests.get(URL.format(shot_no=shot_no)).content
        # reads data
        raw_data = pd.read_csv(io.StringIO(content.decode('iso-8859-1')), sep = ',', names = ['t', 'Bt'])
    else:
        content = requests.get(URL_Bt_coil.format(shot_no=shot_no)).content
        # reads data
        raw_data = pd.read_csv(io.StringIO(content.decode('iso-8859-1')), sep = ',', names = ['t', 'Bt'])
    
    return raw_data

# loads integrated Bt data and saves them as csv file localy
def get_integrated_Bt_data(shot_no: int) -> pd.DataFrame:
    # data path
    URL: str = 'http://golem.fjfi.cvut.cz/shotdir/{shot_no}/Diagnostics/PlasmaDetection/U_IntBtCoil.csv'
    content = requests.get(URL.format(shot_no=shot_no)).content
    # reads data
    data = pd.read_csv(io.StringIO(content.decode('iso-8859-1')), sep = ',', names = ['t', 'Bt'])
    
    return data

# loads Ip data and saves them as csv file localy
def get_TF_data(shot_no: int) -> pd.DataFrame:
    # data path
    content = requests.get(URL_Bt_coil.format(shot_no=shot_no)).content
    # reads data
    raw_data = pd.read_csv(io.StringIO(content.decode('iso-8859-1')), sep = ',', names = ['t', 'Ip'])
    
    return raw_data

# loads U_loop data and saves them as csv file localy
def get_U_loop_data(shot_no: int) -> pd.DataFrame:
    # data path
    content = requests.get(URL_Uloop.format(shot_no=shot_no)).content
    # reads data
    raw_data = pd.read_csv(io.StringIO(content.decode('iso-8859-1')), sep = ',', decimal = '.', names = ['t', 'U_loop'])
    
    return raw_data

# loads data from mirnov coils and saves them as csv file localy
def get_mirnov_data(shot_no: int) -> pd.DataFrame:
    # data path and reads data
    if shot_no < 47680:
        content = requests.get(URL_MIRNOV1.format(shot_no=shot_no)).content
        raw_data = pd.read_csv(io.StringIO(content.decode('iso-8859-1')), sep = '\t', decimal = ',')
        
        no_of_diagnostics = raw_data.shape[1] - 1
        print(no_of_diagnostics)
        raw_data.columns = ['t', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16']
    else:
        content = requests.get(URL_MIRNOV2.format(shot_no=shot_no)).content
        raw_data = pd.read_csv(io.StringIO(content.decode('iso-8859-1')), sep=',', decimal='.')
        
        raw_data = raw_data.iloc[:, :17]
        raw_data.columns = ['t', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16']
        
    # some coils have flipeed polarity
    if shot_no < 46688:  
        raw_data['f10'] *= -1
        raw_data['f11'] *= -1
        raw_data['f12'] *= -1
        raw_data['f14'] *= -1
        raw_data['f16'] *= -1    
    else:
        raw_data['f1'] *= -1
        raw_data['f9'] *= -1
        raw_data['f13'] *= -1
    
    raw_data = raw_data.apply(pd.to_numeric, errors='coerce').fillna(0.0)

    return raw_data

# loads data from diamagnetic coils and saves them as csv file localy
def get_diamagnet_data(shot_no: int) -> pd.DataFrame:
    # data path and reads data
    if shot_no < 47680:
        content1 = requests.get(URL_DIAMAGNET_OLD_INNER.format(shot_no=shot_no)).content
        raw_data_inner_diamagnet = pd.read_csv(io.StringIO(content1.decode('iso-8859-1')), sep = ',', decimal = '.', names = ['t', 'diamagnet'])
        
        content2 = requests.get(URL_DIAMAGNET_OLD_OUTER.format(shot_no=shot_no)).content
        raw_data_outer_diamagnet = pd.read_csv(io.StringIO(content2.decode('iso-8859-1')), sep=',', decimal='.', names = ['t', 'diamagnet'])
    
        # interpolate data from outer Rogowski coil to match sampling time of inner Rogowski coil 
        interpolated_outer_diamagnet = np.interp(raw_data_inner_diamagnet['t'], raw_data_outer_diamagnet['t'], raw_data_outer_diamagnet['diamagnet'])
        
        # make pandas.DataFrame of both signals
        raw_data = pd.DataFrame({'t': raw_data_inner_diamagnet['t'], 'diamagnet1': raw_data_inner_diamagnet['diamagnet'], 'diamagnet2': interpolated_outer_diamagnet})
        
    else:
        content = requests.get(URL_MIRNOV2.format(shot_no=shot_no)).content
        raw_data = pd.read_csv(io.StringIO(content.decode('iso-8859-1')), sep=',', decimal='.')
        raw_data.columns = ['t', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'diamagnet1', 'diamagnet2', 'rogowski']
    
        # delete data from Mirnov coils and Rogowski coil
        raw_data = raw_data.drop(columns=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'rogowski'])
    
    raw_data = raw_data.apply(pd.to_numeric, errors='coerce').fillna(0.0)

    return raw_data

# loads data from Rogowski coil and saves them as csv file localy
def get_rogowski_data(shot_no: int) -> pd.DataFrame:
    # data path and reads data
    if shot_no < 47680:
        content1 = requests.get(URL_ROGOWSKI_OLD_INNER.format(shot_no=shot_no)).content
        raw_data_inner_RogCoil = pd.read_csv(io.StringIO(content1.decode('iso-8859-1')), sep = ',', decimal = '.', names = ['t', 'rogowski'])
        
        content2 = requests.get(URL_ROGOWSKI_OUTER.format(shot_no=shot_no)).content
        raw_data_outer_RogCoil = pd.read_csv(io.StringIO(content2.decode('iso-8859-1')), sep = ',', decimal = '.', names = ['t', 'rogowski'])
    else:
        content1 = requests.get(URL_MIRNOV2.format(shot_no=shot_no)).content
        raw_data_inner_RogCoil = pd.read_csv(io.StringIO(content1.decode('iso-8859-1')), sep = ',', decimal = '.')
        raw_data_inner_RogCoil.columns = ['t', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'diamagnet1', 'diamagnet2', 'rogowski']

        content2 = requests.get(URL_ROGOWSKI_OUTER.format(shot_no=shot_no)).content
        raw_data_outer_RogCoil = pd.read_csv(io.StringIO(content2.decode('iso-8859-1')), sep=',', decimal='.', names = ['t', 'rogowski'])
    
        # delete data from Mirnov coils and diamagnetic loops
        raw_data_inner_RogCoil = raw_data_inner_RogCoil.drop(columns=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'diamagnet1', 'diamagnet2'])
    
    # interpolate data from outer Rogowski coil to match sampling time of inner Rogowski coil 
    interpolated_rogowski = np.interp(raw_data_inner_RogCoil['t'], raw_data_outer_RogCoil['t'], raw_data_outer_RogCoil['rogowski'])
    
    # make pandas.DataFrame of both signals
    raw_data = pd.DataFrame({'t': raw_data_inner_RogCoil['t'], 'rogowski1': raw_data_inner_RogCoil['rogowski'], 'rogowski2': interpolated_rogowski})
    raw_data = raw_data.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    
    return raw_data


def spectrogram(shot_no: int, coil_id: int, nperseg: int, hop: int, f_sample: float = 1 / T_SAMPLE, win: str = 'boxcar'):
    raw_data = get_mirnov_data(shot_no)
    
    raw_data['t'] *= 1000 # from seconds to miliseconds
    
    raw_data = raw_data.to_numpy()

    win = boxcar(nperseg)
    n_sample = len(raw_data)

    SFT = ShortTimeFFT(win, hop, f_sample)
    
    Sx = SFT.stft(raw_data[:,coil_id])
    t = (SFT.t(n_sample) + raw_data[0,0]) * 1000
    f = SFT.f / 1000

    return Sx, t, f


def moving_average(input: np.array, n: int) -> np.array:
    output = []
    for i in range(0, int(np.round(n / 2))):
        output.append(np.sum(input[0:i]) / (i + 1))

    for i in range(int(np.round(n / 2)), int(np.round(len(input) - n / 2))):
        output.append(np.sum(input[int(np.round(i - n/2)):int(np.round(i + n / 2))]) / n)

    for i in range(int(np.round(len(input) - n / 2)), len(input)):
        output.append(np.sum(input[i:len(input)]) / (len(input) - i + 1))

    return output


def compute_pearson_correlation(data, ref_data):
    output = []
    for L in range(0, int(np.ceil(np.shape(data)[0]/2))):
        output.append(pearsonr(ref_data[L:-1], data[0:-1-L]).statistic)

    output.reverse()

    for L in range(1, int(np.ceil(np.shape(data)[0]/2))):
        output.append(pearsonr(ref_data[0:-1-L], data[L:-1]).statistic)

    return output


def parallel_correlation_progress(shot_no: int, L_interval: list[float], ref_coil: int, progress_bar_label: tk.Label, progress_bar: ttk.Progressbar, ma_win_len: int = 10) -> list[list[float], list[list[float]]]:
    # loads needed data
    progress_bar_label.config(text='Loading data...')
    raw_data = get_mirnov_data(shot_no)
    cut_data = select_time_interval(raw_data, L_interval[0], L_interval[1]).transpose().to_numpy()
    progress_bar.step(1)
    # applies moving average over data
    progress_bar_label.config(text='Computing...')
    for coil_id in range(1, 17):
        cut_data[coil_id] = moving_average(cut_data[coil_id], ma_win_len)
    progress_bar.step(1)
    # place holder list for results
    results = [None] * (len(cut_data) - 1)
    # organizes results and updates progress bar
    def collect_results(result, index):
        results[index] = result
        progress_bar.step(1)
    # Using Pool with apply_async for tracking progress
    with Pool() as pool:
        for index, data in enumerate(cut_data[1:]):
            pool.apply_async(compute_pearson_correlation, args=(data, cut_data[ref_coil]), callback=lambda result, idx=index: collect_results(result, idx))

        pool.close()
        pool.join()
    
    L_time = np.linspace((-L_interval[1] + L_interval[0]) / 2, (L_interval[1] - L_interval[0]) / 2, int(np.shape(results[0])[0]))

    return L_time, results


def parallel_correlation(shot_no: int, L_interval: list[float], ref_coil: int, ma_win_len: int = 10) -> list[list[float], list[list[float]]]:
    # loads needed data
    raw_data = get_mirnov_data(shot_no)
    cut_data = select_time_interval(raw_data, L_interval[0], L_interval[1]).transpose().to_numpy()
    # applies moving average over data
    for coil_id in range(1, 17):
        cut_data[coil_id] = moving_average(cut_data[coil_id], ma_win_len)
    # fixes ref_data argument
    correlation_part = partial(compute_pearson_correlation, ref_data=cut_data[ref_coil])
    # parallelizes computation
    with Pool(processes=16) as pool:
        results = pool.map(correlation_part, cut_data[1:])

        pool.close()
        pool.join()
    
    L_time = np.linspace((-L_interval[1] + L_interval[0]) / 2, (L_interval[1] - L_interval[0]) / 2, int(np.shape(results[0])[0]))

    return L_time, results


def q_profile(shot_no: int, nu: int):
    data_Bt = get_Bt_selected_data(shot_no)
    data_Ip = get_Ip_selected_data(shot_no)
    data_Ip['Ip'] *= 1000

    if nu == 1:
        j_0 = 2 * data_Ip['Ip'] / np.pi / a ** 2
    elif nu == 2:
        j_0 = 3 * data_Ip['Ip'] / np.pi / a ** 2
    else:
        return None

    r = np.linspace(0.001, a - 0.001, 850)
    q = []
    for i in range(0, len(data_Bt['t'])):
        q.append((2 * (nu + 1)) / (mu_0 * j_0[i]) * (data_Bt['Bt'][i] / R) * (r ** 2 / a ** 2) / (1 - (1 - r ** 2 / a ** 2) ** (nu + 1)))

    return data_Bt['t'], r, q


def q_dynamic_profile(shot_no: int, nu: int):
    data_Bt = get_Bt_selected_data(shot_no)
    data_Ip = get_Ip_selected_data(shot_no)
    data_Ip['Ip'] *= 1000
    data_deltaR = get_R_data(shot_no)

    data_R = np.interp(data_Ip['t'], data_deltaR['t'], data_deltaR['R']) / 1000
    data_a = -np.subtract(np.abs(data_R), a)

    i = 0
    while any(np.isnan(data_a)):
        if np.isnan(data_a[i]):
            i += 1
        else:
            for j in range(i):
                data_a[j] = data_a[i]
            i = 0


    j_0 = []
    for i in range(0, len(data_a)):
        if nu == 1:
            j_0.append(float(2 * data_Ip['Ip'][i] / np.pi / data_a[i] ** 2))
        elif nu == 2:
            j_0.append(float(3 * data_Ip['Ip'][i] / np.pi / data_a[i] ** 2))
        else:
            return None

    r = []
    q = []
    for i in range(0, len(data_Bt['t'])):
        r.append(np.linspace(0.001, data_a[i] - 0.001, 850))
        q.append((2 * (nu + 1)) / (mu_0 * j_0[i]) * (data_Bt['Bt'][i] / R) * (r[-1] ** 2 / data_a[i] ** 2) / (1 - (1 - r[-1] ** 2 / data_a[i] ** 2) ** (nu + 1)))


    return r, q




class VerticalNavigationToolbar2Tk(NavigationToolbar2Tk):
    def __init__(self, canvas, window):
        super().__init__(canvas, window, pack_toolbar=False)

    # override _Button() to re-pack the toolbar button in vertical direction
    def _Button(self, text, image_file, toggle, command):
        b = super()._Button(text, image_file, toggle, command)
        b.pack(side=tk.TOP) # re-pack button in vertical direction
        return b

    # override _Spacer() to create vertical separator
    def _Spacer(self):
        s = tk.Frame(self, width=26, relief=tk.RIDGE, bg='DarkGray', padx=2)
        s.pack(side=tk.TOP, pady=5) # pack in vertical direction
        return s

    # disable showing mouse position in toolbar
    def set_message(self, s):
        pass