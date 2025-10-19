import pandas as pd
import numpy as np
import scipy as sp
import pywt

import matplotlib.pyplot as plt

from scipy.integrate import cumulative_trapezoid
from scipy.signal import savgol_filter


def analyse_data(raw_data):
    """Integruje signál z Rogowského cívky a odstraní offset."""
    # odstranění offsetu (fit polynomem 0. stupně)
    trend = np.mean(raw_data['U'][:-1000])
    print(f"trend = {trend:.3e}")
    
    raw_data_detrended = raw_data.copy()
    raw_data_detrended['U'] = raw_data['U'] - trend
    
    # vyhlazení signálu
    raw_data['U'] = weighted_moving_average(raw_data['U'], 100)

    # numerická integrace
    integral = cumulative_trapezoid(raw_data_detrended['U'], raw_data_detrended['t'], initial=0)
    integrated_data = pd.DataFrame({'t': raw_data_detrended['t'], 'I': integral})
    
    return raw_data, integrated_data

def weighted_moving_average(x, N):
    w = np.hanning(N)
    #w = sp.signal.windows.boxcar(N)
    w = w / w.sum()
    return np.convolve(x, w, mode='same')

def moving_average(x, N):
    kernel = np.ones(N)/N
    return np.convolve(x, kernel, mode='same')  # centered approximate

def rms(x): return np.sqrt(np.mean(x**2))

def compute_snr(signal, noise_est):
    return 20*np.log10(rms(signal)/rms(noise_est))

def bandpass_filter(raw):
    fs = 1000  # vzorkovací frekvence, upravte
    # Notch 50 Hz
    f0 = 50.0
    Q = 30.0
    b_notch, a_notch = sp.signal.iirnotch(f0, Q, fs)
    filtered = sp.signal.filtfilt(b_notch, a_notch, raw)
    
    # Bandpass (např. zachovat 1-300 Hz)
    low, high = 1, 300
    b, a = sp.signal.butter(4, [low/(fs/2), high/(fs/2)], btype='band')
    bandpassed = sp.signal.filtfilt(b, a, filtered)
    return bandpassed

def wavelet_denoising(raw):
    coeffs = pywt.wavedec(raw, 'db4', level=6)
    # prahování koeficientů (soft threshold)
    sigma = np.median(np.abs(coeffs[-1]))/0.6745
    uthresh = sigma * np.sqrt(2*np.log(len(raw)))
    coeffs_thresh = [pywt.threshold(c, uthresh, mode='soft') if i>0 else c for i,c in enumerate(coeffs)]
    denoised = pywt.waverec(coeffs_thresh, 'db4')
    return denoised

''' MAIN '''

saveFig = 0
fileFormat = 'png'

# experimenty
min_i = 4
max_i = 4

# kanály
min_j = 1
max_j = 4

# počáteční čas
t0 = 30

for i in range(min_i, max_i + 1):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    for j in range(min_j, max_j + 1):
        filename = f'data/owon{i:04d}CH{j}.csv'

        # načtení dat
        raw_data = pd.read_csv(filename, skiprows=20, names=['number', 't', 'digital', 'U'])
        raw_data['t'] = raw_data['t'] * 1e-6
        raw_data['U'] = raw_data['U'] * 1e-3

        # posun časové osy
        raw_data['t'] = raw_data['t'] - raw_data['t'].iloc[0]

        # ořez časového intervalu
        raw_data = raw_data[raw_data['t'] >= t0]

        # vyhlazení signálu
        N = 50
        #raw_data['U'] = weighted_moving_average(raw_data['U'], N)
        #raw_data['U'] = moving_average(raw_data['U'], N)
        #raw_data['U'] = savgol_filter(raw_data['U'], window_length=N+1, polyorder=3, mode='mirror')
        
        # vyhlazení pomocí SNR
        # raw_data['U'] = compute_snr(signal=raw_data['U'][42000:68000], noise_est=raw_data['U'][30000:42000])
        
        # bandpass filter
        # raw_data['U'] = bandpass_filter(raw_data['U'])
        
        # vyhlazení pomocí wavelet
        raw_data['U'] = wavelet_denoising(raw_data['U'])
        
        # analyse
        raw_data, measured = analyse_data(raw_data)

        # --- horní subplot: raw signály ---
        ax1.plot(raw_data['t'], raw_data['U'], label=f'CH{j}')

        # --- dolní subplot: integrované signály ---
        ax2.plot(measured['t'], measured['I'], label=f'CH{j}')

    # --- nastavení os a titulků ---
    ax1.set_title(f'Experiment #{i} – raw voltage signals')
    ax1.set_ylabel('U (V)')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title(f'Experiment #{i} – integrated signals')
    ax2.set_xlabel('t (s)')
    ax2.set_ylabel('∫U dt (V·s)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if saveFig == 1:
        plt.savefig(f'results/Raw_and_Integrated_{i:04d}.{fileFormat}', dpi=300)

    plt.show()

