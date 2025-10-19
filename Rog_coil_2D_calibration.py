import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize_scalar
from scipy.signal import butter, filtfilt, spectrogram


def analyse_data(raw_data, reference_data):
    """Integruje signál z Rogowského cívky a odstraní offset."""
    # odstranění offsetu (fit polynomem 0. stupně)
    # n = 0
    # coeffs = np.polyfit(raw_data['t'].values, raw_data['U'].values, n)
    # trend = np.polyval(coeffs, raw_data['t'].values)
    trend = np.mean(raw_data['U'][:-1000])
    print(f"trend = {trend}")
    
    raw_data_detrended = raw_data.copy()
    raw_data_detrended['U'] = raw_data['U'] - trend

    # numerická integrace
    integral = cumulative_trapezoid(raw_data_detrended['U'], raw_data_detrended['t'], initial=0)
    integrated_data = pd.DataFrame({'t': raw_data_detrended['t'], 'I_raw': integral})
    
    return integrated_data


def compute_calibration_constant(measured, reference, geom_factor):
    """
    Najde kalibrační konstantu k tak, aby platilo:
      k * geom_factor * I_raw ≈ I_reference
    (tj. funkce minimalizuje rozdíl mezi k*(geom_factor*I_raw) a I_ref).
    """
    ref_interp = np.interp(measured['t'], reference['t'], reference['I'])

    def objective(k):
        return np.mean((k * geom_factor * measured['I_raw'] - ref_interp) ** 2)

    res = minimize_scalar(objective)
    return res.x

def find_calibration_constant(measured, reference, geom_factor):
    """
    Najde k tak, aby maximum |k * geom_factor * I_raw| odpovídalo max{|I_ref|}.
    Vrací nan pokud geom_factor je 0 nebo měřený extrém je 0.
    """
    if geom_factor == 0 or np.isclose(geom_factor, 0.0):
        return np.nan
    
    ref_max_abs = np.max(np.abs(reference['I']))

    # index maxima v absolutní hodnotě u měřeného signálu
    idx_max_meas = np.argmax(np.abs(measured['I_raw']))
    meas_extreme_value = measured['I_raw'].iloc[idx_max_meas]

    # výpočet kalibrační konstanty se znaménkem
    if meas_extreme_value == 0:
        return np.nan

    k = np.sign(meas_extreme_value) * ref_max_abs / (geom_factor * np.abs(meas_extreme_value))
    return k

def bandpass_filter(signal, fs, lowcut=100, highcut=5000, order=4):
    nyq = 0.5 * fs
    highcut = min(highcut, 0.99 * nyq)  # zajistí, že highcut < Nyquist
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, signal)

def remove_error_signal(raw_data, fs=None, pre_trigger=3):
    """
    Odstraní chybnou úvodní část signálu na základě detekce indukovaného napětí.

    Parametry:
        raw_data: pd.DataFrame s poli 't' (čas v sekundách) a 'U' (napětí)
        fs: volitelně vzorkovací frekvence [Hz], pokud ji nelze odvodit z dat
        pre_trigger: kolik sekund před t1 má začínat očištěný signál

    Vrací:
        clean_data: pd.DataFrame s poli 't', 'U' (od t1 - pre_trigger dál)
    """

    t = raw_data['t'].values
    U = raw_data['U'].values

    # --- 1️⃣ Odhad vzorkovací frekvence ---
    if fs is None:
        dt = np.median(np.diff(t))
        fs = 1.0 / dt

    # --- 2️⃣ Urči průměr U_m z posledních 2 sekund ---
    last_segment = U[t >= (t[-1] - 2)]
    if len(last_segment) == 0:
        last_segment = U[-int(2 * fs):]  # fallback
    U_m = np.mean(np.abs(last_segment))

    # --- 3️⃣ Najdi všechny indexy, kde U > 1.5 * U_m ---
    over_idx = np.where(U > 1.5 * U_m)[0]
    if len(over_idx) < 2:
        print("⚠️ Nenalezeny alespoň dva výskyty U > 1.5*U_m.")
        return raw_data

    best_pair = None
    max_distance = 0

    # --- 4️⃣ Pro všechny dvojice (t1, t2) hledej tu nejlepší ---
    for i in range(len(over_idx)):
        for j in range(i + 1, len(over_idx)):
            t1 = t[over_idx[i]]
            t2 = t[over_idx[j]]
            dt_pair = t2 - t1

            # Podmínky na rozestup
            if not (2 <= dt_pair <= (t[-1] - t[0])):
                continue

            # Najdi body t3 mezi t1 a t2, kde abs(U - U_m) < 0.1*U_m
            mask_between = (t >= t1) & (t <= t2)
            t3_candidates = t[mask_between & (np.abs(U - U_m) < 0.1 * U_m)]

            if len(t3_candidates) == 0:
                continue  # žádný t3 → zamítnout

            # Pokud existují dva t3 body vzdálené více než 2 s → zamítnout
            if (np.max(t3_candidates) - np.min(t3_candidates)) > 2:
                continue

            # Pokud je tato dvojice nejdál od sebe, vyber ji
            if dt_pair > max_distance:
                best_pair = (t1, t2)
                max_distance = dt_pair

    if best_pair is None:
        print("⚠️ Nenalezena vhodná dvojice (t1, t2) splňující všechny podmínky.")
        return raw_data

    t1, t2 = best_pair

    # --- 5️⃣ Urči čas začátku platných dat (t1 - pre_trigger s) ---
    start_time = max(t1 - pre_trigger, t[0])
    mask_valid = t >= start_time

    t_clean = t[mask_valid]
    U_clean = U[mask_valid]

    print(f"U_m = {U_m:.3f}")
    print(f"t1 = {t1:.3f}, t2 = {t2:.3f}, Δt = {t2 - t1:.3f} s")
    print(f"Start time: {start_time:.3f} s (t1 - {pre_trigger} s)")

    clean_data = pd.DataFrame({'t': t_clean, 'U': U_clean})
    return clean_data

def find_spectrum(raw_data):
    # --- vstupní data ---
    t = raw_data['t']
    U = raw_data['U']
    
    # --- vzorkovací frekvence ---
    dt = np.mean(np.diff(t))
    fs = 1 / dt  # Hz
    
    # --- výpočet spektrogramu ---
    f, t_spec, Sxx = spectrogram(U, fs=fs, nperseg=1024, noverlap=512, scaling='density')
    
    # --- nalezení dominantních frekvencí v čase ---
    # dominantní frekvence = frekvence s maximální energií v každém sloupci Sxx
    dominant_freqs = f[np.argmax(Sxx, axis=0)]
    
    # --- vykreslení spektrogramu ---
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='auto')
    plt.plot(t_spec, dominant_freqs, color='r', linewidth=1.5, label='Dominant frequency')
    plt.colorbar(label='Power spectral density [dB]')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('Spectrogram of induced voltage signal')
    plt.legend()
    plt.tight_layout()
    plt.show()


''' MAIN '''

saveFig = 0
fileFormat = 'png'

# experimenty
min_i = 27
max_i = 27

# kanály
min_j = 1
max_j = 4    # čísla kanálů, které skutečně čteš (1..2)

# počáteční čas
t0 = 0

# čísla kalibrovaných cívek v těchto kanálech (délka musí být >= max_j)
# 0 = Rogowského pásek, jinak položka = index fyzické Mirnovovy cívky (1..16)
coil_no = [5,1,4,8]

# kontrola délky
if len(coil_no) < max_j:
    raise ValueError("coil_no musí mít alespoň tolik položek, kolik je kanálů (max_j).")

# parametry tokamaku
a = 0.085 # poloměr, kde jsou Mirnovovy cívky umístěny (m)

# vytvoření slovníku coil_positions podle coil_no (bez pevného indexování)
coil_positions = {}
for ch_idx, cno in enumerate(coil_no, start=1):
    if cno == 0:
        # Rogowski: pozice není relevantní, typ 'rogowski'
        coil_positions[ch_idx] = {'x': 0.0, 'y': 0.0, 'angle': None, 'type': 'rogowski'}
    else:
        # poloha cívky na kružnici (1 = nahoře, krok π/8)
        theta = np.pi/2 - (cno - 1) * np.pi/8
        # konvence (poloidální osa = theta - pi/2)
        coil_angle = theta - np.pi/2
        # normalizace do (-pi, pi]
        coil_angle = (coil_angle + np.pi) % (2 * np.pi) - np.pi

        coil_positions[ch_idx] = {
            'x': a * np.cos(theta),
            'y': a * np.sin(theta),
            'angle': coil_angle,
            'type': 'mirnov_poloidal'
        }

wire_position = {'x': 0.0, 'y': 0.0}


for i in range(min_i, max_i + 1):
    for j in range(min_j, max_j + 1):
        
        filename = f'data/owon{i:04d}CH{j}.csv'
        
        # načtení dat
        raw_data = pd.read_csv(filename, skiprows=20, names=['number', 't', 'digital', 'U'])
        raw_data['t'] = raw_data['t']*1e-6
        raw_data['U'] = raw_data['U']*1e-3
        
        # posun časové osy tak, aby začínala od nuly
        raw_data['t'] = raw_data['t'] - raw_data['t'].iloc[0]
        
        # ořez error signálu
        # raw_data = remove_error_signal(raw_data)
        
        # ořez časového intervalu
        mask = raw_data['t'] >= t0
        raw_data = raw_data[mask]
        
        # --- vstupní data ---
        # find_spectrum(raw_data)

        
        # # bandpass filter
        # fs = 1 / np.mean(np.diff(raw_data['t']))  # vzorkovací frekvence
        # raw_data['U'] = bandpass_filter(raw_data['U'], fs, lowcut=50, highcut=2000)
        
        # referenční signál
        plt.plot(raw_data['t'], raw_data['U'], label=f'CH{j} raw')
        
    plt.xlabel('t (s)')
    plt.ylabel('U (V)')
    plt.title(f'Experiment #{i:04d}')
    plt.legend()
    plt.grid(True)
    
    plt.show()
        
for i in range(min_i, max_i + 1): 
    
    plt.figure(figsize=(10, 6))
    calibration_constants = []
    
    # pro výpis
    for j in range(min_j, max_j + 1):
        
        filename1 = f'data/owon{i:04d}CH{j}'
        filename2 = f'data/magnets_current_2025-10-08_{i:04d}.txt'
        
        # načtení dat
        reference_data = pd.read_csv(filename2, skiprows=2, sep=' ', names=['t', 'I'])
        raw_data = pd.read_csv(f"{filename1}.csv", skiprows=20, names=['number', 't', 'digital', 'U'])
        raw_data['t'] = raw_data['t'] * 1e-6
        raw_data['U'] = raw_data['U'] * 1e-3
        
        # posun časové osy tak, aby začínala od nuly
        raw_data['t'] = raw_data['t'] - raw_data['t'].iloc[0]
        reference_data['t'] = reference_data['t'] + t0
        
        # ořez error signálu
        # raw_data = remove_error_signal(raw_data)
        
        # ořez časového intervalu
        mask = raw_data['t'] >= t0
        raw_data = raw_data[mask]
        
        # # bandpass filter
        # fs = 1 / np.mean(np.diff(raw_data['t']))  # vzorkovací frekvence
        # raw_data['U'] = bandpass_filter(raw_data['U'], fs, lowcut=50, highcut=2000)
        
        # geometrická korekce pro Mirnovku
        if j not in coil_positions:
            raise KeyError(f"Channel {j} není definován v coil_positions.")
            
        coil = coil_positions[j]
        
        if coil['type'] == 'rogowski':
            
            geom_factor = 1.0
        
        else:
            dx = coil['x'] - wire_position['x']
            dy = coil['y'] - wire_position['y']
            r = np.hypot(dx, dy)
            print(f'vzdálenost {r} m')
            
            if np.isclose(r, 0.0):
                geom_factor = np.nan
            else:
                phi = np.arctan2(dy, dx)
                B_dir = phi + np.pi/2
                # tangenciální směr pole kolem vodiče
                alpha = coil['angle'] - B_dir
                geom_factor = np.abs(np.cos(alpha) / r) # = projekce / 1/r
                
        # zpracování
        measured = analyse_data(raw_data, reference_data)
        
        # spočti k (vrací k tak, že k * geom_factor * I_raw ≈ I_ref)
        k = find_calibration_constant(measured, reference_data, geom_factor)
        calibration_constants.append(k)
        
        # aplikace kalibrace (POZOR: geom_factor je součást kalibrace)
        measured['I_calibrated'] = k * geom_factor * measured['I_raw']
        # k = compute_calibration_constant(measured, reference_data, geom_factor) # uses linear fit
        
        # vykreslení kanálu
        plt.plot(measured['t'], measured['I_calibrated'], label=f'CH{j} (k={k:.3e}, g={geom_factor:.3e})')
        # referenční signál
        plt.plot(reference_data['t'], reference_data['I'], 'k--', label='reference')
        
    plt.xlabel('t (s)')
    plt.ylabel('I (A)')
    plt.title(f'Experiment #{i:04d}')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    if saveFig == 1:
        plt.savefig(f'results/Calibration_{i:04d}.{fileFormat}', dpi=300)
        
    plt.show()


