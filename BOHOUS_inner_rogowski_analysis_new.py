import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from scipy import constants, integrate, signal, interpolate
from scipy.optimize import curve_fit
import sqlalchemy   # high-level library for SQL in Python
from scipy import signal as sigproc
from scipy.fft import next_fast_len
import math
import pandas as pd
import scipy as spi
from numpy.lib.npyio import DataSource


from functions import get_plasma_time_parameters, get_mirnov_data, get_diamagnet_data, get_rogowski_data



ds = np.lib.npyio.DataSource('/tmp')  # temporary storage for downloaded files


address_new_diag = 'http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/MHDring-TM/{name}'# new MHD ring
address_basic = 'http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/BasicDiagnostics/Results/{name}'
address_basic_raw= "http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/PlasmaDetection/{name}"
scalars_URL = 'http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/PlasmaDetection/Results/{name}'
new= "http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/MHDring-TM/{name}"


''' FUNCTIONS '''
def get_scalar(shot_no, name):
    return float(ds.open(scalars_URL.format(shot_no=shot_no, name=name)).read())

def open_remote(shot_no, name, url_template=address_basic):
    return ds.open(url_template.format(shot_no=shot_no, name=name))

def open_new(shot_no, name, url_template=new):
    return ds.open(url_template.format(shot_no=shot_no, name=name))

def open_remote_raw(shot_no, name, url_template=address_basic_raw+ ".csv"):
    return ds.open(url_template.format(shot_no=shot_no, name=name))

def open_new_diag(shot_no, name, url_template=address_new_diag + '.csv'):
    return ds.open(url_template.format(shot_no=shot_no, name=name))

def read_signal(shot_no, name): 
    file = open_remote(shot_no, name, address_basic + '.csv')
    signal= pd.read_csv(file, names=['Time [s]',name], index_col='Time [s]')  # squeeze makes simple 1-column signals a Series
    return signal[name]

def read_signal_new(shot_no, name): 
    file = open_remote(shot_no, name, new + '.csv')
    signal= pd.read_csv(file, names=['Time [s]',name], index_col='Time [s]')  # squeeze makes simple 1-column signals a Series
    return signal[name]

def read_signal_raw(shot_no, name): 
    file = open_remote(shot_no, name, address_basic_raw + '.csv')
    signal= pd.read_csv(file, names=['Time [s]',name], index_col='Time [s]')  # squeeze makes simple 1-column signals a Series
    return signal[name]

def read_new_diag(shot_no, name): 
    file = open_new_diag(shot_no, name, address_new_diag + '.csv')
    signal= pd.read_csv(file, names=['Time [s]',name], index_col='Time [s]')  # squeeze makes simple 1-column signals a Series
    return signal[name]

def remove_offset(shot_no, raw_series):
    ''' computes offset fit and substructs it from raw signal
    
        shot_no = number of the shot in Golem database
        raw_data = pd.DataFrame of raw signal from Rogowski coils
    '''
    # fit for off-set subtruction
    n = 0  # degree of fit polynomial
    
    # t = index, y = values
    t = raw_series.index.values
    y = raw_series.values
    
    coeffs = np.polyfit(t, y, n)
    trend = np.polyval(coeffs, t)
    
    return pd.Series(y - trend, index=t, name=raw_series.name)

''' MAIN '''
vac_shot_no=50247
shot_no=50301



t_plasma_start, t_plasma_end=get_plasma_time_parameters(shot_no)
plasma_endpoints= [t_plasma_start,t_plasma_end]

mirnov_data=get_mirnov_data(shot_no)
diamagnet_data=get_diamagnet_data(shot_no)
rogowski_data=get_rogowski_data(shot_no)

HFS=pd.Series(data=mirnov_data.iloc[:, 13].values, index=mirnov_data.iloc[:, 0], name='U_Bt_HFS')
LFS=pd.Series(data=mirnov_data.iloc[:, 5].values, index=mirnov_data.iloc[:, 0], name='U_Bt_LFS')
inner_coil = pd.Series(data=diamagnet_data.iloc[:, 1].values, index=diamagnet_data.iloc[:, 0], name='U_diam_inner')
outer_coil = pd.Series(data=diamagnet_data.iloc[:, 2].values, index=diamagnet_data.iloc[:, 0], name='U_diam_outer')
Ip=-pd.Series(data=rogowski_data.iloc[:, 1].values, index=rogowski_data.iloc[:, 0], name='U_rogowski')

# inner_coil = remove_offset(shot_no, inner_coil)
# outer_coil = remove_offset(shot_no, outer_coil)
Ip = remove_offset(shot_no, Ip)

### VACUUM SHOT
mirnov_data_vac=get_mirnov_data(vac_shot_no)
diamagnet_data_vac=get_diamagnet_data(vac_shot_no)
rogowski_data_vac=get_rogowski_data(vac_shot_no)

HFS_vac=pd.Series(data=mirnov_data_vac.iloc[:, 13].values, index=mirnov_data_vac.iloc[:, 0], name='U_Bt_HFS')
LFS_vac=pd.Series(data=mirnov_data_vac.iloc[:, 5].values, index=mirnov_data_vac.iloc[:, 0], name='U_Bt_LFS')
inner_coil_vac = pd.Series(data=diamagnet_data_vac.iloc[:, 1].values, index=diamagnet_data_vac.iloc[:, 0], name='U_diam_inner')
outer_coil_vac = pd.Series(data=diamagnet_data_vac.iloc[:, 2].values, index=diamagnet_data_vac.iloc[:, 0], name='U_diam_outer')
Ip_vac=-pd.Series(data=rogowski_data_vac.iloc[:, 1].values, index=rogowski_data_vac.iloc[:, 0], name='U_rogowski')

# inner_coil_vac = remove_offset(vac_shot_no, inner_coil_vac)
# outer_coil_vac = remove_offset(vac_shot_no, outer_coil_vac)
Ip_vac = remove_offset(vac_shot_no, Ip_vac)

## SIGNALS FROM OLD DIAGNOSTIC
I_p=read_signal(shot_no,"Ip")
Ip_sig=read_signal_raw(shot_no,"U_RogCoil")
U_loop=read_signal(shot_no,"U_loop")
Bt=read_signal(shot_no,"Bt")
U_Bt= read_signal_raw(shot_no,"U_BtCoil")
Ich_vac= read_signal(vac_shot_no, "Ipch" )
Ich= read_signal(shot_no, "Ich" )
U_Bt_vac= read_signal_raw(vac_shot_no,"U_BtCoil")
Ip_sig_vac=read_signal_raw(vac_shot_no,"U_RogCoil")
U_loop_vac=read_signal(vac_shot_no,"U_loop")
#Ich_vac=pd.to_numeric(Ich_vac, errors="coerce")
Bt_vac= read_signal(vac_shot_no,"Bt")


#Calibration signals

#Ip_sig_old= read_signal_raw(45165, "U_RogCoil")
#UBt_sig_old= read_signal_raw(45165, "U_BtCoil")


## Function TO FIND INDICES OF THE START AND END OF PLASMA
def find_indices_for_time_range(series, start_time, end_time):
    float_indices = pd.Series(series.index.astype(float))
    start_index = (float_indices - start_time).abs().idxmin()
    end_index = (float_indices - end_time).abs().idxmin()

    return start_index, end_index

duration_index=find_indices_for_time_range(Ip, plasma_endpoints[0],plasma_endpoints[1])


## ROGOWSKI COIL RAW SIGNAL INTEGRATION FOR BOTH VACUUM SHOT AND PLASMA SHOT #N=1974 turns
Current_vac= pd.Series(-integrate.cumulative_trapezoid(Ip_vac, Ip_vac.index, initial=0)*1.2566/(1000*2*np.pi*1974*0.005*0.005*4*np.pi*1e-7), index=Ip_vac.index)
Current_pl=pd.Series(-integrate.cumulative_trapezoid(Ip, Ip.index, initial=0)*1.2566/(1000*2*np.pi*1974*0.005*0.005*4*np.pi*1e-7),index=Ip.index)


#RLC Model for to remove offset and toroidal field pick up by Rogowski Coil
def model_function(t, a1, a2, a3):
    return a1 * np.exp(-a2 * t) * np.sin(a3 * t)


#SELECTION OF DATA TO BE CLEANED AND MODELLED
x = Current_pl.index[np.r_[2000:duration_index[0]-500, duration_index[1]+1000:len(Current_pl)]] #Selecting non plasma portions
y = Current_pl.iloc[np.r_[2000:duration_index[0]-500, duration_index[1]+1000:len(Current_pl)]]
x_clean = x[np.isfinite(x) & np.isfinite(y)] # To remove infs
y_clean = y[np.isfinite(x) & np.isfinite(y)]
t_data_points= x_clean
I_data_points=y_clean*1000 #To convert Current from kA to A. 


# DOING THE FIT
initial_guess = [0, 2, 30]
params, covariance = curve_fit(model_function, t_data_points, I_data_points, p0=initial_guess,maxfev=10000)

a1_fit, a2_fit, a3_fit = params
print(f"Fitted parameters: a1 = {a1_fit}, a2 = {a2_fit}, a3 = {a3_fit}")

# fitted data
t_fit = Current_pl.index
I_fit = model_function(t_fit, *params)

plt.figure(figsize=(10, 6))
for x in plasma_endpoints:
    plt.axvline(x=x, color='black', linestyle='--') 
plt.plot(t_data_points, I_data_points, 'bo', label='I - Data')
plt.plot(t_fit, I_fit, 'r-', label=f'$I = {a1_fit}\  exp(-{a2_fit} t) \ \sin({a3_fit}t)$')
plt.xlabel('Time [s]')
plt.title(f"#{shot_no}")
plt.ylabel('I [A]')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("fit_image.png")


#Subtracting Bt Cross-Talk
correct_current= Current_pl*1000-(I_fit)


### PLOTTING ALL CURRENT
plt.figure()
#plt.plot(Ip.index,-integrate.cumulative_trapezoid(Ip, Ip.index, initial=0)*1000*
          #1.2566/(1000*2*np.pi*1974*0.005*0.005*4*np.pi*1e-7), label="New Inner Rogowski-uncorrected",color="red")
plt.plot(I_p.index/1000, I_p*1000, label="Old Outer Rogowski", color="blue")
plt.plot(correct_current.index, correct_current, label="New Rogowski - RLC-Fit corrected", color="brown")

for x in plasma_endpoints:
    plt.axvline(x=x, color='black', linestyle='--') 
plt.xlabel("Time [s]")
plt.ylabel('$I_\mathrm{p}$ [A]')
plt.legend()
plt.xlim(0)
#plt.ylim(0)
plt.grid(True)
plt.title(f"#{shot_no}")
plt.savefig("gggg.png")


# REMODELLING CHAMBER CURRENT 
#The idea is to subtract the inner Rogowski measurement from the outer Rogowski measurement to get the current flowing through the chamber
#For Offset Removal from the outer Rogowski signal
t_Bt=0  
t_CD=1000e-6
offset_sl = slice(None, min(t_Bt, t_CD) - 1e-4)
Ip_sig-=Ip_sig[offset_sl].mean()

OuterCurr= -5300.0*integrate.cumulative_trapezoid(Ip_sig, Ip_sig.index, initial=0)
InnerCurr=-1.2566/(1000*2*np.pi*1974*0.005*0.005*4*np.pi*1e-7)*integrate.cumulative_trapezoid(Ip, Ip.index, initial=0)

# Re-Indexing of Data By Interpolation
data_1 = OuterCurr*1000
data_2 = correct_current

common_time = np.linspace(0, 0.024, 24000)

old_rog_interp= pd.Series(np.interp(common_time, Ip_sig.index, data_1), index= common_time)
new_rog_interp = pd.Series(np.interp(common_time, Ip.index, data_2[:len(Ip.index)]), index= common_time)  

# New Chamber Current
Ich_new= (old_rog_interp-new_rog_interp)


#Here you have to pick from a series of shots made with only B_toroidal. 
#Select the shot with the same UBt as you are using for your own shots

#shot_only_Bt=45154  # 100V
#shot_only_Bt=45155  # 200V
#shot_only_Bt=45156  # 300V
#shot_only_Bt=45157  # 400V
#shot_only_Bt=45158  # 500V
#shot_only_Bt=45159  # 600V
#shot_only_Bt=45160  # 700V
#shot_only_Bt=45161  # 800V
shot_only_Bt=45162  # 900V
#shot_only_Bt=45163  # 1000V
#shot_only_Bt=45164 # 1100V
#shot_only_Bt=45165 # 1200V


#Calibration signals
outer_rog_sig_only_Bfield= read_signal_raw(shot_only_Bt, "U_RogCoil")
outer_UBt_sig_only_Bfield= read_signal_raw(shot_only_Bt, "U_BtCoil")

# Outer Rogowski Coil Pick-up Calibration Signals Offset Removal
outer_rog_sig_only_Bfield-=outer_rog_sig_only_Bfield[offset_sl].mean()
outer_UBt_sig_only_Bfield-=outer_UBt_sig_only_Bfield[offset_sl].mean()

# Integrating the signals
I_only_Bfield= pd.Series(-5300.0*integrate.cumulative_trapezoid(outer_rog_sig_only_Bfield, outer_rog_sig_only_Bfield.index, initial=0), index=outer_rog_sig_only_Bfield.index)
Bt_only_Bfield= pd.Series(70.42*integrate.cumulative_trapezoid(outer_UBt_sig_only_Bfield, outer_UBt_sig_only_Bfield.index, initial=0), index=outer_UBt_sig_only_Bfield.index)

plt.figure()
plt.plot(I_only_Bfield.index, I_only_Bfield*1000, label="$B_{tor}$ cross-talk on Outer Rogowski ", color="red")
plt.xlabel("Time [s]")
plt.ylabel("$I$ [A]")
plt.legend()
plt.xlim(0)
#plt.ylim(0)
plt.grid(True)
plt.title(f"#{shot_only_Bt}")


## Re-Indexing the data for Chamber current correction
#U_loop= U_loop.to_numpy()
I_only_Bfield= I_only_Bfield.to_numpy()
Bt_only_Bfield=Bt_only_Bfield.to_numpy()

U_loop_interp= pd.Series(np.interp(common_time,Ich.index/1000, U_loop), index= common_time)

min_len = min(len(Ich.index), len(I_only_Bfield))
time_axis = Ich.index[:min_len] / 1000
I_only_Bfield_cut = I_only_Bfield[:min_len]
I_only_Bfield_interp = pd.Series(np.interp(common_time, time_axis, I_only_Bfield_cut*1000), index=common_time)

min_len_bt = min(len(Ich.index), len(Bt_only_Bfield))
time_axis_bt = Ich.index[:min_len_bt] / 1000
Bt_only_Bfield_cut = Bt_only_Bfield[:min_len_bt]
Bt_only_interp = pd.Series(np.interp(common_time, time_axis_bt, Bt_only_Bfield_cut), index=common_time)

I_chamber_correct=Ich_new - I_only_Bfield_interp

plt.figure()
I_chamber_correct.plot(label="Ich-Bt removed", color="cyan")
(old_rog_interp).plot(label="Outer Rogowski", color="blue")
(new_rog_interp).plot(label="Inner Rogowski", color="red")
(Ich_new).plot(label="New - $I_{ch}$", color="brown")
plt.plot(Ich.index/1000, (Ich*1000),label="Old - $I_{ch}$", color="green" )
for x in plasma_endpoints:
    plt.axvline(x=x, color='black', linestyle='--') 
plt.xlabel("Time [s]")
plt.ylabel("$I_\mathrm{p}$ [A]")
plt.legend()
plt.xlim(0)
#plt.ylim(0)
plt.grid(True)
plt.title(f"#{shot_no}")
#plt.savefig("FullChamber_current"+f"#{shot_no}")


def model_Ich(t, R, L, U_t):
    return U_t / R * (1 - np.exp(-R / L * t))

t_data =Ich_new.index
U_data = U_loop_interp
Ich_data=I_chamber_correct

# Uloop function
def model_I_with_U(t, R, L):
    U_t = np.interp(t, t_data, U_data)  # Interpolate U_data to match t_data
    return model_Ich(t, R, L, U_t)

initial_guess = [9.7e-3, 1e-6] # We use the parameters in the former analysis

popt, pcov = curve_fit(model_I_with_U, t_data, Ich_data, p0=initial_guess)
fitted_R, fitted_L = popt

print(f"Fitted R: {fitted_R}")
print(f"Fitted L: {fitted_L}")

plt.figure()
for x in plasma_endpoints:
    plt.axvline(x=x, color='black', linestyle='--') 
plt.scatter(t_data, Ich_data, label='$I_{ch}$ - Data', color='blue', s=10)
plt.plot(t_data, model_I_with_U(t_data, *popt), label=r'$I_{ch} = \frac{U_L}{R_{ch}} - \frac{U_L}{R_{ch}} e^{-\frac{R_{ch}}{L_{ch}} t}$', color='red', linewidth=2)
plt.xlabel('Time [s]')
plt.ylabel(' $I_\mathrm{ch}$ [A]')
plt.legend()
plt.grid(True)
plt.title(f'#{shot_no}')
plt.show()


# # Local TF Measuring Coils
#The function below process the data from both the TF Measuring Coils as well as the inner Diamagnetic coils
def plot_magnetic_fields(shot_no, vac_shot_no):
    # PATHS
    ds = DataSource('/tmp')  # temporary storage for downloaded files

    address_new_diag = 'http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/MHDring-TM/{name}'# new coils
    address_basic = 'http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/BasicDiagnostics/Results/{name}'
    address_basic_raw= "http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/PlasmaDetection/{name}"
    scalars_URL = 'http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/PlasmaDetection/Results/{name}'
    new= "http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/MHDring-TM/{name}"

    # Functions for reading data

    def get_scalar(shot_no, name):
        return float(ds.open(scalars_URL.format(shot_no=shot_no, name=name)).read())

    def open_remote(shot_no, name, url_template=address_basic):
        return ds.open(url_template.format(shot_no=shot_no, name=name))

    def open_new(shot_no, name, url_template=new):
        return ds.open(url_template.format(shot_no=shot_no, name=name))

    def open_remote_raw(shot_no, name, url_template=address_basic_raw+ ".csv"):
        return ds.open(url_template.format(shot_no=shot_no, name=name))

    def open_new_diag(shot_no, name, url_template=address_new_diag + '.csv'):
        return ds.open(url_template.format(shot_no=shot_no, name=name))

    def read_signal(shot_no, name): 
        file = open_remote(shot_no, name, address_basic + '.csv')
        signal= pd.read_csv(file, names=['Time [s]',name],
                         index_col='Time [s]')  # squeeze makes simple 1-column signals a Series
        return signal[name]

    def read_signal_new(shot_no, name): 
        file = open_remote(shot_no, name, new + '.csv')
        signal= pd.read_csv(file, names=['Time [s]',name],
                         index_col='Time [s]')  # squeeze makes simple 1-column signals a Series
        return signal[name]


    def read_signal_raw(shot_no, name): 
        file = open_remote(shot_no, name, address_basic_raw + '.csv')
        signal= pd.read_csv(file, names=['Time [s]',name],
                         index_col='Time [s]')  # squeeze makes simple 1-column signals a Series
        return signal[name]

    def read_new_diag(shot_no, name): 
        file = open_new_diag(shot_no, name, address_new_diag + '.csv')
        signal= pd.read_csv(file, names=['Time [s]',name],
                         index_col='Time [s]')  # squeeze makes simple 1-column signals a Series
        return signal[name]

    #Read plasma duration
    t_plasma_start, t_plasma_end=get_plasma_time_parameters(shot_no)
    plasma_endpoints= [t_plasma_start,t_plasma_end]

    #Read all signal from new coils
    mirnov_data=get_mirnov_data(shot_no)
    diamagnet_data=get_diamagnet_data(shot_no)
    rogowski_data=get_rogowski_data(shot_no)
    
    HFS_1=pd.Series(data=mirnov_data.iloc[:, 13].values, index=mirnov_data.iloc[:, 0], name='U_Bt_HFS')
    LFS_1=pd.Series(data=mirnov_data.iloc[:, 5].values, index=mirnov_data.iloc[:, 0], name='U_Bt_LFS')
    inner_coil = pd.Series(data=diamagnet_data.iloc[:, 1].values, index=diamagnet_data.iloc[:, 0], name='U_diam_inner')
    outer_coil = pd.Series(data=diamagnet_data.iloc[:, 2].values, index=diamagnet_data.iloc[:, 0], name='U_diam_outer')
    Ip=pd.Series(data=rogowski_data.iloc[:, 1].values, index=rogowski_data.iloc[:, 0], name='U_rogowski')

    # inner_coil = remove_offset(shot_no, inner_coil)
    # outer_coil = remove_offset(shot_no, outer_coil)
    Ip = remove_offset(shot_no, Ip)

    LFS= LFS_1
    HFS= HFS_1
    
    
    #Read signal from Vacuum shot
    mirnov_data_vac=get_mirnov_data(vac_shot_no)
    diamagnet_data_vac=get_diamagnet_data(vac_shot_no)
    rogowski_data_vac=get_rogowski_data(vac_shot_no)
    
    HFS_vac=pd.Series(data=mirnov_data_vac.iloc[:, 13].values, index=mirnov_data_vac.iloc[:, 0], name='U_Bt_HFS')
    LFS_vac=pd.Series(data=mirnov_data_vac.iloc[:, 5].values, index=mirnov_data_vac.iloc[:, 0], name='U_Bt_LFS')
    inner_coil_vac = pd.Series(data=diamagnet_data_vac.iloc[:, 1].values, index=diamagnet_data_vac.iloc[:, 0], name='U_diam_inner')
    outer_coil_vac = pd.Series(data=diamagnet_data_vac.iloc[:, 2].values, index=diamagnet_data_vac.iloc[:, 0], name='U_diam_outer')
    Ip_vac=pd.Series(data=rogowski_data_vac.iloc[:, 1].values, index=rogowski_data_vac.iloc[:, 0], name='U_rogowski')

    # inner_coil_vac = remove_offset(vac_shot_no, inner_coil_vac)
    # outer_coil_vac = remove_offset(vac_shot_no, outer_coil_vac)
    Ip_vac = remove_offset(vac_shot_no, Ip_vac)

    #Read other signals
    Bt=read_signal(shot_no,"Bt")

    #indices of plasma
    def find_indices_for_time_range(series, start_time, end_time):
        float_indices = pd.Series(series.index.astype(float))
        start_index = (float_indices - start_time).abs().idxmin()
        end_index = (float_indices - end_time).abs().idxmin()

        return start_index, end_index

    duration_index=find_indices_for_time_range(Ip, plasma_endpoints[0],plasma_endpoints[1])

    #Process the  Read signal to remove NaNs and infs
    def process_Bt_signal(LFS,HFS):
        HFS = pd.to_numeric(HFS, errors='coerce')  # Convert HFS values to numeric, coercing errors to NaN
        HFS.index = pd.to_numeric(HFS.index, errors='coerce')
        HFS = HFS.dropna()
        LFS = pd.to_numeric(LFS, errors='coerce') 
        LFS.index = pd.to_numeric(LFS.index, errors='coerce') 

        LFS = LFS.dropna()

        common_index = HFS.index.intersection(LFS.index)

        # Reindex both series to the common index
        HFS = HFS.reindex(common_index)
        LFS = LFS.reindex(common_index)

        # Drop any remaining NaNs that may have been introduced by reindexing
        HFS = HFS.dropna()
        LFS = LFS.dropna()

        # Ensure the indices are still aligned after dropping NaNs
        common_index = HFS.index.intersection(LFS.index)
        HFS = HFS.reindex(common_index)
        LFS = LFS.reindex(common_index)

        return LFS, HFS

    #TFS Coils Parameters
    mu= 4*np.pi*1e-7
    D= (8.6+4*0.3)*1e-3
    h=14e-3
    l=np.pi*8.6e-3
    L=0.26053e-3
    A=2*np.pi*(D/2)*h
    #N= round(np.sqrt(l*L/(mu*A)))
    N=150
    Aeff= (2*N*np.pi*D**2)/4
    K=1/Aeff
    K1=1/(172e-4)
    K2=1/(159.2e-4)

    LFS, HFS= process_Bt_signal(LFS,HFS)
    Bt_LFS=pd.Series(-K*integrate.cumulative_trapezoid(LFS, LFS.index, initial=0), index=LFS.index)
    Bt_HFS=pd.Series(-K*integrate.cumulative_trapezoid(HFS, HFS.index, initial=0), index=HFS.index)

    #Diamagnetic Loops
    d_inner= 175e-3
    d_outer= 204e-3
    A_inner= (np.pi* d_inner**2)/4
    A_outer= (np.pi* d_outer**2)/4

    inner_coil, outer_coil=process_Bt_signal(inner_coil,outer_coil)
    inner_coil_vac, outer_coil_vac=process_Bt_signal(inner_coil_vac,outer_coil_vac)

    inner_flux= pd.Series(integrate.cumulative_trapezoid(inner_coil,inner_coil.index, initial=0),index=inner_coil.index, name="inner flux")
    outer_flux= pd.Series(integrate.cumulative_trapezoid(outer_coil,outer_coil.index, initial=0),index=outer_coil.index, name="outer flux")

    inner_flux_vac= pd.Series(integrate.cumulative_trapezoid(inner_coil_vac,inner_coil_vac.index, initial=0),index=inner_coil_vac.index, name="inner flux")
    outer_flux_vac= pd.Series(integrate.cumulative_trapezoid(outer_coil_vac,outer_coil_vac.index, initial=0),index=outer_coil_vac.index, name="outer flux")

    #Perform a model on RLC for the fields
    def model_function(t, a1, a2, a3, c):
        return a1 * np.exp(-a2 * t) * np.sin(a3 * t) +c

    def fitt(Bt_HFS, Bt_LFS, duration_index):

        #For HFS
        x1 = Bt_HFS.index[np.r_[2000:duration_index[0]-500, duration_index[1]+1000:len(Bt_HFS)]]
        y1 = Bt_HFS.iloc[np.r_[2000:duration_index[0]-500, duration_index[1]+1000:len(Bt_HFS)]]
        x1_clean = x1[np.isfinite(x1) & np.isfinite(y1)]
        y1_clean = y1[np.isfinite(x1) & np.isfinite(y1)]
        t1_data= x1_clean
        Bt_HFS_data=y1_clean

        initial_guess = [2, 10, 30,0]

        HFS_params, covariance = curve_fit(model_function, t1_data, Bt_HFS_data, p0=initial_guess, maxfev= 100000000)

        HFS_a1_fit, HFS_a2_fit, HFS_a3_fit, HFS_c_fit = HFS_params
        #print(f"Fitted parameters: a1 = {a1_fit}, a2 = {a2_fit}, a3 = {a3_fit}")

        t1_fit = Bt_HFS.index
        Bt_HFS_fit = model_function(t1_fit, *HFS_params)

        #For LFS
        x2 = Bt_LFS.index[np.r_[2000:duration_index[0]-500, duration_index[1]+1000:len(Bt_LFS)]]
        y2 = Bt_LFS.iloc[np.r_[2000:duration_index[0]-500, duration_index[1]+1000:len(Bt_LFS)]]
        x2_clean = x2[np.isfinite(x2) & np.isfinite(y2)]
        y2_clean = y2[np.isfinite(x2) & np.isfinite(y2)]
        t2_data= x2_clean
        Bt_LFS_data=y2_clean

        LFS_params, covariance = curve_fit(model_function, t2_data, Bt_LFS_data, p0=initial_guess, maxfev= 100000000)

        LFS_a1_fit, LFS_a2_fit, LFS_a3_fit,LFS_c_fit = LFS_params
        #print(f"Fitted parameters: a1 = {a1_fit}, a2 = {a2_fit}, a3 = {a3_fit}")

        # fitted data
        t2_fit = Bt_LFS.index
        Bt_LFS_fit = model_function(t2_fit, *LFS_params)

        return HFS_params,t1_fit, Bt_HFS_fit,t1_data,Bt_HFS_data, LFS_params,t2_fit,Bt_LFS_fit, t2_data,Bt_LFS_data

    HFS_params,t1_fit, Bt_HFS_fit,t1_data,Bt_HFS_data, LFS_params,t2_fit,Bt_LFS_fit, t2_data,Bt_LFS_data=fitt(Bt_HFS, Bt_LFS, 
                                                                                                              duration_index)
    #Print Parameters
    print(HFS_params, LFS_params)
    return inner_coil,outer_coil,inner_coil_vac,outer_coil_vac,inner_flux,outer_flux,inner_flux_vac,outer_flux_vac,Bt,plasma_endpoints,vac_shot_no,shot_no,HFS_1,LFS_1,Bt_HFS,Bt_LFS,HFS_params,t1_fit, Bt_HFS_fit,t1_data,Bt_HFS_data, LFS_params,t2_fit,Bt_LFS_fit, t2_data,Bt_LFS_data


inner_coil,outer_coil,inner_coil_vac,outer_coil_vac,inner_flux,outer_flux,inner_flux_vac,outer_flux_vac,Bt,plasma_endpoints,vac_shot_no,shot_no,HFS_1,LFS_1,Bt_HFS,Bt_LFS,HFS_params,t1_fit, Bt_HFS_fit,t1_data,Bt_HFS_data, LFS_params,t2_fit,Bt_LFS_fit, t2_data,Bt_LFS_data=plot_magnetic_fields(45261,45264)


d_inner= 175e-3
d_outer= 204e-3
A_inner= (np.pi* d_inner**2)/4
A_outer= (np.pi* d_outer**2)/4


# Raw Signal
plt.figure()
plt.plot(HFS_1.index, HFS_1, label='HFS', color="orange")
plt.plot(LFS_1.index, LFS_1, label='LFS', color="blue")
for x in plasma_endpoints:
    plt.axvline(x=x, color='black', linestyle='--')
plt.title(f'Raw Signal-#{shot_no}')
plt.ylabel("Voltage [V]")
plt.xlabel("Time [s]")
plt.grid(True)
plt.legend()

#plt.savefig(f"{shot_no} Raw signals of BT coils.png")
#plt.xlim(0)
#plt.ylim(0)


# Plot toroidal field after processing
plt.figure()
plt.plot(Bt_LFS.index, Bt_LFS, label='Bt LFS', color="blue")
plt.plot(Bt_HFS.index, Bt_HFS, label='Bt HFS', color="orange")
for x in plasma_endpoints:
    plt.axvline(x=x, color='black', linestyle='--')
plt.title('$B_\mathrm{tor}$ - '+f"#{shot_no}")
plt.ylabel("$B_\mathrm{tor}$ [T]")
plt.xlabel("Time [s]")
plt.grid(True)
plt.legend()
#plt.xlim(0)
plt.ylim(0)
plt.savefig(f"{shot_no} B-field BT coils.png")
#plt.xlim(0)
#plt.ylim(0)


# Plot toroidal field after processing
plt.figure()
plt.plot(t2_data, Bt_LFS_data,"ro" ,label='Data')
plt.plot(t2_fit, Bt_LFS_fit, label='RLC Fit',linestyle='-.', color="indigo")
for x in plasma_endpoints:
    plt.axvline(x=x, color='black', linestyle='--')
plt.title('$B_\mathrm{tor}$ -LFS- '+f"#{shot_no}")
plt.ylabel("$B_\mathrm{tor}$ [T]")
plt.xlabel("Time [s]")
plt.grid(True)
plt.legend()
#plt.xlim(0)
plt.ylim(0)
plt.savefig(f"{shot_no} LFS fit.png")


# Plot toroidal field after processing
plt.figure()
plt.plot(t2_fit, Bt_LFS_fit-LFS_params[3],color="deeppink" ,label='Corrected Fit')
plt.plot(t2_fit, Bt_LFS_fit, label='RLC Fit',linestyle='-.', color="indigo")
plt.plot(Bt_LFS.index, Bt_LFS-LFS_params[3], label='Bt LFS -Fit- Corrected', color="blue")
for x in plasma_endpoints:
    plt.axvline(x=x, color='black', linestyle='--')
plt.title('$B_\mathrm{tor}$ -LFS- '+f"#{shot_no}")
plt.ylabel("$B_\mathrm{tor}$ [T]")
plt.xlabel("Time [s]")
plt.grid(True)
plt.legend()
#plt.xlim(0)
#plt.ylim(0)
plt.savefig(f"{shot_no} LFS fit_correction.png")


# Plot toroidal field after processing
plt.figure()
plt.plot(t1_data, Bt_HFS_data,"bo", label='Data')
plt.plot(t1_fit, Bt_HFS_fit, label='RLC Fit', color="magenta",linestyle='-.')
for x in plasma_endpoints:
    plt.axvline(x=x, color='black', linestyle='--')
plt.title('$B_\mathrm{tor}$- HFS - '+f"#{shot_no}")
plt.ylabel("$B_\mathrm{tor}$ [T]")
plt.xlabel("Time [s]")
plt.grid(True)
plt.legend()
#plt.xlim(0)
plt.ylim(0)
plt.savefig(f"{shot_no} HFS fit.png")

plt.figure()
plt.plot(t1_fit,Bt_HFS_fit-HFS_params[3],color="green", label='Corrected Fit')
plt.plot(t1_fit, Bt_HFS_fit, label='RLC Fit', color="magenta",linestyle='-.')
plt.plot(Bt_HFS.index, Bt_HFS-HFS_params[3], label='Bt HFS - Corrected', color="orange")
for x in plasma_endpoints:
    plt.axvline(x=x, color='black', linestyle='--')
plt.title('$B_\mathrm{tor}$- HFS - '+f"#{shot_no}")
plt.ylabel("$B_\mathrm{tor}$ [T]")
plt.xlabel("Time [s]")
plt.grid(True)
plt.legend()
#plt.xlim(0)
#plt.ylim(0)
plt.savefig(f"{shot_no} HFS fit_correction.png")

plt.figure()
plt.plot(t1_fit,Bt_HFS_fit-HFS_params[3],color="green", label='Corrected HFS Fit')
plt.plot(t1_fit, Bt_HFS_fit, label='RLC Fit HFS', color="magenta",linestyle='-.')
plt.plot(Bt_HFS.index, Bt_HFS-HFS_params[3], label='Bt HFS - Corrected', color="orange")
plt.plot(Bt.index/1000, Bt, label="Outer TFS Coil")
plt.plot(t2_fit, Bt_LFS_fit-LFS_params[3],color="deeppink" ,label='Corrected LFS Fit')
plt.plot(t2_fit, Bt_LFS_fit, label='RLC Fit LFS',linestyle='-.', color="indigo")
plt.plot(Bt_LFS.index, Bt_LFS-LFS_params[3], label='Bt LFS -Fit- Corrected', color="blue")
plt.plot(inner_flux.index, (1/A_inner)*inner_flux, label='Inner Diamagnetic', color="red")
plt.plot(outer_flux.index, (1/A_outer)*outer_flux, label='Outer Diamagnetic', color="purple")
for x in plasma_endpoints:
    plt.axvline(x=x, color='black', linestyle='--')
plt.title(f"#{shot_no}")
plt.ylabel("$B_\mathrm{tor}$ [T]")
plt.xlabel("Time [s]")
plt.grid(True)
plt.legend(loc="lower right")
#plt.xlim(0)
##plt.ylim(0)
plt.savefig(f"{shot_no} All fit")


# # Diamagnetic Loops
# Raw Signal of Vacuum Shots
plt.figure()
plt.plot(inner_coil_vac.index, inner_coil_vac, label='Inner Vacuum', color="red")
plt.plot(outer_coil_vac.index, outer_coil_vac, label='Outer Vacuum', color="purple")
#for x in plasma_endpoints:
#    plt.axvline(x=x, color='black', linestyle='--')
plt.title(f'Raw Signal-#{vac_shot_no}')
plt.ylabel("Voltage [V]")
plt.xlabel("Time [s]")
plt.grid(True)
plt.legend()
plt.savefig(f"{vac_shot_no} Raw signals of diam_vac coils.png")


# Raw Signal of Plasma Shots
plt.figure()
#plt.plot(inner_coil.index, inner_coil, label='Inner', color="blue")
#plt.plot(outer_coil.index, outer_coil, label='Outer', color="orange")
plt.plot(inner_coil.index, inner_coil, label='Inner Loop', color="red")
plt.plot(outer_coil.index, outer_coil, label='Outer Loop', color="purple")
for x in plasma_endpoints:
    plt.axvline(x=x, color='black', linestyle='--')
plt.title(f'Raw Signal-#{shot_no}')
plt.ylabel("Voltage [V]")
plt.xlabel("Time [s]")
plt.grid(True)
plt.legend()
plt.savefig(f"{shot_no} diam coils.png")

plt.figure()
plt.plot(inner_flux.index, inner_flux, label='Inner', color="blue")
plt.plot(outer_flux.index, outer_flux, label='Outer', color="orange")
plt.plot(inner_flux_vac.index, inner_flux_vac, label='Inner Vacuum', color="red")
plt.plot(outer_flux_vac.index, outer_flux_vac, label='Outer Vacuum', color="purple")
for x in plasma_endpoints:
    plt.axvline(x=x, color='black', linestyle='--')
plt.title(f'#{shot_no}')
plt.ylabel("Flux [Wb]")
plt.xlabel("Time [s]")
plt.grid(True)
plt.legend()

plt.figure()
plt.plot(inner_flux.index, (1/A_inner)*inner_flux, label='Inner', color="blue")
plt.plot(outer_flux.index, (1/A_outer)*outer_flux, label='Outer', color="orange")
plt.plot(inner_flux_vac.index, (1/A_inner)*inner_flux_vac, label='Inner Vacuum', color="red")
plt.plot(outer_flux_vac.index, (1/A_outer)*outer_flux_vac, label='Outer Vacuum', color="purple")
for x in plasma_endpoints:
    plt.axvline(x=x, color='black', linestyle='--')
plt.title(f'#{shot_no}')
plt.ylabel("B [T]")
plt.xlabel("Time [s]")
plt.grid(True)
plt.legend()

from sklearn.linear_model import LinearRegression

y=np.array(inner_flux_vac.iloc[duration_index[0]:duration_index[1]])
X= np.array(outer_flux_vac.iloc[duration_index[0]:duration_index[1]])
X=X.reshape(-1,1)
model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
m=model.coef_[0]
c= model.intercept_
print("Coefficient (slope):", model.coef_[0])
print("Intercept:", model.intercept_)

jj= (A_outer/A_inner -1)*m

plt.figure()
plt.plot(inner_flux_vac.index[duration_index[0]:duration_index[1]], inner_flux_vac.iloc[duration_index[0]:duration_index[1]], label='Inner Diamagnetic Vacuum', color="red")
plt.plot(outer_flux_vac.index[duration_index[0]:duration_index[1]], m*outer_flux_vac.iloc[duration_index[0]:duration_index[1]]+c, label='Outer Diamagnetic Vacuum', color="purple")
plt.plot(outer_flux_vac.index[duration_index[0]:duration_index[1]], inner_flux_vac.iloc[duration_index[0]:duration_index[1]]-(m*outer_flux_vac.iloc[duration_index[0]:duration_index[1]]+c), label='Difference', color="gold")
for x in plasma_endpoints:
    plt.axvline(x=x, color='black', linestyle='--')
plt.title(f'#{vac_shot_no}')
plt.ylabel("Flux [Wb]")
plt.xlabel("Time [s]")
plt.ylim([-10e-6,10e-6])
plt.grid(True)
plt.legend()
#plt.savefig(f"{vac_shot_no}Matching.png")

plt.figure()
plt.plot(inner_flux_vac.index[duration_index[0]:duration_index[1]]*1000, inner_flux_vac.iloc[duration_index[0]:duration_index[1]], label='Inner Diamagnetic Vacuum', color="red")
plt.plot(outer_flux_vac.index[duration_index[0]:duration_index[1]]*1000, m*outer_flux_vac.iloc[duration_index[0]:duration_index[1]]+c, label='Outer Diamagnetic Vacuum', color="purple")
plt.plot(outer_flux_vac.index[duration_index[0]:duration_index[1]]*1000, inner_flux_vac.iloc[duration_index[0]:duration_index[1]]-(m*outer_flux_vac.iloc[duration_index[0]:duration_index[1]]+c), label='Difference', color="gold")
for x in plasma_endpoints:
    plt.axvline(x=x*1000, color='black', linestyle='--')
plt.title(f'#{vac_shot_no}')
plt.ylabel("Flux [Wb]")
plt.xlabel("Time [ms]")
#plt.ylim([-10e-6,10e-6])
plt.grid(True)
plt.legend()
plt.savefig(f"{vac_shot_no}Matching.png")

plt.figure()
plt.plot(inner_flux.index[duration_index[0]:duration_index[1]]*1000, inner_flux.iloc[duration_index[0]:duration_index[1]], label='Inner Diamagnetic ', color="red")
plt.plot(outer_flux.index[duration_index[0]:duration_index[1]]*1000, m*outer_flux.iloc[duration_index[0]:duration_index[1]]+c , label='Outer Diamagnetic', color="purple")
plt.plot(outer_flux.index[duration_index[0]:duration_index[1]]*1000, -(1/jj)*(inner_flux.iloc[duration_index[0]:duration_index[1]]-(m*outer_flux.iloc[duration_index[0]:duration_index[1]] +c)), label='Flux Change', color="gold")
for x in plasma_endpoints:
    plt.axvline(x=x*1000, color='black', linestyle='--')
plt.title(f'#{shot_no}')
plt.ylabel("Flux [Wb]")
plt.xlabel("Time [ms]")
plt.ylim([-10e-6,10e-6])
#plt.ylim([-0.0001,0.0001])
plt.grid(True)
plt.legend()

plt.figure()
plt.plot(outer_flux.index[duration_index[0]:duration_index[1]]*1000, -(1/jj)*(inner_flux.iloc[duration_index[0]:duration_index[1]]-(m*outer_flux.iloc[duration_index[0]:duration_index[1]]+c)), label='Flux Change', color="gold")
plt.plot(outer_flux_vac.index[duration_index[0]:duration_index[1]]*1000, inner_flux_vac.iloc[duration_index[0]:duration_index[1]]-(m*outer_flux_vac.iloc[duration_index[0]:duration_index[1]]+c), label='Vacuum Difference', color="red")
plt.plot(
    outer_flux_vac.index[duration_index[0]:duration_index[1]]*1000,
    -((1 / jj) * (
        inner_flux.iloc[duration_index[0]:duration_index[1]] - 
        (m * outer_flux.iloc[duration_index[0]:duration_index[1]] + c)
    )) - (
        inner_flux_vac.iloc[duration_index[0]:duration_index[1]] - 
        (m * outer_flux_vac.iloc[duration_index[0]:duration_index[1]] + c)
    ),
    label='Real Flux Change',
    color="green"
)
for x in plasma_endpoints:
    plt.axvline(x=x*1000, color='black', linestyle='--')
plt.title(f'#{shot_no}')
plt.ylabel("Flux [Wb]")
plt.xlabel("Time [ms]")
#plt.ylim([-0.0001,0.0001])
plt.grid(True)
plt.legend()
plt.savefig(f"{shot_no}CompletefluxChange.png")

mu = 4*np.pi*1e-7

Bt.values

del_phi_param = (mu*mu*new_rog_interp.values*new_rog_interp.values)/(8*np.pi*Bt)
len(del_phi_param)


plt.figure()
plt.plot(common_time*1000,del_phi_param, label="Paramagnetic Contribution")
for x in plasma_endpoints:
    plt.axvline(x=x*1000, color='black', linestyle='--')
plt.title(f'#{shot_no}')
plt.ylabel("Flux [Wb]")
plt.xlabel("Time [ms]")
plt.xlim(1)
plt.ylim([-10e-6,10e-6])
plt.grid(True)
plt.savefig(f"{shot_no}Paramagneticcontribution.png")


Z= -((1 / jj) * (
        inner_flux.iloc[duration_index[0]:duration_index[1]] - 
        (m * outer_flux.iloc[duration_index[0]:duration_index[1]] + c)
    )) - (
        inner_flux_vac.iloc[duration_index[0]:duration_index[1]] - 
        (m * outer_flux_vac.iloc[duration_index[0]:duration_index[1]] + c)
    )
M=del_phi_param.iloc[duration_index[0]:duration_index[1]]
BB=[]
for i in range(0,len(M)):
    BB.append(Z.iloc[i]-M.iloc[i])


plt.figure()
plt.plot(common_time[duration_index[0]:duration_index[1]]*1000,del_phi_param.iloc[duration_index[0]:duration_index[1]], label="Paramagnetic")
for x in plasma_endpoints:
    plt.axvline(x=x*1000, color='black', linestyle='--')
plt.title(f'#{shot_no}')
plt.ylabel("Flux [Wb]")
plt.xlabel("Time [ms]")
plt.xlim(1)
plt.ylim([-10e-6,10e-6])
plt.grid(True)

plt.plot(
    outer_flux_vac.index[duration_index[0]:duration_index[1]]*1000,
    -((1 / jj) * (
        inner_flux.iloc[duration_index[0]:duration_index[1]] - 
        (m * outer_flux.iloc[duration_index[0]:duration_index[1]] + c)
    )) - (
        inner_flux_vac.iloc[duration_index[0]:duration_index[1]] - 
        (m * outer_flux_vac.iloc[duration_index[0]:duration_index[1]] + c)
    ),
    label='Real Flux Change',
    color="green"
)
plt.plot(outer_flux.index[duration_index[0]:duration_index[1]]*1000, BB, label='Total Flux Minus Paramagnetic Flux', color="red")
plt.legend()
plt.savefig(f"{shot_no}-Diamagnetic Contribution")

plt.figure()
for x in plasma_endpoints:
    plt.axvline(x=x*1000, color='black', linestyle='--')
plt.title(f'#{shot_no}')
plt.ylabel("$W_{\perp}$ [J/m]")
plt.xlabel("Time [ms]")
#plt.ylim([-10e-6,10e-6])
plt.grid(True)
plt.plot(outer_flux.index[duration_index[0]:duration_index[1]]*1000, BB*(-1*Bt.iloc[duration_index[0]:duration_index[1]]/(mu)), label='Perpendicular Energy', color="purple")
plt.legend()
plt.savefig(f"{shot_no} - Perpendicular Energy")


R=0.4 #Major Radius

perp_energy=pd.Series(BB*(-1*Bt.iloc[duration_index[0]:duration_index[1]]/(mu)), index=Bt.index[duration_index[0]:duration_index[1]],name="perp_energy")
Thermal_energy= pd.Series(3/2*2*np.pi*perp_energy,index=perp_energy.index,name="Thermal_Energy")
Thermal_energy


plt.figure()
for x in plasma_endpoints:
    plt.axvline(x=x*1000, color='black', linestyle='--')
plt.title(f'#{shot_no}')
plt.ylabel("$W_{th}$ [J]")
plt.xlabel("Time [ms]")
#plt.ylim([-10e-6,10e-6])
plt.grid(True)
plt.plot(outer_flux.index[duration_index[0]:duration_index[1]]*1000,Thermal_energy, label='$W_{th} = 3/2 \cdot W_{\perp} \cdot (2\pi \cdot R)$', color="red")
plt.legend()
plt.savefig(f"{shot_no} - Thermal Energy")


Power=new_rog_interp*U_loop_interp
Power[plasma_endpoints[0]:plasma_endpoints[1]]


plt.figure()
for x in plasma_endpoints:
    plt.axvline(x=x*1000, color='black', linestyle='--')
plt.title(f'#{shot_no}')
plt.ylabel("$P_{\Omega}$ [kW]")
plt.xlabel("Time [ms]")
#plt.ylim([-10e-6,10e-6])
plt.grid(True)
plt.plot(Power.index*1000, Power/1000, label='$P_{\Omega}$', color="red")
plt.legend()
plt.savefig(f"{shot_no} - Power")


Thermal_energy.index=Thermal_energy.index/1000
T_E=Thermal_energy
P_E=Power

Power_interp= pd.Series(np.interp(common_time,P_E.index, Power), index= common_time)
Thermal_energy_interp=pd.Series(np.interp(common_time,Thermal_energy.index, Thermal_energy), index= common_time)

Thermal_energy_interp[0:plasma_endpoints[0]]=0
Power_interp[0:plasma_endpoints[0]]=0
Thermal_energy_interp[plasma_endpoints[1]:]=0
Power_interp[plasma_endpoints[1]:]=0


plt.figure()
for x in plasma_endpoints:
    plt.axvline(x=x*1000, color='black', linestyle='--')
plt.title(f'#{shot_no}')
plt.ylabel(r"$\tau_{E}$ [ms]")
plt.xlabel("Time [ms]")
plt.ylim([-10e-6,2])
plt.xlim()
plt.grid(True)
plt.plot(common_time*1000, 1000*Thermal_energy_interp/Power_interp)
plt.savefig(f"{shot_no} - Tau_E")
