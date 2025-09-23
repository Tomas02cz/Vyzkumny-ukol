from functions import *
from input_parameters import *

import numpy as np
import pandas as pd

import requests
import os



def scan_database(shot_start, shot_end, url):
    shot_no = np.arange(shot_start, shot_end)
    
    # check accessibility of data file
    URL = ['http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/BasicDiagnostics/Results/Bt.csv', 'http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/BasicDiagnostics/Results/Ich.csv', 'http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/BasicDiagnostics/Results/U_loop.csv']
    print('...CHECKING BASIC DIAGNOSTICS...')
    error_no = 0
    for url_template in URL:
        shot_no = check_basic_diagnostics(shot_no, url_template)
    print(f'Overall {error_no} errors in page accessibility')
    
    print('...CHECKING ADVANCED DIAGNOSTICS...')
    error_no = 0
    for url_template in url:
        shot_no = check_accessibility(shot_no, url_template)
    print(f'Overall {error_no} errors in page accessibility')
    
    # find out if the shot is plasma or vacuum shot
    shot_no = plasma_or_vacuum_shot(shot_no)
    
    # save shots with diagnostics to csv
    shot_no.to_csv(f'results/shots_with_MHDring_between_{shot_start}_and_{shot_end}.csv', index=False)
    
    return shot_no

def check_basic_diagnostics(shot_no, url_template):
    presence = np.zeros_like(shot_no, dtype=int)
    i = 0
    for shot in shot_no:
        url = url_template.format(shot_no=shot)
        try:
            response = requests.head(url, allow_redirects=True, timeout=5)
            if response.status_code == 200:
                presence[i] = 1
            elif response.status_code == 404:
                presence[i] = 0
            else:
                presence[i] = 0
                error_no += 1
                print(f'Error {response.status_code} in accessibility of shot {shot}')
        except requests.RequestException:
            presence[i] = 0
            error_no += 1
        i += 1
    # keep only shots with basic diagnostics
    valid_shot_no = shot_no[presence == 1]
    return valid_shot_no
    
def check_accessibility(shot_no, url_template):
    presence = np.zeros_like(shot_no, dtype=int)
    i = 0
    error_no = 0
    for shot in shot_no:
        url = url_template.format(shot_no=shot)
        try:
            response = requests.head(url, allow_redirects=True, timeout=5)
            if response.status_code == 200:
                presence[i] = 1
            elif response.status_code == 404:
                presence[i] = 0
            else:
                presence[i] = 0
                error_no += 1
                print(f'Error {response.status_code} in accessibility of shot {shot}')
        except requests.RequestException:
            presence[i] = 0
            error_no += 1
        i += 1
    # keep only shots with accessible diagnostics
    valid_shot_no = shot_no[presence == 1]
    
    return valid_shot_no

def plasma_or_vacuum_shot(shots):
    # find out if the shot is plasma or vacuum shot
    plasma_or_vacuum_shot = np.array([1 if check_plasma_presence(shot) else 0 for shot in shots])
    no_of_plasma_shots = sum(plasma_or_vacuum_shot)
    no_of_vacuum_shots = len(plasma_or_vacuum_shot) - no_of_plasma_shots
    
    # numbers of shots with running diagnostics
    print('...')
    print(f'{no_of_plasma_shots} plasma discharges and {no_of_vacuum_shots} vacuum shots with presence of diagnostics')
    valid_shots = pd.DataFrame(np.column_stack((shots, plasma_or_vacuum_shot)))
    valid_shots.columns = ['shot number', 'shot type']
    
    return valid_shots

def check_parameter_value(shot_no: int, file) -> bool:
    url_template: str = 'http://golem.fjfi.cvut.cz/shots/{shot_no}/Operation/Discharge/'
    content = requests.get(url.format(shot_no=shot_no)).content
    parameter_value = pd.read_csv(io.StringIO(content.decode('iso-8859-1')), names = ['value'])
    if parameter_value['value'][0] == requested_value:
        return True
    else:
        return False

def filter_by_basic_parameters(shots):
    
    
    
    
    U_Bt = pd.read_csv(f"{url_template}/U_bt_discharge_request")
    return valid_shots

def group_by_basic_parameters(shot_start, shot_end):
    
    return valid_shots


''' MAIN '''

# Create output directory
os.makedirs("results", exist_ok=True)

# scan Golem database
shot_no_with_diagnostics = scan_database(49090, 49093, [URL_MIRNOV2])
print(shot_no_with_diagnostics)