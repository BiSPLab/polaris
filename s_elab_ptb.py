# %% Imports.
import numpy as np
import pandas as pd
import wfdb
from wfdb.processing import XQRS
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from functools import reduce

# %%
folder = 'datasets\\ptb-diagnostic-ecg-database-1.0.0\\ptb-diagnostic-ecg-database-1.0.0'

def list_all_files(directory):
    # Get all files and directories
    entries = os.listdir(directory)
    files = [f for f in entries if os.path.isfile(os.path.join(directory, f))]
    return files

max_patient_id = 294

# Sampling rate and filter coefficients.
fs = 1000.0
b, a = butter(3, np.array([0.5, 40])/(fs/2), btype='bandpass')

# Leads.
leads = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'vx', 'vy', 'vz']

output = []

for ii in range(1, max_patient_id):   
    curr_patient = f'patient{ii:03d}'
    patient_folder = os.path.join(folder, curr_patient)

    if os.path.exists(patient_folder):
        print(f'{curr_patient} in progress...')
        
        files = list_all_files(patient_folder)

        # Get record_names.
        record_names = list(set([x.split('.')[0] for x in files]))
        record_names.sort()

        # Get the first record_name
        record_name = record_names[0]
        
        signals, fields = wfdb.rdsamp(os.path.join(patient_folder, record_name))
        comments = fields['comments']

        # Check fs.
        if fields['fs'] != fs:
            print(f'patient{ii} does not have singlas sampled at {fs} Hz')
            continue

        # Check units.
        all_mV = reduce(lambda prev, curr: prev & (curr == 'mV'), fields['units'], True)
        if not all_mV:
            print(f'patient{ii} does not have recordings all in mV')
            continue

        # Check lead order.
        all_good_leads = reduce(lambda prev, curr: prev & (curr[1] == leads[curr[0]]), enumerate(fields['sig_name']), True)
        if not all_good_leads:
            print(f'patient{ii} does not have a good order of leads')
            continue

        # Get age.
        age = np.nan
        try:
            age_list = [x[4:].strip() for x in comments if x[:4] == 'age:']
            if len(age_list) >= 1:
                age = int(age_list[0])
        except:
            print(f'patient{ii} does not have age')
            continue

        # Get sex.
        sex = 0
        try:
            sex_list = [x[4:].strip() for x in comments if x[:4] == 'sex:']
            if len(sex_list) >= 1:
                sex = 1 if sex_list[0] == 'male' else 0
        except:
            print(f'patient{ii} does not have sex')
            continue

        signals = filtfilt(b, a, signals, axis=0, method='gust')

        for ss in range(signals.shape[1]):
            hist, bin_edges = np.histogram(signals[:, ss], range=[-2, 2], bins=200)
            imax = np.argmax(hist)
            bin_size = np.mean(np.diff(bin_edges))
            offset = bin_edges[imax] + bin_size/2
            signals[:, ss] -= offset

        # Get QRS peaks.
        vm = np.sum(signals**2, axis=1)
        xqrs = XQRS(sig=vm, fs=fs)
        xqrs.detect()
        qrs_peaks = xqrs.qrs_inds

        # Select stable beats.
        RR = np.concatenate([[np.nan], np.diff(qrs_peaks/fs)])
        RR_bins = np.arange(0.5, 1.5, 0.01)
        RR_counts = np.zeros_like(RR_bins)
        th_RR = 0.01 # s
        for irr in range(0, len(RR_bins)):
            RR_counts[irr] = np.sum((np.abs(RR[:-1] - RR_bins[irr]) <= th_RR) & (np.abs(RR[1:] - RR_bins[irr]) <= th_RR))

        imax = np.argmax(RR_counts)
        tf = np.concatenate([[False], (np.abs(RR[:-1] - RR_bins[imax]) <= th_RR) & (np.abs(RR[1:] - RR_bins[imax]) <= th_RR)])
        beat_indices = np.array([ix for ix, x in enumerate(tf) if x == True])

        if(len(beat_indices) < 11):
            print(f'patient{ii} does not have a sufficient number of stable beats')
            continue

        n_beats = len(beat_indices)
        N_before = round(300*fs/1000)
        N_after = round(450*fs/1000)
        beats = np.zeros((n_beats, N_before + N_after))*np.nan
        for ibeat in range(len(beat_indices)-1):
            curr_beat_index = beat_indices[ibeat]
            i_left = qrs_peaks[curr_beat_index] - N_before
            i_right = np.min([qrs_peaks[curr_beat_index] + N_after, qrs_peaks[curr_beat_index + 1] - N_before])
            if i_left >= 0:                
                beats[ibeat, :(i_right-i_left)] = signals[i_left:i_right, 7]

        avg_beat = np.nanmean(beats, axis=0)
        avg_beat[np.isnan(avg_beat)] = 0.0

        result_mi = any('acute' in item.lower() for item in comments) & any('infarction' in item.lower() for item in comments) & any('antero' in item.lower() for item in comments)
        
        diagnose = None
        if result_mi:
            diagnose = 1
            signal_to_save = signals

        result_norm = any('healthy control' in item.lower() for item in comments) 
        if result_norm:
            diagnose = 0
            signal_to_save = signals

        if result_mi and result_norm:
            print(f'patient{ii} has both MI and healthy control')
            continue

        if result_mi or result_norm:
            # To save.
            # Selection of the first 10 s.
            output.append([age, sex, diagnose] + avg_beat.tolist())
            pass
            

# Save csv.
dataset = np.array(output)
np.savetxt('datasets/ecg.csv', dataset, delimiter=',')

