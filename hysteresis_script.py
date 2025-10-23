# -*- coding: utf-8 -*-
"""
College Brook hysteresis

Parameters
----------
q_files : list of strings of discharge (Q) datafiles. i.e. ['CLGBag_Q_2022-2025.csv']
n_files : list of strings of discharge (Q) datafiles. i.e. ['CLGBag_N_2022-2023.csv', 'CLGBag_N_2024.csv']

@author: josephbaldus
@date = 2025-10-20
@license = MIT -- https://opensource.org/licenses/MIT
"""

#%% Imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#%% Specified parameters to change!

q_files = ['CLGBag_Q_2022-2025.csv']
n_files = ['CLGBag_N_2022-2023.csv', 'CLGBag_N_2024.csv']

#%% GLOBAL VARIABLE/UNIT CONVERSIONS

#%% Load data function

def load(filename):
    df = pd.read_csv(filename, index_col=[0])
    df.index = pd.to_datetime(df.index, format='mixed', errors='coerce')
    df.columns = df.columns.str.lower().str.strip() #handles NO3.mgL vs no3.mgl, etc. for future merge
    return df

#%% Load all N file(s)

raw_ndata = pd.DataFrame()

for file in n_files:
    raw_ndata = pd.concat([raw_ndata, load(file)], axis = 0)
    
# N QC flag columns
if 'no3.mgl.qf' in raw_ndata.columns.tolist():
    n_data = raw_ndata[raw_ndata['no3.mgl.qf'] == False] #false is good

if 'flag' in raw_ndata.columns.tolist():
    n_data = raw_ndata[~raw_ndata['flag'].isin([1,2])] #1,2 is bad

#%% Load Q file(s)

raw_qdata = pd.DataFrame()

for file in q_files:
    raw_qdata = pd.concat([raw_qdata, load(file)], axis = 0)
    
# Q QC flag columns
if 'q.m3sqf' in raw_qdata.columns.tolist():
    q_data = raw_qdata[raw_qdata['q.m3sqf'] == False]

#%% Combine N and Q files

keep_cols = ['no3.mgl', 'q.m3s', 'measured.q.m3s']

data = pd.merge(n_data, q_data, how='outer', left_index=True, right_index=True)
data = data[keep_cols]

data.rename(columns={'no3.mgl': 'N'}, inplace=True)
data.rename(columns={'q.m3s': 'Q'}, inplace=True)

#%% Checks function

check_n = n_data[n_data['no3.mgl'].notna()]
check_n2 = data[data['N'].notna()]

check_q = q_data[q_data['q.m3s'].notna()]
check_q2 = data[data['Q'].notna()]

#%% Plots!

data = data.loc['2024-08-01':'2024-08-15'] #to subset data/zoom in, will integrate into a plot function

fig1, ax1 = plt.subplots(figsize=(12, 4))

# Plot nitrate concentration (left Y-axis)
ax1.plot(data.index, data.N, color='orange', label="Nitrate (mg/L)", linewidth=1)
ax1.set_ylabel("Nitrate concentration (mg/L)", color='orange')
ax1.tick_params(axis='y', labelcolor='orange')

# Create a twin axis sharing the same X-axis
ax2 = ax1.twinx()

# Plot discharge (right Y-axis)
ax2.plot(data.index, data.Q, color='blue', label="Discharge (m³/s)", linewidth=.5)
ax2.set_ylabel("Discharge (m³/s)", color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Add title
ax1.set_title("College Brook, NH: Q versus N over time")

# Auto-format date labels
fig1.autofmt_xdate()

# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.show()

#%% Plot 2 (turn into function?) in progress...

fig2, ax1 = plt.subplots(figsize=(8, 4))

subset = data.loc['2022-08':'2022-12']
ax1.plot(subset.index, subset.N, label='NO₃')
ax1.plot(subset.index, subset.Q, label='Discharge')
