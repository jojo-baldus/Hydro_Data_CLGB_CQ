#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ESCI 895 Lab 6: 

This script takes a collection of annual SCAN soil moisture (SMS) datafiles, cleans and concatonates them,
then analyzes trends in annual precipitation and SMS using various linear regressions. 

Parameters
----------
data_folder : str
    The name of the folder containing the Fluxnet flux data files for analysis; must be in working directory.

station_info : str
    The name of the text file containing the site location, porosity, and date
    (in that order, seperated by commas) within the data folder
    
@author: josephbaldus
@date = 2025-10-13
@license = MIT -- https://opensource.org/licenses/MIT
"""

#%% Imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from pathlib import Path
import sms_functions as sms

#%% Specified parameters to change!

# Name of subdirectory containing unzipped datafiles
data_folder = 'data'  

# Name of text file containing station info with .txt extension
station_info = 'station_info.txt'

#%% GLOBAL VARIABLE/UNIT CONVERSIONS
IN_TO_CM = 2.54

#%% info file reading
info = pd.read_csv(Path(data_folder)/station_info)
study_location = info.columns[0]

#%% run data reading function over for loop for all data files

# generate list of files in data folder
filenames = os.listdir(data_folder)
data = pd.DataFrame()

for name in filenames:
    if name[-3:] == 'csv':
        append = sms.load_data(name, data_folder = data_folder)
        data = pd.concat([data, append], axis = 0)

#%% trim with start date, resample
data = data[pd.to_datetime(info.columns[2]):].resample('D').first()

#%% Pull out soil depths, drop SMS values greater than derived max porosity
depths = []
porosity = float(info.columns[1])

for name in data.columns.tolist():
    if name.startswith('SMS'):
        depths.append(int(name[9:11]))
        data.loc[data[name] > porosity*100, name] = np.nan

depths = np.array(depths)

#%% Filling specific missing SMS depth data IF other depths exist
pre_fill = data.copy() # Can use to double-check fill

sms.fill_missing_sms(data)

#%% 1.10 trapezoid

data['sms_total'] = np.trapz(data.iloc[:,1:6]/100, x = depths*IN_TO_CM, axis = 1)

#%% time series plots
fig1, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(12, 12))

count=0
for col in data.columns[1:6]:
    ax[0].plot(data.index, data[col], label=f'{depths[count]} cm', linewidth=.7)
    count = count+1

ax[1].plot(data.sms_total, linewidth=1)
ax[2].plot(data.p_cm, linewidth=.7)

# set all y-axes labels
y_labels = ['Soil moisture (%) by depth', 'Total soil moisture (cm)', 'Precipitation (cm/day)']

for i in range(len(ax)):
    ax[i].set_ylabel(y_labels[i], fontsize=11)
    ax[i].tick_params(axis='both', width=.5, labelsize=8)

# additional formatting
ax[0].set_title(study_location, fontsize=20)
ax[0].legend(fontsize=8)

for a in ax:  # iterate over each subplot
    for spine in a.spines.values():
        spine.set_color('gray')
        spine.set_linewidth(1.5)
        
plt.show()

#%% Annual Calculations
# generate wateryear col
data['wateryear'] = data.index.year
data.loc[data.index.month > 9, 'wateryear'] = data.loc[data.index.month > 9, 'wateryear'] +1  # correct those problem months

# create annual complete data count by wateryear
clean = data.dropna()
annual = pd.DataFrame()
annual['complete_data_days'] = clean.groupby('wateryear')['p_cm'].count()

# Fill ALL nans in p and sms cols, calculate annual stats
data.interpolate(method='linear', axis=0, inplace=True)
annual['total_p'] = data.groupby('wateryear')['p_cm'].sum()
annual['avg_sms'] = data.groupby('wateryear')['sms_total'].mean()

# select only year
complete_threshold = 354
annual = annual.loc[annual['complete_data_days'] > complete_threshold, :]

print(f'Aggregated by year, then averaged, the annual average precipitation is {annual.total_p.mean():.2f} cm, and the average annual soil moisture is {annual.avg_sms.mean():.2f}%.')

#%% call regressions

annual_int = annual.index.values.astype(int) - annual.index.min()

sms.regressplot(annual_int, annual.total_p, nonparam=True, xlabel='Year', ylabel='Total precip (cm/yr)', xtoplot=annual.index, title = study_location)
sms.regressplot(annual_int, annual.avg_sms, nonparam=True, xlabel='Year', ylabel='Average annaul soil moisture (cm)', xtoplot=annual.index, title = study_location)
sms.regressplot(annual.total_p, annual.avg_sms, nonparam=True, xlabel='Total precip (cm/yr)', ylabel='Average annaul soil moisture (cm)', title = study_location)
