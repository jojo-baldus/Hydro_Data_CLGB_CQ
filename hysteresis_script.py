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
from matplotlib import dates as mdates

#%% Specified parameters to change!

sitename = 'CLGB.Ag'
q_files = ['CLGBag_Q_2022-2025.csv']
n_files = ['CLGBag_N_2022-2023.csv', 'CLGBag_N_2024.csv']

OGraw_n_files = ['AG_SUNA_legible_june2024.csv', 'AG_SUNA_legible_july2024.csv', 'AG_SUNA_legible_sept2024.csv']

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

n_data = raw_ndata.copy()

# Drop negative N values
n_data.loc[n_data['no3.mgl'] < 0, 'no3.mgl'] = np.nan

# Set zeros to Nan in error col
n_data['no3.mgl.error'] = pd.to_numeric(n_data['no3.mgl.error'], errors='coerce')
n_data.loc[n_data['no3.mgl.error'] == 0, 'no3.mgl.error'] = np.nan

#%% Check OG raw data
OG_n = pd.DataFrame()

for file in OGraw_n_files:
    OG_n = pd.concat([OG_n, load(file)], axis = 0)
    
OG_n = OG_n.loc[OG_n['no3.mgl'] >= 0, :]

#%% Load Q file(s)

raw_qdata = pd.DataFrame()

for file in q_files:
    raw_qdata = pd.concat([raw_qdata, load(file)], axis = 0)
    
# Q QC flag columns
if 'q.m3sqf' in raw_qdata.columns.tolist():
    q_data = raw_qdata[raw_qdata['q.m3sqf'] == False]

#%% Combine N and Q files

keep_cols = ['no3.mgl', 'no3.mgl.error', 'q.m3s', 'measured.q.m3s']

data = pd.merge(n_data, q_data, how='outer', left_index=True, right_index=True)
data = data[keep_cols]

data.rename(columns={'no3.mgl': 'N'}, inplace=True)
data.rename(columns={'q.m3s': 'Q'}, inplace=True)

#%% Checks that merge worked

# check_n = n_data[n_data['no3.mgl'].notna()]
# check_n2 = data[data['N'].notna()]

# check_q = q_data[q_data['q.m3s'].notna()]
# check_q2 = data[data['Q'].notna()]

#%% Full data time-series

fig1, ax2 = plt.subplots(figsize=(16, 4))

# Plot discharge (right Y-axis)
ax2.plot(data.index, data.Q, color='navy', label="Discharge (m³/s)", linewidth=.3)
ax2.set_ylabel("Discharge (m³/s)", color='navy')
ax2.tick_params(axis='y', labelcolor='navy')

# Create a twin axis sharing the same X-axis
ax1 = ax2.twinx()

# Plot nitrate concentration (left Y-axis)
ax1.plot(data.index, data.N, color='darkorange', label="Nitrate (mg/L)", linewidth=.3)
#ax1.plot(data.index, data['no3.mgl.error'], color='red', label="Error Nitrate (mg/L)", linewidth=1)
#ax1.plot(OG_n.index, OG_n['no3.mgl'], color='red', label="Raw Nitrate (mg/L)", linewidth=.5)
ax1.set_ylabel("Nitrate concentration (mg/L)", color='darkorange')
ax1.tick_params(axis='y', labelcolor='darkorange')

# ax2.set_yscale('log')

# Add title
ax1.set_title(sitename, fontsize = 20)

# Auto-format date labels
fig1.autofmt_xdate()

# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.show()

#%% AI try at interactive

# import plotly.graph_objects as go

# # --- Create the figure
# fig = go.Figure()

# # --- Nitrate trace (left Y-axis, WebGL)
# fig.add_trace(go.Scattergl(
#     x=data.index,
#     y=data["N"],
#     mode='lines',
#     name="Nitrate (mg/L)",
#     line=dict(color="orange", width=1),
#     yaxis="y1"
# ))

# # --- Discharge trace (right Y-axis, WebGL)
# fig.add_trace(go.Scattergl(
#     x=data.index,
#     y=data["Q"],
#     mode='lines',
#     name="Discharge (m³/s)",
#     line=dict(color="blue", width=1),
#     yaxis="y2"
# ))

# # --- Layout settings
# fig.update_layout(
#     title="College Brook, NH: Q versus N over time (Interactive)",
#     xaxis=dict(
#         title="Date",
#         type="date",
#         rangeslider=dict(visible=True),  # draggable zoom bar
#         rangeselector=dict(              # quick zoom buttons
#             buttons=list([
#                 dict(count=7, label="1w", step="day", stepmode="backward"),
#                 dict(count=1, label="1m", step="month", stepmode="backward"),
#                 dict(count=6, label="6m", step="month", stepmode="backward"),
#                 dict(count=1, label="1y", step="year", stepmode="backward"),
#                 dict(step="all")
#             ])
#         )
#     ),
#     yaxis=dict(
#         title="Nitrate concentration (mg/L)",
#         titlefont=dict(color="orange"),
#         tickfont=dict(color="orange")
#     ),
#     yaxis2=dict(
#         title="Discharge (m³/s)",
#         titlefont=dict(color="blue"),
#         tickfont=dict(color="blue"),
#         overlaying="y",
#         side="right"
#     ),
#     hovermode="x unified",
#     template="plotly_white",
#     legend=dict(x=0, y=1.1, orientation="h"),
#     height=600
# )

# fig.show()

#%% Create storm df

# List of tuples with (start, end)
storm_dates = [
    ('2022-09-05 00:00', '2022-09-05 23:45'),
    ('2022-09-19 19:30', '2022-09-20 05:00'),
    ('2022-09-22 08:00', '2022-09-22 22:30'),
    ('2022-10-05 18:30', '2022-10-06 00:30'),
    ('2023-06-05 04:00', '2023-06-05 18:00'),
    ('2023-06-13 00:00', '2023-06-13 16:45'),
    ('2023-06-14 16:00', '2023-06-15 16:00'),
    ('2023-06-16 22:00', '2023-06-17 10:30'),
    ('2023-06-17 11:00', '2023-06-18 16:00')
]

storms = pd.DataFrame(storm_dates, columns=['start', 'end'])
storms['start'] = pd.to_datetime(storms['start'])
storms['end'] = pd.to_datetime(storms['end'])

# Set index as storm number (1, 2, 3, ...)
storms.index = range(1, len(storms) + 1)
storms.index.name = 'storm'

# Initialize columns for hysteresis data
storms[['h', 'hyst_class', 'diff_area_max', 'diff_area_min', 'x_fixed_start']] = np.nan

#%% Individual storm plots

def CQplot(start, end, sitename=sitename):
    
    trim = data.loc[start:end] #to subset data/zoom in, will integrate into a plot function
    
    fig, ax1 = plt.subplots(figsize=(8, 4))
    
    # Plot nitrate concentration (left Y-axis)
    ax1.plot(trim.index, trim.N, color='darkorange', label="Nitrate (mg/L)", linewidth=1)
    ax1.set_ylabel("Nitrate concentration (mg/L)", color='darkorange')
    ax1.tick_params(axis='y', labelcolor='darkorange')
    
    # Create a twin axis sharing the same X-axis
    ax2 = ax1.twinx()
    
    # Plot discharge (right Y-axis)
    ax2.plot(trim.index, trim.Q, color='navy', label="Discharge (m³/s)", linewidth=.8)
    ax2.set_ylabel("Discharge (m³/s)", color='navy')
    ax2.tick_params(axis='y', labelcolor='navy')
    
    # Add title
    ax1.set_title(f'{sitename}: Q and N time series, {start} to {end}')
    
    # Auto-format date labels
    fig.autofmt_xdate()
    
    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    
    # Format x-axis ticks: month/day (no leading zero) and HH:MM
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%#m/%#d %H:%M'))  # On Windows, use '%#m/%#d %H:%M'
    
    plt.show()
    
    # PLOT C-Q
    # Convert datetime index to numeric (for color mapping)
    time_numeric = mdates.date2num(trim.index)
    
    # Create the scatter plot with color mapping
    fig, ax1 = plt.subplots(figsize=(8, 8))
    
    # Line connecting points (sorted in time order)
    ax1.plot(
        trim.Q, trim.N,
        color='gray', linewidth=0.8, alpha=0.6, zorder=1
    )
    
    # Scatter of points
    sc = ax1.scatter(
        trim.Q,
        trim.N,
        c=time_numeric,                # Color by time
        cmap='viridis_r',                # Choose a nice gradient colormap (try 'plasma', 'cividis', 'turbo' too)
        linewidth=0.5,
        edgecolor='none'
    )
    
    ax1.set_ylabel("Nitrate concentration (mg/L)", fontsize=13)
    ax1.set_xlabel("Discharge (m³/s)", fontsize=13)
    ax1.set_title(f'{sitename}: Nitrate C-Q, {start} to {end}')
    
    # Add colorbar showing the date scale
    cbar = plt.colorbar(sc, ax=ax1)
    
    # Format the colorbar ticks as readable dates
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%H'))
    cbar.set_label("Storm event time progression", fontsize=13)
    
    cbar.ax.text(1.0, -0.01, 'Start', transform=cbar.ax.transAxes,
                 ha='right', va='top', fontsize=10, color='black')
    cbar.ax.text(1.0, 1.01, 'End', transform=cbar.ax.transAxes,
                 ha='right', va='bottom', fontsize=10, color='black')  
    plt.show()

#%% C-Q plot function calls

for id in storms.index:
    start=storms.loc[id, 'start']
    end=storms.loc[id, 'end']
    
    CQplot(start, end, f'{sitename}, storm {id}: {start} to {end}')
    
#%% Hysteresis analyses

import zuecco_h as zh

for id in storms.index:

    start=storms.loc[id, 'start']
    end=storms.loc[id, 'end']
    
    event_df = data.loc[start:end, ['Q','N']].interpolate(method ='linear')
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    
    x = event_df['Q']
    y = event_df['N']
    
    x_fixed_full = pd.Series([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 
                         0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])

    for drop_start in range(5):
        x_fixed_try = x_fixed_full[drop_start:].reset_index(drop=True)
        try:
            diff_area, h, hyst_class = zh.hysteresis_class(x, y, x_fixed_try)
            storms.loc[id, 'h'] = h
            storms.loc[id, 'hyst_class'] = hyst_class
            storms.loc[id, 'diff_area_max'] = diff_area.max()
            storms.loc[id, 'diff_area_min'] = diff_area.min()
            storms.loc[id, 'x_fixed_start'] = x_fixed_try.iloc[0]
            
            # Plot after successful run
            ax1 = axes[0]
            ax2 = axes[1]
            
            ax1.plot(x,y)
            ax1.set_xlabel("Discharge (Q)")
            ax1.set_ylabel("Nitrate (C)")
            ax1.set_title(f'Storm {id}: Zuecco hysteresis analysis')
            
            x2 = [0, 0.5, 1]
            y2 = [0, 0, 0]
            
            ax2.plot(x_fixed_try[:-1], diff_area, color="red")
            ax2.plot(x2, y2, color="black")
            ax2.set_xlabel('Streamflow (-)')
            ax2.set_ylabel('ΔA (-)')
            ax2.set_title(f'Storm {id}: Difference between the integrals')
            
            fig.tight_layout()
            break  # success, exit fallback loop
            
        except ValueError:
            if drop_start == 4:
                storms.loc[id, ['h', 'hyst_class', 'diff_area_max', 'diff_area_min', 'x_fixed_start']] = np.nan
                print(f"Storm {id}: all x_fixed attempts failed, skipping.")
                CQplot(start, end, f'Failed hysteresis analysis,{sitename}: storm {id}')

            continue  # try again with next shorter x_fixed
        
#%% Results table

def color_pos_neg(val):
    if pd.isna(val):
        return ""  # leave NaNs uncolored
    elif val > 0:
        return "background-color: lightblue"
    elif val < 0:
        return "background-color: lightyellow"
    else:
        return ""  # zero stays default

storms.style.set_caption("Hysteresis Analysis of Storms") \
    .applymap(color_pos_neg, subset=['h', 'diff_area_max', 'diff_area_min']) \
    .format({
        "h": "{:.2f}",
        "hyst_class": "{:.0f}",
        "diff_area_max": "{:.4f}",
        "diff_area_min": "{:.4f}",
        "x_fixed_start": "{:.2f}"
    })