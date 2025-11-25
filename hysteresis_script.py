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

import plotly.graph_objects as go
#from scipy.integrate import simpson

#%% Specified parameters to change!

CLGB_AG_q_files = ['CLGBag_Q_2022-2025.csv']
CLGB_AG_n_files = ['CLGBag_N_2022-2023.csv', 'CLGBag_N_2024.csv']

CLGB_UP_q_files = ['CLGB.UP_2024-2025_DISCHARGE_OFFICIAL-2025-09-17.csv']
CLGB_UP_n_files = ['CLGBup_N_raw_2025.csv']

# OGraw_n_files = ['AG_SUNA_legible_june2024.csv', 'AG_SUNA_legible_july2024.csv', 'AG_SUNA_legible_sept2024.csv']

#%% Load data function

def load(filename):
    df = pd.read_csv(filename, index_col=[0])
    df.index = pd.to_datetime(df.index, errors='coerce', format='mixed')
    df.columns = df.columns.str.lower().str.strip() #handles NO3.mgL vs no3.mgl, etc. for future merge       
    return df

#%% data df creation

def data_wizardy(n_files, q_files):
    
    # Load all N file(s)
    raw_ndata = pd.DataFrame()
    
    for file in n_files:
        raw_ndata = pd.concat([raw_ndata, load(file)], axis = 0)
    
    n_data = raw_ndata.copy()
    
    # Drop negative N values
    n_data.loc[n_data['no3.mgl'] < 0, 'no3.mgl'] = np.nan
    
    # Set zeros to Nan in error col
    if 'no3.mgl.error' in n_data.columns.tolist():
        n_data['no3.mgl.error'] = pd.to_numeric(n_data['no3.mgl.error'], errors='coerce')
        n_data.loc[n_data['no3.mgl.error'] == 0, 'no3.mgl.error'] = np.nan
    
    # need to address N flags?
    
    # Load Q file(s)
    raw_qdata = pd.DataFrame()
    
    for file in q_files:
        raw_qdata = pd.concat([raw_qdata, load(file)], axis = 0)
    
    q_data = raw_qdata.copy()
    
    # Q QC flag columns -- for later
    # if 'q.m3sqf' in q_data.columns.tolist():
    #     q_data = q_data[q_data['q.m3sqf'] == False]
    
    # Combine N and Q files
    keep_cols = ['no3.mgl', 'q.m3s']
    
    data = pd.merge(n_data, q_data, how='outer', left_index=True, right_index=True)
    data = data[keep_cols]
    
    data.rename(columns={'no3.mgl': 'N'}, inplace=True)
    data.rename(columns={'q.m3s': 'Q'}, inplace=True)

    return data, raw_ndata, raw_qdata

#%%
data_CLGB_AG, ag_rawN, ag_rawQ = data_wizardy(n_files=CLGB_AG_n_files, q_files=CLGB_AG_q_files)
data_CLGB_UP, up_rawN, up_rawQ = data_wizardy(n_files=CLGB_UP_n_files, q_files=CLGB_UP_q_files)

#%% Full data record time-series plotting

def data_record_plot(data, sitename):
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

#%%
data_record_plot(data=data_CLGB_AG, sitename="CLGB.AG")
data_record_plot(data=data_CLGB_UP, sitename="CLGB.UP")

#%% AI try at interactive

def AI_interactive_plot(data, sitename):

    # --- Create the figure
    fig = go.Figure()
    
    # --- Nitrate trace (left Y-axis, WebGL)
    fig.add_trace(go.Scattergl(
        x=data.index,
        y=data["N"],
        mode='lines',
        name="Nitrate (mg/L)",
        line=dict(color="orange", width=1),
        yaxis="y1"
    ))
    
    # --- Discharge trace (right Y-axis, WebGL)
    fig.add_trace(go.Scattergl(
        x=data.index,
        y=data["Q"],
        mode='lines',
        name="Discharge (m³/s)",
        line=dict(color="blue", width=1),
        yaxis="y2"
    ))
    
    # --- Layout settings
    fig.update_layout(
        title=f'{sitename}, NH: Q versus N over time (Interactive)',
        xaxis=dict(
            title="Date",
            type="date",
            rangeslider=dict(visible=True),  # draggable zoom bar
            rangeselector=dict(              # quick zoom buttons
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        ),
        yaxis=dict(
            title="Nitrate concentration (mg/L)",
            titlefont=dict(color="orange"),
            tickfont=dict(color="orange")
        ),
        yaxis2=dict(
            title="Discharge (m³/s)",
            titlefont=dict(color="blue"),
            tickfont=dict(color="blue"),
            overlaying="y",
            side="right"
        ),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(x=0, y=1.1, orientation="h"),
        height=600
    )
    
    fig.show()

#%%
AI_interactive_plot(data=data_CLGB_AG, sitename="CLGB.AG")
AI_interactive_plot(data=data_CLGB_UP, sitename="CLGB.UP")

#%% Create storm df

# List of tuples with (start, end)
storm_list = [
    ('CLGB.AG', '2022-09-05 00:00', '2022-09-05 23:45'),
    ('CLGB.AG', '2022-09-19 19:30', '2022-09-20 05:00'),
    ('CLGB.AG', '2022-09-22 08:00', '2022-09-22 22:30'),
    ('CLGB.AG', '2022-10-05 18:30', '2022-10-06 00:30'),
    ('CLGB.AG', '2023-06-05 04:00', '2023-06-05 18:00'),
    ('CLGB.AG', '2023-06-13 00:00', '2023-06-13 16:45'),
    ('CLGB.AG', '2023-06-14 16:00', '2023-06-15 16:00'),
    ('CLGB.AG', '2023-06-16 22:00', '2023-06-17 10:30'),
    ('CLGB.AG', '2023-06-17 11:00', '2023-06-18 16:00'),
    ('CLGB.UP', '2025-05-03 14:45', '2025-05-04 00:45'),
    ('CLGB.UP', '2025-05-04 17:00', '2025-05-05 19:30'),
    ('CLGB.UP', '2025-05-09 16:00', '2025-05-10 21:00')
]

storms = pd.DataFrame(storm_list, columns=['site', 'start', 'end'])
storms['start'] = pd.to_datetime(storms['start'])
storms['end'] = pd.to_datetime(storms['end'])

# Set index as storm number (1, 2, 3, ...), by date
storms = storms.sort_values('start')
storms.index = range(1, len(storms) + 1)
storms.index.name = 'storm'

#%% Individual storm plots

def CQplot(data, start, end, title_preamble):
    
    trim = data.loc[start:end]
    #print(start)
    #print(trim)
    
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
    ax1.set_title(f'{title_preamble}: Q and N time series: \n{start} to {end}')
    
    # Auto-format date labels
    fig.autofmt_xdate()
    
    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    
    # Format x-axis ticks: month/day (no leading zero) and HH:MM
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%#m/%#d %H:%M'))  # On Windows, use '%#m/%#d %H:%M'
    
    plt.show()
    
    # -------------------   PLOT C-Q   ------------------------
    
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
        c=time_numeric,           # Color by time
        cmap='viridis_r',         # Other options: 'plasma', 'cividis', 'turbo', _r reverses
        linewidth=0.5,
        edgecolor='none'
    )
    
    ax1.set_ylabel("Nitrate concentration (mg/L)", fontsize=13)
    ax1.set_xlabel("Discharge (m³/s)", fontsize=13)
    ax1.set_title(f'{title_preamble} nitrate C-Q: \n{start} to {end}')
    
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

for event in storms.index:
    start=storms.loc[event, 'start']
    end=storms.loc[event, 'end']
    site = storms.loc[event, 'site']
    
    if site == "CLGB.AG":
        data = data_CLGB_AG
    elif site == "CLGB.UP":
        data = data_CLGB_UP
    #elif site == "CLGB":
    #    data = data_CLGB
    else:
        print(f'Unknown site for storm {event}: {site}.')
        continue
    
    CQplot(data=data, start=start, end=end, title_preamble=f'{site}, storm {event}')
    
#%% Hysteresis analyses

# Initialize columns in storms df to hold hysteresis data
storms[['h', 'hyst_class', 'diff_area_max', 'diff_area_min', 'x_fixed_start']] = np.nan

def hi(storm_data, start, end, idx, title_preamble):
    
    import zuecco_h as zh
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    
    x = storm_data['Q']
    y = storm_data['N']
    
    x_fixed_full = pd.Series([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 
                         0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])

    for drop_start in range(5):
        x_fixed_try = x_fixed_full[drop_start:].reset_index(drop=True)
        try:
            diff_area, h, hyst_class = zh.hysteresis_class(x, y, x_fixed_try)
            storms.loc[idx, 'h'] = h
            storms.loc[idx, 'hyst_class'] = hyst_class
            storms.loc[idx, 'diff_area_max'] = diff_area.max()
            storms.loc[idx, 'diff_area_min'] = diff_area.min()
            storms.loc[idx, 'x_fixed_start'] = x_fixed_try.iloc[0]
            
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
            # Plot after successful run
            ax1 = axes[0]
            ax2 = axes[1]
            
            ax1.plot(x,y)
            ax1.set_xlabel("Discharge (Q)")
            ax1.set_ylabel("Nitrate (C)")
            ax1.set_title(f'{title_preamble}: Zuecco hysteresis analysis')
            
            x2 = [0, 0.5, 1]
            y2 = [0, 0, 0]
            
            ax2.plot(x_fixed_try[:-1], diff_area, color="red")
            ax2.plot(x2, y2, color="black")
            ax2.set_xlabel('Streamflow (-)')
            ax2.set_ylabel('ΔA (-)')
            ax2.set_title(f'{title_preamble}: Difference between the integrals')
            
            fig.tight_layout()
            break  # success, exit fallback loop
            
        except ValueError:
            if drop_start == 4:
                storms.loc[idx, ['h', 'hyst_class', 'diff_area_max', 'diff_area_min', 'x_fixed_start']] = np.nan
                print(f"Storm {idx}: all x_fixed attempts failed, skipping.")

            continue  # try again with next shorter x_fixed

#%% Calc flux calculations

data_CLGB_UP['flux_mg_s'] = data_CLGB_UP['N'] * data_CLGB_UP['Q'] * 1000.0
data_CLGB_AG['flux_mg_s'] = data_CLGB_AG['N'] * data_CLGB_AG['Q'] * 1000.0

#%% storm analysis function

def storm_analysis(storms, site1, data1, site2, data2):

    # initialize metric columns
    storms['Qmax_time'] = pd.NaT  # NaT = Not-a-Time for datetime
    storms[['Nstart_conc', 'Qmax', 'Nmax_conc', 'N_conc_percent_change', 'flushing_slope', 'N_storm_yield_kg', 'Nflux_max_mg_s', 'Nflux_avg_mg_s']] = np.nan
    
    for event in storms.index:
        start=storms.loc[event, 'start']
        end=storms.loc[event, 'end']  
        site = storms.loc[event, 'site']
        
        # assign the correct data df to use based on 'site' in storms df
        if site == site1:
            data = data1
        elif site == site2:
            data = data2
        #elif site == site3:
        #    data = data3
        else:
            print(f'Unknown site for storm {event}: {site}.')
            continue
        
        trim = data.loc[start:end]
        
        # calculate fraction of missing values for N and Q
        frac_missing_N = trim['N'].isna().sum() / len(trim)
        frac_missing_Q = trim['Q'].isna().sum() / len(trim)
        
        # skip storm if either exceeds 20%
        if frac_missing_N > 0.2 or frac_missing_Q > 0.2:
            print(f"Storm {event} skipped due to too many missing N or Q values")
            continue
        trim = trim.interpolate(method='time', limit=2)
        
        Nstart = trim.loc[trim.index[0], 'N']
        Qmax_time = trim['Q'].idxmax()
        Qmax = trim['Q'].max()
        Nmax = trim['N'].max()
        
        storms.loc[event, 'Nstart_conc'] = Nstart
        storms.loc[event, 'Qmax_time'] = Qmax_time
        storms.loc[event, 'Qmax'] = Qmax
        storms.loc[event, 'Nmax_conc'] = Nmax
        storms.loc[event, 'flushing_slope'] = (Nmax-Nstart) / ((Qmax_time - start).total_seconds()/3600) # mg/L per hour
        storms.loc[event, 'Nflux_max_mg_s'] = trim['flux_mg_s'].max()
        storms.loc[event, 'Nflux_avg_mg_s'] = trim['flux_mg_s'].mean()
        storms.loc[event, 'N_conc_percent_change'] = (Nmax/Nstart)*100
    
    
        # calculate yield
        t_seconds = (trim.index.view('int64') // 1_000_000_000).astype(float)
        storms.loc[event, 'N_storm_yield_kg'] = np.trapz(trim['flux_mg_s'], t_seconds) / 1e6
        
        hi(storm_data=trim, start=start, end=end, idx=event, title_preamble=f'{site}, storm {event}')
        
    return storms
        
#%%
storms = storm_analysis(storms=storms, site1='CLGB.AG', data1=data_CLGB_AG, site2='CLGB.UP', data2=data_CLGB_UP)

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
    .applymap(
        color_pos_neg,
        subset=[
            'h', 'diff_area_max', 'diff_area_min',
            'flushing_slope'
        ]
    ) \
    .format({
        # existing metrics
        "h": "{:.2f}",
        "hyst_class": "{:.0f}",
        "diff_area_max": "{:.4f}",
        "diff_area_min": "{:.4f}",
        "x_fixed_start": "{:.2f}",

        # new metrics — added & formatted
        "Nstart_conc": "{:.3f}",             # mg/L
        "Qmax": "{:.3f}",                    # m³/s typically
        "Nmax_conc": "{:.3f}",               # mg/L
        "N_conc_percent_change": "{:.1f}",   # percent
        "flushing_slope": "{:.3f}",          # mg/L per hr
        "N_storm_yield_kg": "{:.4f}",        # kg
        "Nflux_max_mg_s": "{:.2f}",          # mg/s
        "Nflux_avg_mg_s": "{:.2f}",          # mg/s
    })