import pandas as pd
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import subprocess
import time
import webbrowser
import plotly.graph_objects as go
from neuralhydrology.evaluation.metrics import calculate_all_metrics

def clean_df(df):
    # Clean columns/rows
    df.columns = df.iloc[0]
    df = df[3:]
    df.columns = df.columns.str.strip()
    df = df.drop(columns=['Ordinate'])
    df = df.rename(columns={'Date': 'Day', 'Time': 'Time'})
    
    # Increment the date by 1 day where Time is '24:00:00' and replace '24:00:00' with '00:00:00'
    mask = df['Time'] == '24:00:00'
    df.loc[mask, 'Day'] = (pd.to_datetime(df.loc[mask, 'Day'], format='%d-%b-%y') + pd.Timedelta(days=1)).dt.strftime('%d-%b-%y')
    df['Time'] = df['Time'].replace('24:00:00', '00:00:00')

    # Combine 'Day' and 'Time' columns to create a new 'date' column, make sure duplicated columns are dropped
    df['date'] = pd.to_datetime(df['Day'], format='%d-%b-%y') + pd.to_timedelta(df['Time'])
    df.dropna(subset=['date'], inplace=True)
    df = df.loc[:, ~df.columns.duplicated(keep=False)]
    df.set_index('date', inplace=True)

    # Drop 'Day' and 'Time' columns
    df.drop(columns=['Day', 'Time'], inplace=True)
    return df

def combinedPlot(lstm_results: Path, lstmPhysics_results: Path, HMS_results: Path, title: str, fName = "metrics.csv"):
    
    lstm_df = pd.read_csv(lstm_results).rename(columns={'Predicted': 'LSTM_Predicted'}) #columns: Date, Observed, Predicted --> LSTM_Predicted
    lstm_df['Date'] = pd.to_datetime(lstm_df['Date'])
    lstm_df.loc[lstm_df['LSTM_Predicted'] < 0, 'LSTM_Predicted'] = 0
    
    physics_lstm_df = pd.read_csv(lstmPhysics_results).rename(columns={'Predicted': 'PLSTM_Predicted'}) #columns: Date, Observed, Predicted --> PLSTM_Predicted
    physics_lstm_df['Date'] = pd.to_datetime(physics_lstm_df['Date'])
    physics_lstm_df.drop(columns=['Observed'], inplace=True)
    physics_lstm_df.loc[physics_lstm_df['PLSTM_Predicted'] < 0, 'PLSTM_Predicted'] = 0
    
    hms_df = pd.read_csv(HMS_results) #columns: Date, HMS_predicted
    cleaned_hms_df = clean_df(hms_df)
    # Rename the first column to 'HMS_Predicted'
    cleaned_hms_df.rename(columns={cleaned_hms_df.columns[0]: 'HMS_Predicted'}, inplace=True)

    # Add a 'Date' column containing the index values
    cleaned_hms_df = cleaned_hms_df.reset_index().rename(columns={'date': 'Date'})
    cleaned_hms_df = cleaned_hms_df[['Date', 'HMS_Predicted']]

    # Merge the LSTM, PLSTM and HMS DataFrames on the 'Date' column
    df = lstm_df.merge(cleaned_hms_df, how='right', on='Date').merge(physics_lstm_df, how='right', on='Date')

    # Ensure that the columns contain only numeric values
    df['Observed'] = pd.to_numeric(df['Observed'], errors='coerce')
    df['HMS_Predicted'] = pd.to_numeric(df['HMS_Predicted'], errors='coerce')
    df['LSTM_Predicted'] = pd.to_numeric(df['LSTM_Predicted'], errors='coerce')
    df['PLSTM_Predicted'] = pd.to_numeric(df['PLSTM_Predicted'], errors='coerce')
    
    # Convert pandas Series to xarray DataArray with a datetime coordinate
    obs_da = xr.DataArray(df['Observed'].values, dims=["date"], coords={"date": df['Date']})
    sim_da_hms = xr.DataArray(df['HMS_Predicted'].values, dims=["date"], coords={"date": df['Date']})
    sim_da_lstm = xr.DataArray(df['LSTM_Predicted'].values, dims=["date"], coords={"date": df['Date']})
    sim_da_plstm = xr.DataArray(df['PLSTM_Predicted'].values, dims=["date"], coords={"date": df['Date']})

    # Collect metrics into a dictionary
    metrics = {
        "HMS": calculate_all_metrics(obs_da, sim_da_hms),
        "LSTM": calculate_all_metrics(obs_da, sim_da_lstm),
        "Physics_Informed_LSTM": calculate_all_metrics(obs_da, sim_da_plstm),
    }

    metrics_df = pd.DataFrame(metrics)
    output_csv_path = fName
    metrics_df.to_csv(output_csv_path)

    # Plot all columns against the "date_col" (x-axis)
    plt.figure(figsize=(30, 10))
    
    plt.plot(df["Date"], df["Observed"], label='Observed', linewidth=2)
    plt.plot(df["Date"], df["HMS_Predicted"], label='HMS Prediction',  linewidth=2, alpha=0.7)
    plt.plot(df["Date"], df["LSTM_Predicted"], label='LSTM Prediction', linewidth=2, alpha=0.8)
    plt.plot(df["Date"], df["PLSTM_Predicted"], label='Physics Informed LSTM Prediction', linewidth=2, alpha=0.7)
    
    # Customize the plot
    plt.tick_params(axis='x', labelsize=15)  # For x-axis tick labels
    plt.tick_params(axis='y', labelsize=15) 
    plt.xlabel("Date", fontsize=20)
    plt.ylabel("Inflow (cubic feet per second)", fontsize=20)
    plt.title(title, fontsize=30)
    plt.legend(fontsize=25, loc="upper right")
    plt.grid(True, alpha=0.4)

    # Set x-axis limits to the extent of the observed data
    plt.xlim(df['Date'].min(), df['Date'].max())

    plt.tight_layout()
    plt.show()

    return plt, metrics_df

def fancyCombinedPlot(lstm_results: Path, lstmPhysics_results: Path, HMS_results: Path, title: str, fName = "metrics.csv"):
    
    lstm_df = pd.read_csv(lstm_results).rename(columns={'Predicted': 'LSTM_Predicted'})
    lstm_df['Date'] = pd.to_datetime(lstm_df['Date'])
    lstm_df.loc[lstm_df['LSTM_Predicted'] < 0, 'LSTM_Predicted'] = 0
    
    physics_lstm_df = pd.read_csv(lstmPhysics_results).rename(columns={'Predicted': 'PLSTM_Predicted'})
    physics_lstm_df['Date'] = pd.to_datetime(physics_lstm_df['Date'])
    physics_lstm_df.drop(columns=['Observed'], inplace=True)
    physics_lstm_df.loc[physics_lstm_df['PLSTM_Predicted'] < 0, 'PLSTM_Predicted'] = 0
    
    hms_df = pd.read_csv(HMS_results)
    cleaned_hms_df = clean_df(hms_df)
    cleaned_hms_df.rename(columns={cleaned_hms_df.columns[0]: 'HMS_Predicted'}, inplace=True)
    cleaned_hms_df = cleaned_hms_df.reset_index().rename(columns={'date': 'Date'})
    cleaned_hms_df = cleaned_hms_df[['Date', 'HMS_Predicted']]

    df = lstm_df.merge(cleaned_hms_df, how='right', on='Date').merge(physics_lstm_df, how='right', on='Date')
    
    df['Observed'] = pd.to_numeric(df['Observed'], errors='coerce')
    df['HMS_Predicted'] = pd.to_numeric(df['HMS_Predicted'], errors='coerce')
    df['LSTM_Predicted'] = pd.to_numeric(df['LSTM_Predicted'], errors='coerce')
    df['PLSTM_Predicted'] = pd.to_numeric(df['PLSTM_Predicted'], errors='coerce')
    
    obs_da = xr.DataArray(df['Observed'].values, dims=["date"], coords={"date": df['Date']})
    sim_da_hms = xr.DataArray(df['HMS_Predicted'].values, dims=["date"], coords={"date": df['Date']})
    sim_da_lstm = xr.DataArray(df['LSTM_Predicted'].values, dims=["date"], coords={"date": df['Date']})
    sim_da_plstm = xr.DataArray(df['PLSTM_Predicted'].values, dims=["date"], coords={"date": df['Date']})

    metrics = {
        "HMS": calculate_all_metrics(obs_da, sim_da_hms),
        "LSTM": calculate_all_metrics(obs_da, sim_da_lstm),
        "Physics_Informed_LSTM": calculate_all_metrics(obs_da, sim_da_plstm),
    }

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(fName)

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Observed"], mode='lines', name='Observed'))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["HMS_Predicted"], mode='lines', name='HMS Prediction', opacity=0.8))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["LSTM_Predicted"], mode='lines', name='LSTM Prediction', opacity=0.8))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["PLSTM_Predicted"], mode='lines', name='Physics Informed LSTM', opacity=0.8))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Inflow (cubic feet per second)",
        template="seaborn",
        hovermode="x unified",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
    )
    
    fig.show()
    return metrics_df

def open_tensorboard(logdir: str, port: int = 6006):
    """
    Opens TensorBoard and display logs.
    
    Args:
        logdir (str): Path to the directory containing TensorBoard event files.
        port (int): Port to host TensorBoard on (default: 6006).
    """
    logdir_path = Path(logdir)
    
    # Check that the log directory exists
    if not logdir_path.exists():
        raise FileNotFoundError(f"Log directory {logdir} does not exist.")

    # Check if event files exist in the log directory
    event_files = list(logdir_path.rglob("events.out.tfevents*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in log directory {logdir}.")
    
    tb_command = f"tensorboard --logdir={logdir} --port={port} --host=0.0.0.0" # TensorBoard command       
    try:
        # Start TensorBoard as a subprocess
        process = subprocess.Popen(tb_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(5)
        # Open TensorBoard in the default web browser
        url = f"http://localhost:{port}"
        webbrowser.open(url)
        print(f"TensorBoard started at {url} with logs from {logdir}")
    
    except Exception as e:
        raise Exception(f"Failed to start TensorBoard: {e}")

    return process