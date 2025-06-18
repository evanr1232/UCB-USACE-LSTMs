import pandas as pd
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import subprocess
import time
import webbrowser
import plotly.graph_objects as go
from neuralhydrology.evaluation.metrics import calculate_all_metrics
import math


def clean_df(df):
    # EXTRA DEBUG: stepwise debug prints
    # print("\n[DEBUG:clean_df] => initial shape:", df.shape)
    # print("[DEBUG:clean_df] => HEAD(5):\n", df.head(5))

    df.columns = df.iloc[0]
    # print("[DEBUG:clean_df] => After df.columns = df.iloc[0], shape:", df.shape)
    # print("[DEBUG:clean_df] => HEAD(5):\n", df.head(5))

    df = df[3:]
    # print("[DEBUG:clean_df] => After df = df[3:], shape:", df.shape)
    # print("[DEBUG:clean_df] => HEAD(5):\n", df.head(5))

    df.columns = df.columns.str.strip()
    if 'Ordinate' in df.columns:
        df = df.drop(columns=['Ordinate'])
        # print("[DEBUG:clean_df] => Dropped 'Ordinate', shape:", df.shape)

    # rename date/time
    if 'Date' in df.columns:
        df = df.rename(columns={'Date': 'Day'})
    if 'Time' not in df.columns and 'time' in df.columns:
        df = df.rename(columns={'time': 'Time'})

    # print("[DEBUG:clean_df] => Columns after rename date/time:", df.columns.tolist())

    # 24:00:00 fix
    mask = df['Time'] == '24:00:00'
    count_24hr = mask.sum()
    # if count_24hr > 0:
        # print(f"[DEBUG:clean_df] => Found {count_24hr} rows with '24:00:00'")
    df.loc[mask, 'Day'] = (pd.to_datetime(df.loc[mask, 'Day'], format='%d-%b-%y') + pd.Timedelta(days=1)) \
        .dt.strftime('%d-%b-%y')
    df['Time'] = df['Time'].replace('24:00:00', '00:00:00')

    # new date col
    df['date'] = pd.to_datetime(df['Day'], format='%d-%b-%y') + pd.to_timedelta(df['Time'])
    # print("[DEBUG:clean_df] => After creating 'date', shape:", df.shape)
    # print("[DEBUG:clean_df] => HEAD(5):\n", df.head(5))

    df.dropna(subset=['date'], inplace=True)
    # print("[DEBUG:clean_df] => After dropna on 'date', shape:", df.shape)

    df = df.loc[:, ~df.columns.duplicated(keep=False)]
    # print("[DEBUG:clean_df] => After dropping duplicate cols, shape:", df.shape)

    df.set_index('date', inplace=True)
    if 'Day' in df.columns:
        df.drop(columns=['Day'], inplace=True)
    if 'Time' in df.columns:
        df.drop(columns=['Time'], inplace=True)
    # print("[DEBUG:clean_df] => Final shape after set_index & drop Day/Time =>", df.shape)
    # print("[DEBUG:clean_df] => HEAD(5):\n", df.head(5), "\n")

    return df


def combinedPlot(lstm_results: Path,
                 lstmPhysics_results: Path,
                 HMS_results: Path,
                 title: str,
                 fName: str = "metrics.csv",
                 timeseries_filename: str = None,
                 plot_filename: str = None):
    """
    Plot Observed, LSTM, Physics-LSTM, and HMS timeseries with Matplotlib,
    and save metrics + optionally the figure.
    """
    # 1) Load and rename
    lstm_df = pd.read_csv(lstm_results).rename(columns={'Predicted': 'LSTM_Predicted'})
    physics_lstm_df = pd.read_csv(lstmPhysics_results).rename(columns={'Predicted': 'PLSTM_Predicted'})
    physics_lstm_df.drop(columns=['Observed'], inplace=True, errors='ignore')

    # 2) Clean up negative predictions
    lstm_df['Date'] = pd.to_datetime(lstm_df['Date'])
    physics_lstm_df['Date'] = pd.to_datetime(physics_lstm_df['Date'])
    lstm_df.loc[lstm_df['LSTM_Predicted'] < 0, 'LSTM_Predicted'] = 0
    physics_lstm_df.loc[physics_lstm_df['PLSTM_Predicted'] < 0, 'PLSTM_Predicted'] = 0

    # print("[DEBUG - combinedPlot] => LSTM shape:", lstm_df.shape)
    # print("[DEBUG - combinedPlot] => LSTM columns:", lstm_df.columns.tolist())
    # print("[DEBUG - combinedPlot] => Physics LSTM shape:", physics_lstm_df.shape)
    # print("[DEBUG - combinedPlot] => Physics LSTM columns:", physics_lstm_df.columns.tolist())

    # 3) Load & Clean HMS
    hms_df = pd.read_csv(HMS_results)
    # print("[DEBUG - combinedPlot] => raw HMS shape:", hms_df.shape)
    cleaned_hms_df = clean_df(hms_df)
    # print("[DEBUG - combinedPlot] => cleaned HMS shape:", cleaned_hms_df.shape)
    if len(cleaned_hms_df.columns) > 0:
        main_col = cleaned_hms_df.columns[0]
        # rename first column => 'HMS_Predicted'
        cleaned_hms_df.rename(columns={main_col: 'HMS_Predicted'}, inplace=True)

    cleaned_hms_df = cleaned_hms_df.reset_index().rename(columns={'date': 'Date'})
    cleaned_hms_df = cleaned_hms_df[['Date', 'HMS_Predicted']]
    # print("[DEBUG - combinedPlot] => final HMS shape:", cleaned_hms_df.shape)

    # 4) Merge => LSTM + HMS
    # print("[DEBUG - combinedPlot] => Merging LSTM with HMS [INNER JOIN] ...")
    df_1 = lstm_df.merge(cleaned_hms_df, how='inner', on='Date')
    # print("[DEBUG - combinedPlot] => after LSTM/HMS merge => shape:", df_1.shape)

    # 5) Merge => combined + Phys LSTM
    # print("[DEBUG - combinedPlot] => Merging with Phys-LSTM [INNER JOIN] ...")
    df = df_1.merge(physics_lstm_df, how='inner', on='Date')
    # print("[DEBUG - combinedPlot] => after final merge => shape:", df.shape)

    # print("[DEBUG - combinedPlot] => columns:", df.columns.tolist())
    # print("[DEBUG - combinedPlot] => HEAD:\n", df.head(5))
    # print("[DEBUG - combinedPlot] => TAIL:\n", df.tail(5))
    # for c in df.columns:
    #     print(f"[DEBUG - combinedPlot] => NaN count in '{c}' => {df[c].isna().sum()}")

    # Ensure numeric
    df['Observed'] = pd.to_numeric(df['Observed'], errors='coerce')
    df['HMS_Predicted'] = pd.to_numeric(df['HMS_Predicted'], errors='coerce')
    df['LSTM_Predicted'] = pd.to_numeric(df['LSTM_Predicted'], errors='coerce')
    df['PLSTM_Predicted'] = pd.to_numeric(df['PLSTM_Predicted'], errors='coerce')

    # Save timeseries if requested
    if timeseries_filename:
        df[['Date', 'Observed', 'HMS_Predicted', 'LSTM_Predicted',
            'PLSTM_Predicted']].to_csv(timeseries_filename, index=False)
        # print("[DEBUG - combinedPlot] => Wrote merged timeseries CSV =>", timeseries_filename)

    # Convert to xarray
    obs_da = xr.DataArray(df['Observed'].values, dims=["date"], coords={"date": df['Date']})
    sim_da_hms = xr.DataArray(df['HMS_Predicted'].values, dims=["date"], coords={"date": df['Date']})
    sim_da_lstm = xr.DataArray(df['LSTM_Predicted'].values, dims=["date"], coords={"date": df['Date']})
    sim_da_plstm = xr.DataArray(df['PLSTM_Predicted'].values, dims=["date"], coords={"date": df['Date']})

    # Compute metrics
    hms_metrics = calculate_all_metrics(obs_da, sim_da_hms)
    hms_metrics["PBIAS"] = calculate_pbias(obs_da, sim_da_hms)

    lstm_metrics = calculate_all_metrics(obs_da, sim_da_lstm)
    lstm_metrics["PBIAS"] = calculate_pbias(obs_da, sim_da_lstm)

    plstm_metrics = calculate_all_metrics(obs_da, sim_da_plstm)
    plstm_metrics["PBIAS"] = calculate_pbias(obs_da, sim_da_plstm)

    metrics = {
        "HMS": hms_metrics,
        "LSTM": lstm_metrics,
        "Physics_Informed_LSTM": plstm_metrics,
    }

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(fName, index=True)
    # print(f"[INFO] Wrote metrics CSV: {fName}")

    # Plot
    plt.figure(figsize=(30, 10))
    plt.plot(df["Date"], df["Observed"], label='Observed', linewidth=2)
    plt.plot(df["Date"], df["HMS_Predicted"], label='HMS Prediction', linewidth=2, alpha=0.7)
    plt.plot(df["Date"], df["LSTM_Predicted"], label='LSTM Prediction', linewidth=2, alpha=0.8)
    plt.plot(df["Date"], df["PLSTM_Predicted"], label='Physics Informed LSTM', linewidth=2, alpha=0.7)

    plt.xlabel("Date", fontsize=20)
    plt.ylabel("Inflow (cfs)", fontsize=20)
    plt.title(title, fontsize=30)
    plt.legend(fontsize=25, loc="upper right")
    plt.grid(True, alpha=0.4)
    if not df["Date"].isna().all():
        plt.xlim(df['Date'].min(), df['Date'].max())
    plt.tight_layout()

    if plot_filename:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        # print(f"[INFO] Saved figure: {plot_filename}")

    plt.show()

    return plt, metrics_df


def fancyCombinedPlot(lstm_results: Path, lstmPhysics_results: Path, HMS_results: Path, title: str,
                      fName="metrics.csv", timeseries_filename: str = "combined_ts"):
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

    df = lstm_df.merge(cleaned_hms_df, how='inner', on='Date').merge(physics_lstm_df, how='inner', on='Date')

    df['Observed'] = pd.to_numeric(df['Observed'], errors='coerce')
    df['HMS_Predicted'] = pd.to_numeric(df['HMS_Predicted'], errors='coerce')
    df['LSTM_Predicted'] = pd.to_numeric(df['LSTM_Predicted'], errors='coerce')
    df['PLSTM_Predicted'] = pd.to_numeric(df['PLSTM_Predicted'], errors='coerce')

    df[['Date', 'Observed', 'HMS_Predicted', 'LSTM_Predicted',
        'PLSTM_Predicted']].to_csv(timeseries_filename, index=False)

    obs_da = xr.DataArray(df['Observed'].values, dims=["date"], coords={"date": df['Date']})
    sim_da_hms = xr.DataArray(df['HMS_Predicted'].values, dims=["date"], coords={"date": df['Date']})
    sim_da_lstm = xr.DataArray(df['LSTM_Predicted'].values, dims=["date"], coords={"date": df['Date']})
    sim_da_plstm = xr.DataArray(df['PLSTM_Predicted'].values, dims=["date"], coords={"date": df['Date']})

    hms_metrics = calculate_all_metrics(obs_da, sim_da_hms)
    hms_metrics["PBIAS"] = calculate_pbias(obs_da, sim_da_hms)

    lstm_metrics = calculate_all_metrics(obs_da, sim_da_lstm)
    lstm_metrics["PBIAS"] = calculate_pbias(obs_da, sim_da_lstm)

    plstm_metrics = calculate_all_metrics(obs_da, sim_da_plstm)
    plstm_metrics["PBIAS"] = calculate_pbias(obs_da, sim_da_plstm)

    metrics = {
        "HMS": hms_metrics,
        "LSTM": lstm_metrics,
        "Physics_Informed_LSTM": plstm_metrics,
    }

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(fName, index=True)
    # print(f"[INFO] Wrote metrics CSV: {fName}")

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df["Date"], y=df["Observed"], mode='lines', name='Observed'))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["HMS_Predicted"], mode='lines', name='HMS Prediction', opacity=0.8))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["LSTM_Predicted"], mode='lines', name='LSTM Prediction', opacity=0.8))
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["PLSTM_Predicted"], mode='lines', name='Physics Informed LSTM', opacity=0.8))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Inflow (cfs)",
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


def combinedPlotFromDf(
        df: pd.DataFrame,
        title: str,
        fName: str = "metrics.csv",
        timeseries_filename: str = None,
        plot_filename: str = None
):
    """
    Plot Observed, LSTM, Physics-LSTM, and HMS timeseries from a single DataFrame
    that already has columns:
        [Date, Observed, HMS_Predicted, LSTM_Predicted, PLSTM_Predicted]

    This function:
      1) Ensures columns are parsed to numeric/time
      2) Computes metrics between Observed and each prediction
      3) Exports metrics to a CSV (fName)
      4) Optionally saves the final merged timeseries to 'timeseries_filename'
      5) Creates a Matplotlib figure (optionally saved to 'plot_filename')
    """

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    numeric_cols = ["Observed", "HMS_Predicted", "LSTM_Predicted", "PLSTM_Predicted"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if timeseries_filename:
        df.to_csv(timeseries_filename, index=False)

    obs_da = xr.DataArray(df["Observed"].values, dims=["date"], coords={"date": df["Date"]})
    sim_da_hms = xr.DataArray(df["HMS_Predicted"].values, dims=["date"], coords={"date": df["Date"]})
    sim_da_lstm = xr.DataArray(df["LSTM_Predicted"].values, dims=["date"], coords={"date": df["Date"]})
    sim_da_plstm = xr.DataArray(df["PLSTM_Predicted"].values, dims=["date"], coords={"date": df["Date"]})

    hms_metrics = calculate_all_metrics(obs_da, sim_da_hms)
    hms_metrics["PBIAS"] = calculate_pbias(obs_da, sim_da_hms)

    lstm_metrics = calculate_all_metrics(obs_da, sim_da_lstm)
    lstm_metrics["PBIAS"] = calculate_pbias(obs_da, sim_da_lstm)

    plstm_metrics = calculate_all_metrics(obs_da, sim_da_plstm)
    plstm_metrics["PBIAS"] = calculate_pbias(obs_da, sim_da_plstm)

    metrics_dict = {
        "HMS": hms_metrics,
        "LSTM": lstm_metrics,
        "Physics_Informed_LSTM": plstm_metrics,
    }
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv(fName, index=True)
    # print(f"[INFO] Wrote metrics CSV: {fName}")

    plt.figure(figsize=(30, 10))
    plt.plot(df["Date"], df["Observed"], label="Observed", linewidth=2)
    plt.plot(df["Date"], df["HMS_Predicted"], label="HMS Prediction", linewidth=2, alpha=0.7)
    plt.plot(df["Date"], df["LSTM_Predicted"], label="LSTM Prediction", linewidth=2, alpha=0.8)
    plt.plot(df["Date"], df["PLSTM_Predicted"], label="Physics Informed LSTM", linewidth=2, alpha=0.7)

    plt.xlabel("Date", fontsize=20)
    plt.ylabel("Inflow (cfs)", fontsize=20)
    plt.title(title, fontsize=30)
    plt.legend(fontsize=25, loc="upper right")
    plt.grid(True, alpha=0.4)
    plt.xlim(df["Date"].min(), df["Date"].max())
    plt.tight_layout()

    if plot_filename:
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        # print(f"[INFO] Saved figure: {plot_filename}")

    plt.show()
    return plt, metrics_df


def fancyCombinedPlotFromDf(
        df: pd.DataFrame,
        title: str,
        fName: str = "metrics.csv",
        timeseries_filename: str = None
):
    """
    Similar to combinedPlotFromDf, but uses Plotly for an interactive figure.
    Args:
        df (pd.DataFrame): DataFrame with columns
            ["Date", "Observed", "HMS_Predicted", "LSTM_Predicted", "PLSTM_Predicted"]
        title (str): Plot title.
        fName (str): CSV filename to save computed metrics. Defaults to "metrics.csv".
        timeseries_filename (str, optional): If given, saves the timeseries to a CSV file.
    """

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    numeric_cols = ["Observed", "HMS_Predicted", "LSTM_Predicted", "PLSTM_Predicted"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if timeseries_filename:
        df.to_csv(timeseries_filename, index=False)

    obs_da = xr.DataArray(df["Observed"].values, dims=["date"], coords={"date": df["Date"]})
    sim_da_hms = xr.DataArray(df["HMS_Predicted"].values, dims=["date"], coords={"date": df["Date"]})
    sim_da_lstm = xr.DataArray(df["LSTM_Predicted"].values, dims=["date"], coords={"date": df["Date"]})
    sim_da_plstm = xr.DataArray(df["PLSTM_Predicted"].values, dims=["date"], coords={"date": df["Date"]})

    hms_metrics = calculate_all_metrics(obs_da, sim_da_hms)
    hms_metrics["PBIAS"] = calculate_pbias(obs_da, sim_da_hms)

    lstm_metrics = calculate_all_metrics(obs_da, sim_da_lstm)
    lstm_metrics["PBIAS"] = calculate_pbias(obs_da, sim_da_lstm)

    plstm_metrics = calculate_all_metrics(obs_da, sim_da_plstm)
    plstm_metrics["PBIAS"] = calculate_pbias(obs_da, sim_da_plstm)

    metrics_dict = {
        "HMS": hms_metrics,
        "LSTM": lstm_metrics,
        "Physics_Informed_LSTM": plstm_metrics,
    }
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv(fName, index=True)
    # print(f"[INFO] Wrote metrics CSV: {fName}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Observed"], mode="lines", name="Observed"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["HMS_Predicted"], mode="lines", name="HMS Prediction", opacity=0.8))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["LSTM_Predicted"], mode="lines", name="LSTM Prediction", opacity=0.8))
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["PLSTM_Predicted"], mode="lines", name="Physics Informed LSTM", opacity=0.8)
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Inflow (cfs)",
        template="seaborn",
        hovermode="x unified",
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ]
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
    )

    fig.show()
    return metrics_df

def extended_combined_plot(
        lstm_results: Path,
        lstmPhysics_results: Path,
        HMS_results: Path,
        title: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        metrics: list[str] | None = None,
        fName: str = "metrics.csv",
        timeseries_filename: str | None = None,
        plot_filename: str | None = None,
        interactive: bool = False):
    """Extended visualization for comparing HMS/LSTM/PILSTM predictions.

    This function mirrors ``combinedPlot``/``fancyCombinedPlot`` but adds the
    ability to subset the data by ``start_date``/``end_date`` and optionally
    compute a list of metrics. When ``metrics`` are provided, the values are
    printed to the console and appended to the legend labels.
    """

    lstm_df = pd.read_csv(lstm_results).rename(columns={"Predicted": "LSTM_Predicted"})
    physics_lstm_df = pd.read_csv(lstmPhysics_results).rename(columns={"Predicted": "PLSTM_Predicted"})
    physics_lstm_df.drop(columns=["Observed"], inplace=True, errors="ignore")

    hms_df = pd.read_csv(HMS_results)
    cleaned_hms_df = clean_df(hms_df)
    if len(cleaned_hms_df.columns) > 0:
        cleaned_hms_df.rename(columns={cleaned_hms_df.columns[0]: "HMS_Predicted"}, inplace=True)
    cleaned_hms_df = cleaned_hms_df.reset_index().rename(columns={"date": "Date"})
    cleaned_hms_df = cleaned_hms_df[["Date", "HMS_Predicted"]]
    lstm_df["Date"] = pd.to_datetime(lstm_df["Date"])
    physics_lstm_df["Date"] = pd.to_datetime(physics_lstm_df["Date"])
    df = lstm_df.merge(cleaned_hms_df, how="inner", on="Date").merge(physics_lstm_df, how="inner", on="Date")

    df.loc[df["LSTM_Predicted"] < 0, "LSTM_Predicted"] = 0
    df.loc[df["PLSTM_Predicted"] < 0, "PLSTM_Predicted"] = 0

    if start_date or end_date:
        if start_date:
            start_ts = pd.to_datetime(start_date)
            df = df[df["Date"] >= start_ts]
        if end_date:
            end_ts = pd.to_datetime(end_date)
            df = df[df["Date"] <= end_ts]

    numeric_cols = ["Observed", "HMS_Predicted", "LSTM_Predicted", "PLSTM_Predicted"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if timeseries_filename:
        df.to_csv(timeseries_filename, index=False)

    obs_da = xr.DataArray(df["Observed"].values, dims=["date"], coords={"date": df["Date"]})
    sim_da_hms = xr.DataArray(df["HMS_Predicted"].values, dims=["date"], coords={"date": df["Date"]})
    sim_da_lstm = xr.DataArray(df["LSTM_Predicted"].values, dims=["date"], coords={"date": df["Date"]})
    sim_da_plstm = xr.DataArray(df["PLSTM_Predicted"].values, dims=["date"], coords={"date": df["Date"]})

    hms_metrics = calculate_all_metrics(obs_da, sim_da_hms)
    hms_metrics["PBIAS"] = calculate_pbias(obs_da, sim_da_hms)
    lstm_metrics = calculate_all_metrics(obs_da, sim_da_lstm)
    lstm_metrics["PBIAS"] = calculate_pbias(obs_da, sim_da_lstm)
    plstm_metrics = calculate_all_metrics(obs_da, sim_da_plstm)
    plstm_metrics["PBIAS"] = calculate_pbias(obs_da, sim_da_plstm)

    metrics_dict = {
        "HMS": hms_metrics,
        "LSTM": lstm_metrics,
        "Physics_Informed_LSTM": plstm_metrics,
    }
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv(fName, index=True)

    def format_metrics(mdict: dict[str, float]) -> str:
        return ", ".join(f"{m}={mdict[m]:.3f}" for m in metrics if m in mdict) if metrics else ""

    if metrics:
        for name, vals in metrics_dict.items():
            print(name)
            for m in metrics:
                if m in vals:
                    print(f"  {m} = {vals[m]:.3f}")

    if interactive:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Observed"], mode="lines", name="Observed"))
        hms_label = "HMS Prediction"
        lstm_label = "LSTM Prediction"
        plstm_label = "Physics Informed LSTM"
        if metrics:
            hms_label += f" ({format_metrics(hms_metrics)})"
            lstm_label += f" ({format_metrics(lstm_metrics)})"
            plstm_label += f" ({format_metrics(plstm_metrics)})"
        fig.add_trace(go.Scatter(x=df["Date"], y=df["HMS_Predicted"], mode="lines", name=hms_label, opacity=0.8))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["LSTM_Predicted"], mode="lines", name=lstm_label, opacity=0.8))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["PLSTM_Predicted"], mode="lines", name=plstm_label, opacity=0.8))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Inflow (cfs)",
            template="seaborn",
            hovermode="x unified",
            xaxis=dict(
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ]
                ),
                rangeslider=dict(visible=True),
                type="date"
            ),
        )
        fig.show()
    else:
        plt.figure(figsize=(30, 10))
        hms_label = "HMS Prediction"
        lstm_label = "LSTM Prediction"
        plstm_label = "Physics Informed LSTM"
        if metrics:
            hms_label += f" ({format_metrics(hms_metrics)})"
            lstm_label += f" ({format_metrics(lstm_metrics)})"
            plstm_label += f" ({format_metrics(plstm_metrics)})"
        plt.plot(df["Date"], df["Observed"], label="Observed", linewidth=2)
        plt.plot(df["Date"], df["HMS_Predicted"], label=hms_label, linewidth=2, alpha=0.7)
        plt.plot(df["Date"], df["LSTM_Predicted"], label=lstm_label, linewidth=2, alpha=0.8)
        plt.plot(df["Date"], df["PLSTM_Predicted"], label=plstm_label, linewidth=2, alpha=0.7)

        plt.xlabel("Date", fontsize=20)
        plt.ylabel("Inflow (cfs)", fontsize=20)
        plt.title(title, fontsize=30)
        plt.legend(fontsize=18, loc="upper right")
        plt.grid(True, alpha=0.4)
        plt.xlim(df["Date"].min(), df["Date"].max())
        plt.tight_layout()
        if plot_filename:
            plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        plt.show()

    return metrics_df


def extended_combined_plot_from_df(
        df: pd.DataFrame,
        title: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        metrics: list[str] | None = None,
        fName: str = "metrics.csv",
        timeseries_filename: str | None = None,
        plot_filename: str | None = None,
        interactive: bool = False):

    """Extended version of :func:`combinedPlotFromDf` with optional subsetting
    and metric display.

    The input ``df`` must already contain the columns
    ``[Date, Observed, HMS_Predicted, LSTM_Predicted, PLSTM_Predicted]``.
    """

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    numeric_cols = ["Observed", "HMS_Predicted", "LSTM_Predicted", "PLSTM_Predicted"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if start_date:
        df = df[df["Date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["Date"] <= pd.to_datetime(end_date)]

    if timeseries_filename:
        df.to_csv(timeseries_filename, index=False)

    obs_da = xr.DataArray(df["Observed"].values, dims=["date"], coords={"date": df["Date"]})
    sim_da_hms = xr.DataArray(df["HMS_Predicted"].values, dims=["date"], coords={"date": df["Date"]})
    sim_da_lstm = xr.DataArray(df["LSTM_Predicted"].values, dims=["date"], coords={"date": df["Date"]})
    sim_da_plstm = xr.DataArray(df["PLSTM_Predicted"].values, dims=["date"], coords={"date": df["Date"]})

    hms_metrics = calculate_all_metrics(obs_da, sim_da_hms)
    hms_metrics["PBIAS"] = calculate_pbias(obs_da, sim_da_hms)
    lstm_metrics = calculate_all_metrics(obs_da, sim_da_lstm)
    lstm_metrics["PBIAS"] = calculate_pbias(obs_da, sim_da_lstm)
    plstm_metrics = calculate_all_metrics(obs_da, sim_da_plstm)
    plstm_metrics["PBIAS"] = calculate_pbias(obs_da, sim_da_plstm)

    metrics_dict = {
        "HMS": hms_metrics,
        "LSTM": lstm_metrics,
        "Physics_Informed_LSTM": plstm_metrics,
    }
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv(fName, index=True)

    def format_metrics(mdict: dict[str, float]) -> str:
        return ", ".join(f"{m}={mdict[m]:.3f}" for m in metrics if m in mdict) if metrics else ""

    if metrics:
        for name, vals in metrics_dict.items():
            print(name)
            for m in metrics:
                if m in vals:
                    print(f"  {m} = {vals[m]:.3f}")

    if interactive:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Observed"], mode="lines", name="Observed"))
        hms_label = "HMS Prediction"
        lstm_label = "LSTM Prediction"
        plstm_label = "Physics Informed LSTM"
        if metrics:
            hms_label += f" ({format_metrics(hms_metrics)})"
            lstm_label += f" ({format_metrics(lstm_metrics)})"
            plstm_label += f" ({format_metrics(plstm_metrics)})"
        fig.add_trace(go.Scatter(x=df["Date"], y=df["HMS_Predicted"], mode="lines", name=hms_label, opacity=0.8))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["LSTM_Predicted"], mode="lines", name=lstm_label, opacity=0.8))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["PLSTM_Predicted"], mode="lines", name=plstm_label, opacity=0.8))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Inflow (cfs)",
            template="seaborn",
            hovermode="x unified",
            xaxis=dict(
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ]
                ),
                rangeslider=dict(visible=True),
                type="date"
            ),
        )
        fig.show()
    else:
        plt.figure(figsize=(30, 10))
        hms_label = "HMS Prediction"
        lstm_label = "LSTM Prediction"
        plstm_label = "Physics Informed LSTM"
        if metrics:
            hms_label += f" ({format_metrics(hms_metrics)})"
            lstm_label += f" ({format_metrics(lstm_metrics)})"
            plstm_label += f" ({format_metrics(plstm_metrics)})"
        plt.plot(df["Date"], df["Observed"], label="Observed", linewidth=2)
        plt.plot(df["Date"], df["HMS_Predicted"], label=hms_label, linewidth=2, alpha=0.7)
        plt.plot(df["Date"], df["LSTM_Predicted"], label=lstm_label, linewidth=2, alpha=0.8)
        plt.plot(df["Date"], df["PLSTM_Predicted"], label=plstm_label, linewidth=2, alpha=0.7)

        plt.xlabel("Date", fontsize=20)
        plt.ylabel("Inflow (cfs)", fontsize=20)
        plt.title(title, fontsize=30)
        plt.legend(fontsize=18, loc="upper right")
        plt.grid(True, alpha=0.4)
        plt.xlim(df["Date"].min(), df["Date"].max())
        plt.tight_layout()
        if plot_filename:
            plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        plt.show()

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

    tb_command = f"tensorboard --logdir={logdir} --port={port} --host=0.0.0.0"  # TensorBoard command
    try:
        process = subprocess.Popen(tb_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(5)
        url = f"http://localhost:{port}"
        webbrowser.open(url)
        print(f"TensorBoard started at {url} with logs from {logdir}")
    except Exception as e:
        raise Exception(f"Failed to start TensorBoard: {e}")

    return process


def fractional_multi_lr(
        epochs: int,
        fractions: list,
        lrs: list,
        round_up: bool = True
) -> dict:
    """
    Build a milestone dictionary for an arbitrary piecewise learning-rate schedule.
    - 'fractions' are N fractional breakpoints that sum up to <= 1.0
    - 'lrs' are N+1 learning rates
    - We generate N+1 segments in total.

    Example:
      fractions=[0.4, 0.3], lrs=[0.01, 0.005, 0.001]
      => first 40% => LR=0.01
         next 30% => LR=0.005
         final 30% => LR=0.001

    The dictionary might look like: {0: 0.01, 7: 0.005, 11: 0.001} for epochs=16
    (assuming round_up=True).
    """
    if len(lrs) != len(fractions) + 1:
        raise ValueError(
            "Number of learning rates must be len(fractions) + 1. "
            f"Got {len(lrs)} LRs and {len(fractions)} fractions."
        )
    if sum(fractions) > 1.0:
        raise ValueError(
            f"The sum of fractions exceeds 1.0 => {sum(fractions)}"
        )

    schedule = {}
    schedule[0] = lrs[0]
    cumulative = 0.0
    for i, frac in enumerate(fractions, start=1):
        cumulative += frac
        boundary_float = cumulative * epochs
        boundary_index = math.ceil(boundary_float) if round_up else int(boundary_float)
        schedule[boundary_index] = lrs[i]

    return schedule


def calculate_pbias(observed, simulated):
    """
    Calculate Percent Bias (PBIAS) between observed and simulated data.
    PBIAS = [ (sum(Obs-Sim) / sum(Obs)) * 100 ]
    """
    if observed.shape != simulated.shape:
        raise ValueError("Observed and simulated DataArrays must have the same shape.")

    pbias = ((observed - simulated).sum() / observed.sum()) * 100
    return pbias.item()
