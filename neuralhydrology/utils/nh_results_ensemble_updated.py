"""Utility script to average the predictions of several runs and calculate percentiles."""
from neuralhydrology.utils.errors import AllNaNError
from neuralhydrology.utils.config import Config
from neuralhydrology.evaluation.utils import metrics_to_dataframe
from neuralhydrology.evaluation.metrics import calculate_metrics, get_available_metrics
from neuralhydrology.datautils.utils import get_frequency_factor, sort_frequencies
from neuralhydrology.utils import nh_run
import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.parent))


def create_results_ensemble(run_dirs: List[Path],
                            best_k: int = None,
                            metrics: List[str] = None,
                            period: str = 'test',
                            epoch: int = None) -> dict:
    """Average the predictions of several runs for the specified period and calculate new metrics."""
    if len(run_dirs) < 2:
        raise ValueError(
            'Need to provide at least two run directories to be merged.')

    if period not in ['train', 'validation', 'test']:
        raise ValueError(f'Unknown period {period}.')
    if best_k is not None:
        if period != 'test':
            raise ValueError(
                'If best_k is specified, the period must be test.')
        print('Searching for best validation runs.')
        best_val_runs = _get_best_validation_runs(run_dirs, best_k, epoch)
        best_runs = [_get_results_file(run_dir, period, epoch)
                     for run_dir in best_val_runs]
    else:
        best_runs = [_get_results_file(run_dir, period, epoch)
                     for run_dir in run_dirs]

    config = Config(run_dirs[0] / 'config.yml')
    if metrics is not None:
        config.metrics = metrics

    # get frequencies from a results file.
    run_results = pickle.load(open(best_runs[0], 'rb'))
    frequencies = list(run_results[list(run_results.keys())[0]].keys())

    return _create_ensemble(best_runs, frequencies, config)


def _create_ensemble(results_files: List[Path], frequencies: List[str], config: Config) -> dict:
    """Averages the predictions of the passed runs and re-calculates metrics."""
    lowest_freq = sort_frequencies(frequencies)[0]
    ensemble_sum = defaultdict(dict)
    target_vars = config.target_variables

    print('Loading results for each run.')
    for run in tqdm(results_files):
        run_results = pickle.load(open(run, 'rb'))
        for basin, basin_results in run_results.items():
            for freq in frequencies:
                freq_results = basin_results[freq]['xr']

                if freq not in ensemble_sum[basin]:
                    ensemble_sum[basin][freq] = freq_results
                else:
                    for target_var in target_vars:
                        ensemble_sum[basin][freq][f'{target_var}_sim'] += freq_results[f'{target_var}_sim']

    print('Combining results and calculating metrics.')
    ensemble = defaultdict(lambda: defaultdict(dict))
    for basin in tqdm(ensemble_sum.keys()):
        for freq in frequencies:
            ensemble_xr = ensemble_sum[basin][freq]
            frequency_factor = int(get_frequency_factor(lowest_freq, freq))
            freq_date_range = pd.date_range(start=ensemble_xr.coords['date'].values[0],
                                            end=ensemble_xr.coords['date'].values[-1]
                                            + pd.Timedelta(days=1, seconds=-1),
                                            freq=freq)
            mask = np.ones(frequency_factor).astype(bool)
            mask[:-len(ensemble_xr.coords['time_step'])] = False
            freq_date_range = freq_date_range[np.tile(
                mask, len(ensemble_xr.coords['date']))]

            ensemble_xr = ensemble_xr.isel(time_step=slice(-frequency_factor, None)) \
                .stack(datetime=['date', 'time_step']) \
                .drop_vars({'datetime', 'date', 'time_step'})
            ensemble_xr['datetime'] = freq_date_range
            for target_var in target_vars:
                ensemble_xr[f'{target_var}_sim'] = ensemble_xr[f'{target_var}_sim'] / \
                    len(results_files)

                sim = ensemble_xr[f'{target_var}_sim']
                if target_var in config.clip_targets_to_zero:
                    sim = xr.where(sim < 0, 0, sim)

                metrics = config.metrics if isinstance(
                    config.metrics, list) else config.metrics[target_var]
                if 'all' in metrics:
                    metrics = get_available_metrics()
                try:
                    ensemble_metrics = calculate_metrics(ensemble_xr[f'{target_var}_obs'],
                                                         sim,
                                                         metrics=metrics,
                                                         resolution=freq)
                except AllNaNError as err:
                    msg = f'Basin {basin} ' \
                        + (f'{target_var} ' if len(target_vars) > 1 else '') \
                        + (f'{freq} ' if len(frequencies) > 1 else '') \
                        + str(err)
                    print(msg)
                    ensemble_metrics = {metric: np.nan for metric in metrics}

                if len(target_vars) > 1:
                    ensemble_metrics = {
                        f'{target_var}_{key}': val for key, val in ensemble_metrics.items()}
                if len(frequencies) > 1:
                    ensemble_metrics = {
                        f'{key}_{freq}': val for key, val in ensemble_metrics.items()}
                for metric, val in ensemble_metrics.items():
                    ensemble[basin][freq][metric] = val

            ensemble[basin][freq]['xr'] = ensemble_xr

    return dict(ensemble)


def _get_percentiles(results: dict, metric='NSE', median_length=5) -> dict:
    """Calculates percentiles with an adjustable median length across all basins."""
    percentiles = {'25th': {}, '50th': {}, '75th': {}}
    key = metric
    frequencies = list(results[list(results.keys())[0]].keys())

    for freq in frequencies:
        if len(frequencies) > 1:
            key = f'{metric}_{freq}'
        metric_values = [v[freq][key] for v in results.values()
                         if freq in v.keys() and key in v[freq].keys()]

        if median_length > 1:
            rolling_median = pd.Series(metric_values).rolling(median_length).median().dropna()
            metric_values = rolling_median.values

        percentiles['25th'][freq] = np.nanpercentile(metric_values, 25)
        percentiles['50th'][freq] = np.nanmedian(metric_values)
        percentiles['75th'][freq] = np.nanpercentile(metric_values, 75)

    return percentiles


def _get_best_validation_runs(run_dirs: List[Path], k: int, epoch: int = None, median_length=5) -> List[Path]:
    """Returns the k run directories with the best median validation metrics."""
    val_files = list(zip(run_dirs, [_get_results_file(
        run_dir, 'validation', epoch) for run_dir in run_dirs]))

    median_sums = {}
    for run_dir, val_file in val_files:
        val_results = pickle.load(open(val_file, 'rb'))
        val_medians = _get_percentiles(val_results, median_length=median_length)['50th']
        print('validation', val_file, val_medians)
        median_sums[run_dir] = sum(val_medians.values())

    if k > len(run_dirs):
        raise ValueError(
            f'best_k k is larger than number of runs {len(val_files)}.')
    return sorted(median_sums, key=median_sums.get, reverse=True)[:k]


def plot_percentiles(results: dict, metric='NSE', median_length=5):
    """Plot the 25th, 50th (median), and 75th percentiles for each metric."""
    percentiles = _get_percentiles(results, metric=metric, median_length=median_length)

    frequencies = percentiles['50th'].keys()
    for freq in frequencies:
        plt.plot(frequencies, percentiles['50th']
                 [freq], label=f'Median ({freq})')
        plt.fill_between(frequencies, percentiles['25th'][freq], percentiles['75th']
                         [freq], alpha=0.3, label=f'25th-75th Percentile Range ({freq})')

    plt.xlabel('Time')
    plt.ylabel(metric)
    plt.legend()
    plt.show()


def main(args):
    run_dirs = [Path(run_dir) for run_dir in args['run_dirs']]
    output_dir = Path(args['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    ensemble_results = create_results_ensemble(
        run_dirs, best_k=args.get('best_k'), metrics=args.get('metrics'), period=args['period'], epoch=args.get('epoch'))

    df = metrics_to_dataframe(ensemble_results, args['metrics'])

    percentiles = _get_percentiles(ensemble_results, median_length=args.get('median_length', 5))
    for freq in percentiles['50th']:
        df[f'{metric}_25th_{freq}'] = percentiles['25th'][freq]
        df[f'{metric}_median_{freq}'] = percentiles['50th'][freq]
        df[f'{metric}_75th_{freq}'] = percentiles['75th'][freq]

    df.to_csv(output_dir / 'metrics.csv', index=False)

    with open(output_dir / 'results.pickle', 'wb') as f:
        pickle.dump(ensemble_results, f)

    if args.get('plot', False):
        plot_percentiles(ensemble_results, metric=args.get(
            'metric', 'NSE'), median_length=args.get('median_length', 5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dirs', nargs='+', required=True, help='List of run directories.')
    parser.add_argument('--output-dir', required=True, help='Output directory to store results.')
    parser.add_argument('--best-k', type=int, help='Use only k best validation runs.')
    parser.add_argument('--metrics', nargs='+', help='Metrics to evaluate.')
    parser.add_argument('--period', choices=['train', 'validation', 'test'], default='test', help='Period to evaluate.')
    parser.add_argument('--epoch', type=int, help='Epoch to use. Default: use best epoch.')
    parser.add_argument('--median-length', type=int, default=5, help='Length of rolling window for calculating medians.')
    parser.add_argument('--plot', action='store_true', help='Whether to plot percentiles.')
    args = vars(parser.parse_args())
    main(args)
