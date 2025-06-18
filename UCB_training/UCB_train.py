from pathlib import Path
import platform

import matplotlib.pyplot as plt
import pickle
import logging
import pandas as pd
from typing import List

import torch
from neuralhydrology.utils.config import Config
from neuralhydrology.training.train import start_training
from neuralhydrology.nh_run import eval_run
from neuralhydrology.utils.nh_results_ensemble import create_results_ensemble
from neuralhydrology.evaluation.metrics import calculate_all_metrics


class UCB_trainer:
    """
    A wrapper to facilitate easier training/evaluation of neural hydrology models.
    """

    def __init__(self,
                 path_to_csv_folder: Path,
                 yaml_path: Path,
                 hyperparams: dict,
                 input_features: List[str] = None,
                 num_ensemble_members: int = 1,
                 physics_informed: bool = False,
                 physics_data_file: Path = None,
                 hourly: bool = False,
                 extend_train_period: bool = False,
                 gpu: int = -1,
                 is_mts: bool = False,
                 is_mts_data: bool = False,
                 basin: bool = None,
                 verbose: bool = True):
        """
        Initialize the UCB_trainer class with configurations and training parameters.
        """
        self._hyperparams = hyperparams
        self._num_ensemble_members = num_ensemble_members
        self._physics_informed = physics_informed
        self._physics_data_file = physics_data_file
        self._gpu = gpu
        self._data_dir = path_to_csv_folder
        self._dynamic_inputs = input_features
        self._yaml_path = yaml_path
        self._hourly = hourly
        self._extended_train_period = extend_train_period
        self._is_mts = is_mts
        self._is_mts_data = is_mts_data
        self._basin = basin
        self._verbose = verbose

        self._config = None
        self._model = None
        self._predictions = None
        self._observed = None
        self._metrics = None
        self._basin_name = None
        self._target_variable = None

        # print("[DEBUG:UCB_trainer] => Initializing with:")
        # print(f"path_to_csv_folder={path_to_csv_folder}, yaml_path={yaml_path}")
        # print(f"hyperparams={hyperparams}")
        # print(f"input_features={input_features}, basin={basin}")
        # print(f"is_mts={is_mts}, is_mts_data={is_mts_data}, hourly={hourly}")
        # print(f"physics_informed={physics_informed}, physics_data_file={physics_data_file}")

        self._create_config()

    def train(self):
        """
        Train the model or ensemble based on the specified number of ensemble members.
        """
        if self._num_ensemble_members == 1:
            path = self._train_model()
            # print(f"[DEBUG:train] => Single-model run_dir = {path}")
            self._eval_model(path, period="validation")
            self._model = path
        else:
            self._model = self._train_ensemble()
            # print("[DEBUG:train] => Ensemble run_dirs =", self._model)
            for model_path in self._model:
                self._eval_model(model_path, period="validation")

        return self._model

    def results(self, period='validation', mts_trk="1H") -> (Path, dict):
        """
        Public method to return a CSV path and a metrics dict for a given period.
        """
        if self._is_mts:
            time_resolution_key = mts_trk
        else:
            time_resolution_key = '1h' if self._hourly else '1D'

        # print(f"[DEBUG:rgesults] => period='{period}', mts_trk='{mts_trk}', time_resolution_key='{time_resolution_key}'")
        self._get_predictions(time_resolution_key, period)
        # print("[DEBUG:results] => predictions loaded OK")

        if self._verbose:
            self._generate_obs_sim_plt(period)

        self._metrics = calculate_all_metrics(self._observed, self._predictions)
        csv_path = self._generate_csv(period, freq_key=(time_resolution_key if self._is_mts else None))
        return csv_path, self._metrics

    def _eval_model(self, run_directory: Path, period="validation"):
        # print(f"[DEBUG:_eval_model] => run_directory={run_directory}, period={period}")
        eval_run(run_dir=run_directory, period=period)

        results_file = run_directory / period / f"model_epoch{str(self._config.epochs).zfill(3)}" / f"{period}_results.p"
        if results_file.exists():
            with open(results_file, "rb") as fp:
                results = pickle.load(fp)
            basin_name = list(results.keys())[0]
            aggregator_keys = list(results[basin_name].keys())
            # print(f"[DEBUG] => aggregator keys for {run_directory.name}:{period} => {aggregator_keys}")
            for aggregator_key in aggregator_keys:
                xr_dict = results[basin_name][aggregator_key]["xr"]
                for var_name in xr_dict:
                    da = xr_dict[var_name]
                    # if "time_step" in da.dims:
                    #     print(f"[DEBUG] => {run_directory.name}:{period}:{aggregator_key}:{var_name} "
                    #           f"shape={da.shape}, time_step_len={da.sizes['time_step']}")
                    # else:
                    #     print(f"[DEBUG] => {run_directory.name}:{period}:{aggregator_key}:{var_name} shape={da.shape}")
        else:
            print(f"[WARN] => results_file not found: {results_file}")

    def _get_predictions(self, time_resolution_key, period='validation'):
        """
        Load predictions from the model's result files for the given period.
        Only apply multi-timescale flattening logic if self._is_mts is True.
        Otherwise, keep old behavior.
        """
        if self._num_ensemble_members == 1:
            # Single-model
            results_file = self._model / period / f"model_epoch{str(self._config.epochs).zfill(3)}" / f"{period}_results.p"
            if not results_file.exists():
                self._eval_model(self._model, period)
            if not results_file.exists():
                raise FileNotFoundError(f"Failed to evaluate or locate results for {period} => {results_file}")

            with open(results_file, "rb") as fp:
                results = pickle.load(fp)

            self._basin_name = next(iter(results.keys()))
            basin_dict = results[self._basin_name]

            self._target_variable = self._config.target_variables[0]
            observed_key = f"{self._target_variable}_obs"
            simulated_key = f"{self._target_variable}_sim"

            if time_resolution_key not in basin_dict:
                raise KeyError(
                    f"time_resolution_key '{time_resolution_key}' not in results for basin '{self._basin_name}'. "
                    f"Found keys: {list(basin_dict.keys())}"
                )

            xr_dict = basin_dict[time_resolution_key]["xr"]
            if observed_key not in xr_dict or simulated_key not in xr_dict:
                raise KeyError(
                    f"Missing '{observed_key}' or '{simulated_key}' in aggregator "
                    f"'{time_resolution_key}' for basin '{self._basin_name}'."
                )

            obs_da = xr_dict[observed_key]
            sim_da = xr_dict[simulated_key]

            if self._is_mts:
                if time_resolution_key == "1D":
                    if "time_step" in obs_da.dims:
                        obs_da = obs_da.isel(time_step=0)
                        sim_da = sim_da.isel(time_step=0)
                    # print("[DEBUG:_get_predictions] => MTS 1D aggregator final shape:", obs_da.shape)

                elif time_resolution_key == "1H":
                    if "time_step" in obs_da.dims:
                        obs_da = obs_da.stack(stacked_time=("date", "time_step"))
                        sim_da = sim_da.stack(stacked_time=("date", "time_step"))
                        obs_da = obs_da.rename({"stacked_time": "time"})
                        sim_da = sim_da.rename({"stacked_time": "time"})
                        # print("[DEBUG:_get_predictions] => MTS 1H aggregator flattened shape:", obs_da.shape)
                    else:
                        print("[WARN] => The 1H aggregator has no 'time_step' dimension? shape=", obs_da.shape)
                else:
                    print("[DEBUG:_get_predictions] => MTS ignoring unknown aggregator key:", time_resolution_key)

            else:
                if "time_step" in obs_da.dims:
                    obs_da = obs_da.isel(time_step=0)
                    sim_da = sim_da.isel(time_step=0)
                # print("[DEBUG:_get_predictions] => Non-MTS shape after old approach:", obs_da.shape)

            self._observed = obs_da
            self._predictions = sim_da

            # print("[DEBUG:_get_predictions] => final observed shape:", self._observed.shape,
            #       " final predicted shape:", self._predictions.shape)

        else:
            # Ensemble
            results = create_results_ensemble(run_dirs=self._model, period=period)
            self._basin_name = next(iter(results.keys()))
            basin_dict = results[self._basin_name]

            self._target_variable = self._config.target_variables[0]
            observed_key = f"{self._target_variable}_obs"
            simulated_key = f"{self._target_variable}_sim"

            if time_resolution_key not in basin_dict:
                raise KeyError(
                    f"time_resolution_key '{time_resolution_key}' not in ensemble results for "
                    f"basin '{self._basin_name}'. Found keys: {list(basin_dict.keys())}"
                )

            xr_dict = basin_dict[time_resolution_key]["xr"]
            obs_da = xr_dict[observed_key]
            sim_da = xr_dict[simulated_key]

            if self._is_mts:
                # Same MTS logic for ensembles
                if time_resolution_key == "1D":
                    if "time_step" in obs_da.dims:
                        obs_da = obs_da.isel(time_step=0)
                        sim_da = sim_da.isel(time_step=0)
                elif time_resolution_key == "1H":
                    if "time_step" in obs_da.dims:
                        obs_da = obs_da.stack(stacked_time=("date", "time_step"))
                        sim_da = sim_da.stack(stacked_time=("date", "time_step"))
                        obs_da = obs_da.rename({"stacked_time": "time"})
                        sim_da = sim_da.rename({"stacked_time": "time"})

            else:  # (1) #  Ensemble logic
                results = create_results_ensemble(run_dirs=self._model, period=period)
                self._basin_name = next(iter(results.keys()))
                basin_dict = results[self._basin_name]

                # print(f"[DEBUG:ensemble] => create_results_ensemble returned basins: {list(results.keys())}")
                # print(f"[DEBUG:ensemble] => aggregator keys for basin '{self._basin_name}': {list(basin_dict.keys())}")

                self._target_variable = self._config.target_variables[0]
                observed_key = f"{self._target_variable}_obs"
                simulated_key = f"{self._target_variable}_sim"

                if time_resolution_key not in basin_dict:
                    raise KeyError(
                        f"time_resolution_key '{time_resolution_key}' not in ensemble results for "
                        f"basin '{self._basin_name}'. Found keys: {list(basin_dict.keys())}"
                    )

                xr_dict = basin_dict[time_resolution_key]["xr"]
                obs_da = xr_dict[observed_key]
                sim_da = xr_dict[simulated_key]

                # print(f"[DEBUG:ensemble] => aggregator: '{time_resolution_key}'")
                # print(f"[DEBUG:ensemble] => obs_da shape: {obs_da.shape}, dims: {obs_da.dims}")
                # print(f"[DEBUG:ensemble] => sim_da shape: {sim_da.shape}, dims: {sim_da.dims}")

                # If this is a simple hourly or daily run (not multi‐timescale),
                # often the aggregator has a single (or zero-length) "time_step" dimension.
                # We revert to the old approach and ignore that dimension if it’s present:
                if "time_step" in obs_da.dims:
                    # print("[DEBUG:ensemble] => 'time_step' in dims, isel(time_step=0)")
                    obs_da = obs_da.isel(time_step=0)
                    sim_da = sim_da.isel(time_step=0)

                # print(f"[DEBUG:ensemble] => final obs_da shape: {obs_da.shape}")
                # print(f"[DEBUG:ensemble] => final sim_da shape: {sim_da.shape}")

                # now just assign to self
                self._observed = obs_da
                self._predictions = sim_da

                # print(f"[DEBUG:ensemble] => assigned self._observed shape: {self._observed.shape}, "
                #       f"self._predictions shape: {self._predictions.shape}")

    def _generate_obs_sim_plt(self, period='validation'):
        """
        Plot observed vs. simulated values (matplotlib).
        """
        if self._observed is None or self._predictions is None:
            print("[WARN:_generate_obs_sim_plt] => cannot plot, observed or predictions = None")
            return

        fig, ax = plt.subplots(figsize=(16, 10))
        if self._physics_informed:
            simulated_label = "HybridSimulation"
        else:
            simulated_label = "Simulated"

        if self._num_ensemble_members == 1:
            # single model
            if "date" in self._observed.coords:
                ax.plot(self._observed["date"], self._observed, label="Observed", linewidth=1.5)
                ax.plot(self._predictions["date"], self._predictions, label=simulated_label, linewidth=1.5)
            else:
                print("[WARN:_generate_obs_sim_plt] => 'date' not in coords, cannot plot easily.")
        else:
            # ensemble
            pass

        ax.set_ylabel(f"{self._target_variable} (units)", fontsize=14)
        ax.set_xlabel("Date", fontsize=14)
        ax.set_title(f"{self._basin_name} - {self._target_variable} Over Time ({period})", fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.7)
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

    def _generate_csv(self, period='validation', freq_key: str = None) -> Path:
        """
        Save predictions to a CSV file in the run directory, e.g. "results_output_validation_1H.csv".
        Only do special flattening/timestamp logic if self._is_mts == True and freq_key == '1H'.
        """
        if self._observed is None or self._predictions is None:
            print("[ERROR] Observed or predicted are None => skipping CSV.")
            return Path()

        base_name = f"results_output_{period}"
        if self._is_mts and freq_key:
            base_name += f"_{freq_key}"
        out = self._config.run_dir / f"{base_name}.csv"
        # print(f"[DEBUG:_generate_csv] => saving to {out}")

        # --- Debug prints for shapes & coords ---
        # print("[DEBUG:_generate_csv] => Observed DataArray dims:", self._observed.dims)
        # print("[DEBUG:_generate_csv] => Observed DataArray shape:", self._observed.shape)
        # for coord_name in self._observed.coords:
        #     coord_vals = self._observed.coords[coord_name].values
        #     if coord_vals.ndim == 0:
        #         print(f"[DEBUG:_generate_csv] => coord '{coord_name}' is scalar:", coord_vals.item())
        #     else:
        #         print(f"[DEBUG:_generate_csv] => coord '{coord_name}' has shape {coord_vals.shape} -> first 5:", coord_vals[:5])

        # print("----------------------------------------------------------")
        # print("[DEBUG:_generate_csv] => Predictions DataArray dims:", self._predictions.dims)
        # print("[DEBUG:_generate_csv] => Predictions DataArray shape:", self._predictions.shape)
        for coord_name in self._predictions.coords:
            coord_vals = self._predictions.coords[coord_name].values
        #     if coord_vals.ndim == 0:
        #         print(f"[DEBUG:_generate_csv] => coord '{coord_name}' is scalar:", coord_vals.item())
        #     else:
        #         print(f"[DEBUG:_generate_csv] => coord '{coord_name}' has shape {coord_vals.shape} -> first 5:", coord_vals[:5])
        # print("----------------------------------------------------------")

        try:
            # SINGLE-MODEL RUN
            if self._num_ensemble_members == 1:
                # MTS 1H
                if self._is_mts and freq_key == "1H":
                    # print("[DEBUG:_generate_csv] => MTS 1H CSV logic: create an hourly timestamp.")

                    obs_df = self._observed.reset_index(self._observed.dims).to_dataframe(name="Observed")
                    sim_df = self._predictions.reset_index(self._predictions.dims).to_dataframe(name="Predicted")

                    merged_df = obs_df.join(sim_df, how="inner", lsuffix="_obs", rsuffix="_sim")

                    if "date_obs" in merged_df.columns and "time_step_obs" in merged_df.columns:
                        merged_df["Date"] = pd.to_datetime(merged_df["date_obs"]) \
                                            + pd.to_timedelta(merged_df["time_step_obs"], unit="h")

                        final_df = merged_df[["Date", "Observed", "Predicted"]].sort_values("Date")

                    elif "time" in merged_df.columns and merged_df["time"].dtype == "object":
                        def combine_tuple(tup):
                            base_ts, hour_offset = tup
                            return pd.to_datetime(base_ts) + pd.to_timedelta(hour_offset, unit="h")

                        merged_df["Date"] = merged_df["time"].apply(combine_tuple)
                        final_df = merged_df[["Date", "Observed", "Predicted"]].sort_values("Date")

                    else:
                        # print("[DEBUG:_generate_csv] => aggregator appears to have a direct time dimension")
                        possible_timecols = [c for c in merged_df.columns if "time" in c]
                        if possible_timecols:
                            time_col = possible_timecols[0]
                            merged_df.rename(columns={time_col: "Date"}, inplace=True)
                            final_df = merged_df[["Date", "Observed", "Predicted"]].sort_values("Date")
                        else:
                            # print("[WARN] => no recognized 'time' col; output as is.")
                            final_df = merged_df[["Observed", "Predicted"]]

                    df = final_df.reset_index(drop=True)

                # MTS 1D or LOCAL NOTEBOOKS
                else:
                    if "date" in self._observed.coords:
                        obs_times = self._observed["date"].values
                        sim_times = self._predictions["date"].values
                        # print("[DEBUG:_generate_csv] => aggregator 'date' => first few:", obs_times[:5])

                        df = pd.DataFrame({
                            "Date": obs_times,
                            "Observed": self._observed.values,
                            "Predicted": self._predictions.values
                        })
                    else:
                        # print("[WARN:_generate_csv] => 'date' coord not found, fallback indexing.")
                        df = pd.DataFrame({
                            "Date": range(len(self._observed)),
                            "Observed": self._observed.values,
                            "Predicted": self._predictions.values
                        })
            # ENSEMBLE
            else:
                df = pd.DataFrame({
                    "Date": self._observed["datetime"].values,
                    "Observed": self._observed.values,
                    "Predicted": self._predictions.values
                })

            df.to_csv(out, index=False)

        except Exception as exc:
            print(f"[ERROR:_generate_csv] => Could not save CSV: {exc}")

        return out

    def _train_model(self) -> Path:
        """Train a single model instance with start_training()."""
        # print("[DEBUG:_train_model] => hyperparams:", self._hyperparams)
        start_training(self._config)
        # print("[DEBUG:_train_model] => training done => run_dir:", self._config.run_dir)
        return self._config.run_dir

    def _train_ensemble(self) -> List[Path]:
        """Train multiple models as an ensemble."""
        # print("[DEBUG:_train_ensemble] => num_ensemble_members=", self._num_ensemble_members)
        run_dirs = []
        for i in range(self._num_ensemble_members):
            path = self._train_model()
            run_dirs.append(path)
            # print(f"[DEBUG:_train_ensemble] => ensemble member {i} => path={path}")

        for rd in run_dirs:
            self._eval_model(rd, period="validation")
            self._eval_model(rd, period="test")

        return run_dirs

    def _create_config(self) -> Config:
        """
        Private method to create a Configuration object in dev_mode,
        ensuring the saved config also contains 'dev_mode: True'.
        """
        if not self._yaml_path.exists():
            raise FileNotFoundError(f"YAML configuration file not found: {self._yaml_path}")

        config = Config(self._yaml_path, dev_mode=True)

        if 'save_weights_every' not in self._hyperparams:
            self._hyperparams['save_weights_every'] = self._hyperparams['epochs']

        if self._dynamic_inputs is not None:
            config.update_config({'dynamic_inputs': self._dynamic_inputs}, dev_mode=True)

        if self._extended_train_period:
            config.update_config({'train_end_date': config.validation_end_date}, dev_mode=True)

        config.update_config(self._hyperparams, dev_mode=True)
        config.update_config({'data_dir': self._data_dir}, dev_mode=True)
        config.update_config({'physics_informed': self._physics_informed}, dev_mode=True)
        config.update_config({'hourly': self._hourly}, dev_mode=True)
        config.update_config({'is_mts': self._is_mts}, dev_mode=True)
        config.update_config({'is_mts_data': self._is_mts_data}, dev_mode=True)
        config.update_config({'verbose': self._verbose}, dev_mode=True)

        if self._physics_informed:
            if self._physics_data_file:
                config.update_config({'physics_data_file': self._physics_data_file}, dev_mode=True)
            else:
                raise ValueError("Physics-informed is enabled, but no physics data file was provided.")

        # GPU or CPU
        if self._gpu == 0:
            selected_device = "cuda:0"
            if self._verbose:
                print("[UCB Trainer] Using CUDA device: 'cuda:0'")
        elif self._gpu == -2:
            selected_device = "cpu"
            if self._verbose:
                print("[UCB Trainer] Forcing CPU (gpu=-2).")
        else:
            selected_device = "cpu"
            if self._verbose:
                print(f"[UCB Trainer] Using CPU (unhandled gpu={self._gpu}).")

        config.update_config({'device': selected_device}, dev_mode=True)

        self._config = config

        if self._config.epochs % self._config.save_weights_every != 0:
            raise ValueError(
                f"The 'save_weights_every' parameter must divide 'epochs' evenly. "
                f"Got epochs={self._config.epochs}, save_weights_every={self._config.save_weights_every}."
            )

        return config
