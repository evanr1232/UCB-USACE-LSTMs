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
                 hourly: bool=False,
                 extend_train_period: bool=False,
                 gpu: int = -1,
                 is_mts: bool = False,
                 is_mts_data: bool = False,
                 basin: bool = None):
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

        self._config = None
        self._model = None
        self._predictions = None
        self._observed = None
        self._metrics = None
        self._basin_name = None
        self._target_variable = None

        # EXTRA DEBUG: Path details
        print("[DEBUG:UCB_trainer] => Initializing with:")
        print(f"  path_to_csv_folder={path_to_csv_folder}, yaml_path={yaml_path}")
        print(f"  hyperparams={hyperparams}")
        print(f"  input_features={input_features}, basin={basin}")
        print(f"  is_mts={is_mts}, is_mts_data={is_mts_data}, hourly={hourly}")
        print(f"  physics_informed={physics_informed}, physics_data_file={physics_data_file}")

        self._create_config()

    def train(self):
        """
        Train the model or ensemble based on the specified number of ensemble members.
        """
        if self._num_ensemble_members == 1:
            path = self._train_model()
            print(f"[DEBUG:train] => Single-model run_dir = {path}")
            self._eval_model(path, period="validation")
            self._model = path
        else:
            self._model = self._train_ensemble()
            print("[DEBUG:train] => Ensemble run_dirs =", self._model)
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

        print(f"[DEBUG:results] => period='{period}', mts_trk='{mts_trk}', time_resolution_key='{time_resolution_key}'")
        self._get_predictions(time_resolution_key, period)
        print("[DEBUG:results] => predictions loaded OK")

        self._generate_obs_sim_plt(period)

        self._metrics = calculate_all_metrics(self._observed, self._predictions)
        csv_path = self._generate_csv(period, freq_key=(time_resolution_key if self._is_mts else None))
        return csv_path, self._metrics

    def _eval_model(self, run_directory: Path, period="validation"):
        """Evaluate a trained model. Uses the standard neuralhydrology eval_run."""
        print(f"[DEBUG:_eval_model] => run_directory={run_directory}, period={period}")
        eval_run(run_dir=run_directory, period=period)

    def _get_predictions(self, time_resolution_key, period='validation'):
        """
        Load predictions from the model's result files for the given period.
        """
        if self._num_ensemble_members == 1:
            # Single-model
            results_file = self._model / period / f"model_epoch{str(self._config.epochs).zfill(3)}" / f"{period}_results.p"
            print(f"[DEBUG:_get_predictions] => Looking for single-model results at: {results_file}")

            if not results_file.exists():
                print("[WARN:_get_predictions] => results_file doesn't exist, forcing eval_run")
                self._eval_model(self._model, period)
            if not results_file.exists():
                raise FileNotFoundError(f"Failed to evaluate or locate results for {period} => {results_file}")

            with open(results_file, "rb") as fp:
                results = pickle.load(fp)

            print("[DEBUG:_get_predictions] => results loaded, found basins:", list(results.keys()))
            self._basin_name = next(iter(results.keys()))
            print(f"[DEBUG:_get_predictions] => using basin_name={self._basin_name}")

            self._target_variable = self._config.target_variables[0]
            observed_key = f"{self._target_variable}_obs"
            simulated_key = f"{self._target_variable}_sim"
            print(f"[DEBUG:_get_predictions] => Observed key='{observed_key}', Sim key='{simulated_key}'")

            # Dump aggregator frequency keys/time coords
            print("[DEBUG:_get_predictions] => Dumping aggregator coordinate information...")
            basin_dict = results[self._basin_name]
            print("[DEBUG:_get_predictions] => basin keys =>", list(basin_dict.keys()))
            if time_resolution_key not in basin_dict:
                raise KeyError(
                    f"time_resolution_key '{time_resolution_key}' not in results for basin '{self._basin_name}'. "
                    f"Found keys: {list(basin_dict.keys())}"
                )

            for freq_key_in_dict in basin_dict.keys():
                print(f"   [DEBUG:_get_predictions] => freq_key = {freq_key_in_dict}")
                if "xr" in basin_dict[freq_key_in_dict]:
                    da_keys = list(basin_dict[freq_key_in_dict]["xr"].keys())
                    print(f"       data array keys = {da_keys}")
                    for da_key in da_keys:
                        da = basin_dict[freq_key_in_dict]["xr"][da_key]
                        if "date" in da.coords:
                            print(f"       => {da_key} has {da.coords['date'].shape[0]} timesteps; ")
                            print("          first 3 timestamps =", da.coords["date"].values[:3])
                            print("          last 3 timestamps  =", da.coords["date"].values[-3:])
                else:
                    print(f"       => No 'xr' key in basin_dict[{freq_key_in_dict}].")

            if observed_key not in basin_dict[time_resolution_key]['xr']:
                raise KeyError(f"Observed key '{observed_key}' not found in {time_resolution_key} results.")
            if simulated_key not in basin_dict[time_resolution_key]['xr']:
                raise KeyError(f"Simulated key '{simulated_key}' not found in {time_resolution_key} results.")

            self._observed = basin_dict[time_resolution_key]['xr'][observed_key].isel(time_step=0)
            self._predictions = basin_dict[time_resolution_key]['xr'][simulated_key].isel(time_step=0)
            print("[DEBUG:_get_predictions] => _observed shape:", self._observed.shape,
                  " _predictions shape:", self._predictions.shape)

        else:
            # Ensemble
            print("[DEBUG:_get_predictions] => ENSEMBLE mode. run_dirs:", self._model)
            results = create_results_ensemble(run_dirs=self._model, period=period)
            print("[DEBUG:_get_predictions] => ensemble results loaded. keys =>", list(results.keys()))
            self._basin_name = next(iter(results.keys()))
            self._target_variable = self._config.target_variables[0]
            observed_key = f"{self._target_variable}_obs"
            simulated_key = f"{self._target_variable}_sim"

            if observed_key not in results[self._basin_name][time_resolution_key]['xr']:
                raise KeyError(f"Observed key '{observed_key}' not found in results for basin {self._basin_name}.")
            if simulated_key not in results[self._basin_name][time_resolution_key]['xr']:
                raise KeyError(f"Simulated key '{simulated_key}' not found in results for basin {self._basin_name}.")

            self._observed = results[self._basin_name][time_resolution_key]['xr'][observed_key]
            self._predictions = results[self._basin_name][time_resolution_key]['xr'][simulated_key]

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
        """
        if self._observed is None or self._predictions is None:
            print("[ERROR] Observed or predicted are None => skipping CSV.")
            return Path()

        base_name = f"results_output_{period}"
        if self._is_mts and freq_key:
            base_name += f"_{freq_key}"
        out = self._config.run_dir / f"{base_name}.csv"
        print(f"[DEBUG:_generate_csv] => saving to {out}")

        try:
            # Extra debug
            if self._num_ensemble_members == 1:
                if "date" in self._observed.coords:
                    obs_times = self._observed["date"].values
                    sim_times = self._predictions["date"].values
                    print("[DEBUG:_generate_csv] => aggregator daily time coords (observed) => first 5:", obs_times[:5])
                    print("[DEBUG:_generate_csv] => aggregator daily time coords (simulated) => first 5:", sim_times[:5])
                    df = pd.DataFrame({
                        "Date": self._observed["date"].values,
                        "Observed": self._observed.values,
                        "Predicted": self._predictions.values
                    })
                else:
                    print("[WARN:_generate_csv] => date coord not found, fallback indexing.")
                    df = pd.DataFrame({
                        "Date": range(len(self._observed)),
                        "Observed": self._observed.values,
                        "Predicted": self._predictions.values
                    })
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
        print("[DEBUG:_train_model] => hyperparams:", self._hyperparams)
        start_training(self._config)
        print("[DEBUG:_train_model] => training done => run_dir:", self._config.run_dir)
        return self._config.run_dir

    def _train_ensemble(self) -> List[Path]:
        """Train multiple models as an ensemble."""
        print("[DEBUG:_train_ensemble] => num_ensemble_members=", self._num_ensemble_members)
        run_dirs = []
        for i in range(self._num_ensemble_members):
            path = self._train_model()
            run_dirs.append(path)
            print(f"[DEBUG:_train_ensemble] => ensemble member {i} => path={path}")

        # Evaluate each on validation/test if you want
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

        if self._physics_informed:
            if self._physics_data_file:
                config.update_config({'physics_data_file': self._physics_data_file}, dev_mode=True)
            else:
                raise ValueError("Physics-informed is enabled, but no physics data file was provided.")

        # GPU or CPU
        if self._gpu == 0:
            selected_device = "cuda:0"
            print("[UCB Trainer] Using CUDA device: 'cuda:0'")
        elif self._gpu == -2:
            selected_device = "cpu"
            print("[UCB Trainer] Forcing CPU (gpu=-2).")
        else:
            selected_device = "cpu"
            print(f"[UCB Trainer] Using CPU (unhandled gpu={self._gpu}).")

        config.update_config({'device': selected_device}, dev_mode=True)

        self._config = config

        if self._config.epochs % self._config.save_weights_every != 0:
            raise ValueError(
                f"The 'save_weights_every' parameter must divide 'epochs' evenly. "
                f"Got epochs={self._config.epochs}, save_weights_every={self._config.save_weights_every}."
            )

        return config