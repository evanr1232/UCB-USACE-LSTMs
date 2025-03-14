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

    def __init__(self, path_to_csv_folder: Path, yaml_path: Path, hyperparams: dict,
                 input_features: List[str] = None, num_ensemble_members: int = 1,
                 physics_informed: bool = False, physics_data_file: Path = None,
                 hourly: bool=False, extend_train_period: bool=False, gpu: int = -1, is_mts: bool = False):
        """
        Initialize the UCB_trainer class with configurations and training parameters.

        Args:
            path_to_csv_folder (Path): Path to the folder containing training data.
            yaml_path (Path): Path to the YAML configuration file.
            hyperparams (dict): Dictionary of hyperparameters for training.
            input_features (List[str], optional): List of input feature names. Defaults to None.
            num_ensemble_members (int, optional): Number of ensemble models to train. Defaults to 1.
            physics_informed (bool, optional): Whether to include physics-informed inputs. Defaults to False.
                                               If True, physics_data_file must be provided.
            physics_data_file (Path, optional): Path to physics data file. Defaults to None.
            hourly (bool, optional): Whether to use hourly data. Defaults to False.
            extend_train_period (bool, optional): Extend training period to include validation. Defaults to False.
            gpu (int, optional): GPU ID for training. Defaults to -1 (CPU).
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

        self._config = None
        self._create_config()

        self._model = None
        self._predictions = None
        self._observed = None
        self._metrics = None
        self._basin_name = None
        self._target_variable = None

    def train(self):
        """
        Train the model or ensemble based on the specified number of ensemble members.

        Returns:
            Path or List[Path]: The path(s) to the trained model(s).
        """
        if self._num_ensemble_members == 1:
            self._model = self._train_model()
            # Evaluate on validation
            self._eval_model(self._model, period="validation")
        else:
            self._model = self._train_ensemble()
            for model in self._model:
                self._eval_model(model, period="validation")
        return self._model

    def results(self, period='validation', mts_trk = "1H") -> dict:
        """
        Public method to return metrics and optionally plot data visualizations of model performance.

        Args:
            period (str): 'validation', 'test', or 'train'. Defaults to 'validation'.
        """
        if self._is_mts:
            time_resolution_key = mts_trk
        else:
            time_resolution_key = '1h' if self._hourly else '1D'

        # Load predictions from file
        self._get_predictions(time_resolution_key, period)
        print('got predictions')

        # Generate plot
        self._generate_obs_sim_plt(period)

        # Compute metrics
        self._metrics = calculate_all_metrics(self._observed, self._predictions)

        # Save CSV
        path = self._generate_csv(period, freq_key=time_resolution_key if self._is_mts else None)
        return path, self._metrics

    def _eval_model(self, run_directory, period="validation"):
        """
        Evaluate a trained model. Uses the standard neuralhydrology eval_run.

        Args:
            run_directory (Path): Path to the trained model directory.
            period (str, optional): 'validation', 'train', or 'test'.
        """
        eval_run(run_dir=run_directory, period=period)

    def _get_predictions(self, time_resolution_key, period='validation'):
        """
        Private method to load predictions from the model's result files.

        Args:
            time_resolution_key (str): '1h' or '1D'.
            period (str): 'validation', 'test', or 'train'.
        """
        if self._num_ensemble_members == 1:
            results_file = self._model / period / f"model_epoch{str(self._config.epochs).zfill(3)}" / f"{period}_results.p"

            # If results file doesn't exist, evaluate the model
            if not results_file.exists():
                self._eval_model(self._model, period)

            if not results_file.exists():
                raise FileNotFoundError(f"Failed to evaluate or locate results for {period}. Expected file at: {results_file}")

            with open(results_file, "rb") as fp:
                results = pickle.load(fp)

            self._basin_name = next(iter(results.keys()))
            self._target_variable = self._config.target_variables[0]

            observed_key = f"{self._target_variable}_obs"
            simulated_key = f"{self._target_variable}_sim"

            if observed_key not in results[self._basin_name][time_resolution_key]['xr']:
                raise KeyError(f"Observed key '{observed_key}' not found in results for basin {self._basin_name}.")
            if simulated_key not in results[self._basin_name][time_resolution_key]['xr']:
                raise KeyError(f"Simulated key '{simulated_key}' not found in results for basin {self._basin_name}.")

            self._observed = results[self._basin_name][time_resolution_key]['xr'][observed_key].isel(time_step=0)
            self._predictions = results[self._basin_name][time_resolution_key]['xr'][simulated_key].isel(time_step=0)

        else:  # Ensemble case
            results = create_results_ensemble(run_dirs=self._model, period=period)
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
        Private method to plot observed vs simulated values.
        """
        fig, ax = plt.subplots(figsize=(16, 10))

        # Label depends on whether it's physics or not
        if self._physics_informed:
            simulated_label = "HybridSimulation"
        else:
            simulated_label = "Simulated"

        # Single-model
        if self._num_ensemble_members == 1:
            ax.plot(self._observed["date"], self._observed, label="Observed", linewidth=1.5)
            ax.plot(self._predictions["date"], self._predictions, label=simulated_label, linewidth=1.5)
        else:
            # Ensemble
            ax.plot(self._observed["datetime"], self._observed, label="Observed", linewidth=1.5)
            ax.plot(self._predictions["datetime"], self._predictions, label=simulated_label, linewidth=1.5)

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
        Save predictions to a CSV file in the run directory.

        If is_mts=True and freq_key is "1H" vs "1D", we can optionally suffix the CSV name.
        """
        if self._observed is None or self._predictions is None:
            print("[ERROR] Observed or predicted are None. Not saving CSV.")
            return Path()

        base_name = f"results_output_{period}"

        if self._is_mts and freq_key:
            base_name += f"_{freq_key}"

        output_path = Path(self._config.run_dir) / f"{base_name}.csv"

        try:
            if self._num_ensemble_members == 1:
                dates = self._observed["date"].values
            else:
                dates = self._observed["datetime"].values

            df = pd.DataFrame({
                "Date": dates,
                "Observed": self._observed.values,
                "Predicted": self._predictions.values
            })
            df.to_csv(output_path, index=False)
            print(f"[INFO] CSV output saved at: {output_path}")
        except Exception as exc:
            print(f"[ERROR] Could not save CSV. Reason: {exc}")

        return output_path

    def _train_model(self) -> Path:
        """
        Train a single model instance.
        """
        start_training(self._config)
        return self._config.run_dir

    def _train_ensemble(self) -> List[Path]:
        """
        Train multiple models as an ensemble.
        """
        paths = []
        for _ in range(self._num_ensemble_members):
            path = self._train_model()
            paths.append(path)
        for path in paths:
            self._eval_model(path, period="validation")
            self._eval_model(path, period="test")

        return paths

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
        # elif self._gpu == -1:
        #     if platform.system() == "Darwin" and torch.backends.mps.is_available():
        #         selected_device = "mps"
        #         print("[UCB Trainer] Using MPS device on Apple Silicon.")
        #     else:
        #         selected_device = "cpu"
        #         print("[UCB Trainer] Using CPU (auto-detect fallback).")
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