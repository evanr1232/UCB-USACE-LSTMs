from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import logging
import pandas as pd
from typing import List

from neuralhydrology.utils.config import Config
from neuralhydrology.training.train import start_training
from neuralhydrology.nh_run import eval_run
from neuralhydrology.utils.nh_results_ensemble import create_results_ensemble
from neuralhydrology.evaluation.metrics import calculate_all_metrics

class UCB_trainer:
    """
    A class to handle training, evaluation, and configuration of neural hydrology models.
    """
    
    def __init__(self, path_to_csv_folder: Path, yaml_path: Path, hyperparams: dict, 
                 input_features: List[str] = None, num_ensemble_members: int = 1, 
                 physics_informed: bool = False, physics_data_file: Path = None, hourly: bool=False, extend_train_period: bool=False, gpu: int = -1):
        """
        Initialize the UCB_trainer class with configurations and training parameters.

        Args:
            path_to_csv_folder (Path): Path to the folder containing training data.
            yaml_path (Path): Path to the YAML configuration file.
            hyperparams (dict): Dictionary of hyperparameters for training.
            input_features (List[str], optional): List of input feature names. Defaults to None.
            num_ensemble_members (int, optional): Number of ensemble models to train. Defaults to 1.
            physics_informed (bool, optional): Whether to include physics-informed inputs. Defaults to False.
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
            self._eval_model(self._model, period="validation")
        else:
            self._model = self._train_ensemble()
            for model in self._model:
                self._eval_model(model, period="validation")
        return self._model
    
    def results(self, period='validation') -> dict:
        """
        Public method to return metrics and plot data visualizations of model preformance.
        """
        time_resolution_key = '1h' if self._hourly else '1D'

        self._get_predictions(time_resolution_key, period)
        print('got predictions')
        self._generate_obs_sim_plt(period)
        self._metrics = calculate_all_metrics(self._observed, self._predictions)
        self._generate_csv(period)
        return self._metrics
    
    def _eval_model(self, run_directory, period="validation"):
        """
        Evaluate a trained model.

        Args:
            run_directory (Path): Path to the trained model.
            period (str, optional): Evaluation period. Defaults to 'validation'.
        """
        eval_run(run_dir=run_directory, period=period)
        
    
    def _get_predictions(self, time_resolution_key, period='validation') -> List:
        """
        Private method to get and return a list of predicted values from trained models.
        Supports both single-model and ensemble cases.
        
        Args:
            time_resolution_key (str): The time resolution key, e.g., '1h' or '1D'.
            period (str): The evaluation period, defaults to 'validation'.
        
        Returns:
            List: A list containing the predictions, where each element corresponds to a model.
        """
        
        if self._num_ensemble_members == 1:
            results_file = self._model / period / f"model_epoch{str(self._config.epochs).zfill(3)}" / f"{period}_results.p"
            
            if not results_file.exists():
                self._eval_model(self._model, period)
            
            if not results_file.exists():
                raise FileNotFoundError(f"Failed to evaluate or locate results for {period}. Expected file at: {results_file}")
            
            with open(results_file, "rb") as fp:
                results = pickle.load(fp)
            #logging.info(f"Results structure: {results}") #DEBUG
            
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
        
        if self._num_ensemble_members > 1:
            results = create_results_ensemble(run_dirs=self._model, period=period)
            #logging.info(f"Results structure: {results}") #DEBUG
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
        return
    
    def _generate_obs_sim_plt(self, period='validation'):
        """
        #needs to be cleaned up
        Private method to plot observed and simulated values over time with improved aesthetics and dynamic labels.
        """
        #setup plot
        fig, ax = plt.subplots(figsize=(16, 10))
        if self._physics_informed:
            simulated_label = 'HybridSimulation'
        else: simulated_label = 'Simulated'
        if self._num_ensemble_members == 1:
            ax.plot(self._observed["date"], self._observed, label="Observed", linewidth=1.5)
            ax.plot(self._predictions["date"], self._predictions, label=simulated_label, linewidth=1.5)
        else:
            ax.plot(self._observed["datetime"], self._observed, label="Observed", linewidth=1.5)
            logging.info(f"Predictions: {self._predictions}")
            ax.plot(self._predictions["datetime"], self._predictions, label=simulated_label, linewidth=1.5)

        #dynamic labels and title using stored target variable and basin name
        ax.set_ylabel(f"{self._target_variable} (units)", fontsize=14)
        ax.set_xlabel("Date", fontsize=14)
        ax.set_title(f"{self._basin_name} - {self._target_variable} Over Time ({period} period)", fontsize=16) #change this

        ax.legend(fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.7)
        
        fig.autofmt_xdate() #date formatting

        plt.tight_layout()
        plt.show()
        return
    
    def _generate_csv(self, period='validation'):
        """
        Private method to generate a CSV file of observed and predicted values. Used in the .results() function.
        :return:
        """
        if self._observed is None or self._predictions is None:
            print("[ERROR] Observed or predicted values are None. Cannot generate CSV.")
            return

        try:
            if self._num_ensemble_members == 1:
                dates = self._observed['date'].values
            else:
                dates = self._observed['datetime'].values

            df = pd.DataFrame({
                'Date': dates,
                'Observed': self._observed.values,
                'Predicted': self._predictions.values
            })

            output_path = Path(self._config.run_dir) / f"results_output_{period}.csv"
            df.to_csv(output_path, index=False)
            print(f"[INFO] CSV output saved at: {output_path}")
        except Exception as e:
            print(f"[ERROR] Failed to generate CSV: {e}")

        return output_path

    
    def _train_model(self) -> Path:
        """
        Train a single model instance.

        Returns:
            Path: Path to the trained model's directory.
        """

        start_training(self._config)
        return self._config.run_dir

    def _train_ensemble(self) -> List[Path]:
        """
        Train multiple models as an ensemble.

        Returns:
            List[Path]: A list of directories containing trained models.
        """
        paths = []
        for _ in range(self._num_ensemble_members):
            path = self._train_model()
            paths.append(path)
        return paths
    
    def _create_config(self) -> Config:
        """
        Private method to create Configuration object for training from user specifications.
        """
        if not self._yaml_path.exists():
            raise FileNotFoundError(f"YAML configuration file not found: {self._yaml_path}")

        # Load the base configuration from the provided YAML file
        config = Config(self._yaml_path)

        # Ensure 'save_weights_every' is set ##not sure if this is necessary
        if 'save_weights_every' not in self._hyperparams:
            self._hyperparams['save_weights_every'] = self._hyperparams['epochs']
        
        # Update dynamic inputs if provided
        if self._dynamic_inputs is not None: 
            config.update_config({'dynamic_inputs': self._dynamic_inputs})

        # Extend train period if specified
        if self._extended_train_period:
            config.update_config({'train_end_date': config.validation_end_date})

        # Update other hyperparameters
        config.update_config(self._hyperparams)
        config.update_config({'data_dir': self._data_dir})
        config.update_config({'physics_informed': self._physics_informed})
        config.update_config({'hourly': self._hourly})

        # Handle physics-informed setup
        if self._physics_informed:
            if self._physics_data_file:
                config.update_config({'physics_data_file': self._physics_data_file})
            else:
                raise ValueError("Physics-informed is enabled, but no physics data file was provided.")

        # Set the device here
        if self._gpu is not None and self._gpu >= 0:
            config.update_config({'device': f"cuda:{self._gpu}"})
        else:
            config.update_config({'device': "cpu"})

        # Store the config object
        self._config = config

        # Validate config values
        if self._config.epochs % self._config.save_weights_every != 0:
            raise ValueError(
                f"The 'save_weights_every' parameter must divide the 'epochs' parameter evenly. "
                f"Ensure 'epochs' is a multiple of {self._config.save_weights_every} to use the most recent weights for the final model."
            )
        return