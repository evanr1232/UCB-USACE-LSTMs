'''
purpose: 
- abstract away:
    - editing config (yaml) file (currently need to modify yaml file with hyperparamers)
    - whether or not we use physical model as data inputs
    - ensemble runs (currently need to train models in a loop, then manually collect paths, 
        then eval_run in a loop, then create results ensemble, then retrieve data, then plot)
    - model preformance metrics + visualizations (currently no good graphs (percentiles for ensemble run, 
        comparing preformance with physical model), need to write code to get metrics)
TODO:
    - add support for turning on and off physics based inputs
    - add more visualizations
    - make default args better
    - add percentiles to ensemble runs
'''
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import time
import webbrowser

from neuralhydrology.utils.config import Config
from neuralhydrology.training.train import start_training
from neuralhydrology.nh_run import eval_run
from neuralhydrology.utils.nh_results_ensemble import create_results_ensemble
# from neuralhydrology.utils.nh_results_ensemble_updated import create_results_ensemble
from neuralhydrology.evaluation.metrics import calculate_all_metrics


class UCB_trainer:
    def __init__(self, path_to_csv_folder: Path, yaml_path: Path, hyperparams: dict, 
                 input_features: list[str] = None, num_ensemble_members: int = 1, 
                 physics_informed: bool = False, physics_data_file: Path = None, gpu: int = -1):
        """
        Initializes the UCB_trainer object.

        Args:
            hyperparams (dict): A dictionary of hyperparameters for the model.
            num_ensemble_members (int): The number of ensemble members.
            physics_informed (bool): Whether to use internal states of physical model as features.
        """
        self._hyperparams = hyperparams
        self._num_ensemble_members = num_ensemble_members
        self._physics_informed = physics_informed
        self._physics_data_file = physics_data_file
        self._gpu = gpu
        self._data_dir = path_to_csv_folder
        self._dynamic_inputs = input_features
        self._yaml_path = yaml_path
        
        self._config = None
        self._model = None
        self._test_predictions = None
        self._test_observed = None
        self._metrics = None
        self._basin_name = None
        self._target_variable = None

        self._create_config()

    def train(self):
        """
        Public method to handle the training and evaluating process for individual models or ensembles. Sets self.model.
        """
        if self._num_ensemble_members == 1:
            self._model = self._train_model()  # returns run directory of single model
            self._eval_model(self._model)
        else:
            # returns dict with predictions on test set and metrics
            self.model = self._train_ensemble()
            self._model = self._train_ensemble() # returns dict with predictions on test set and metrics
        return

    def results(self) -> dict:
        """
        Public method to return metrics and plot data visualizations of model preformance.
        """
        self._get_predictions()
        self._metrics = calculate_all_metrics(
            self._test_observed, self._test_predictions)
        self._metrics = calculate_all_metrics(self._test_observed, self._test_predictions)

        self._generate_obs_sim_plt()
        self._generate_csv()
        return self._metrics

    def _generate_csv(self):
        """
        Private method to generate a CSV file of observed and predicted values. Used in the .results() function.
        :return:
        """
        if self._test_observed is None or self._test_predictions is None:
            print("[ERROR] Observed or predicted values are None. Cannot generate CSV.")
            return

        try:
            if self._num_ensemble_members == 1:
                dates = self._test_observed['date'].values
            else:
                dates = self._test_observed['datetime'].values

            df = pd.DataFrame({
                'Date': dates,
                'Observed': self._test_observed.values,
                'Predicted': self._test_predictions.values
            })

            output_path = Path(self._config.run_dir) / "results_output.csv"
            df.to_csv(output_path, index=False)
            print(f"[INFO] CSV output saved at: {output_path}")
        except Exception as e:
            print(f"[ERROR] Failed to generate CSV: {e}")

    def _train_model(self) -> Path:
        """
        Private method to train an individual model. Returns the path to the model results.
        """

        # check if a GPU has been specified. If yes, overwrite config
        if self._gpu is not None and self._gpu >= 0:
            self._config.device = f"cuda:{self._gpu}"
        if self._gpu is not None and self._gpu < 0:
            self._config.device = "cpu"

        start_training(self._config)
        path = self._config.run_dir
        return path

    def _eval_model(self, run_directory, period="test"):
        """
        Private method to evaluate an individual model after training.
        """
        eval_run(run_dir=run_directory, period=period)
        return

    def _train_ensemble(self, period="test") -> dict:
        """
        Private method to train and evaluate an ensemble of models.
        """
        paths = []  # store the path of the results of the model
        for _ in range(self._num_ensemble_members):
            path = self._train_model()
            paths.append(path)

        # for each path evaluate the model
        for p in paths:
            self._eval_model(run_directory=p, period=period)
            # self._eval_model(run_dir=p, period="validation")
        # #for each path evaluate the model
        # for p in paths:
        #     self._eval_model(run_directory=p, period=period)
        #     #self._eval_model(run_dir=p, period="validation")

        ensemble_run = create_results_ensemble(paths, period=period)
        return ensemble_run
    
    #Hardcoded for Tuler
    # def _get_predictions(self) -> dict:
    #     """
    #     Private method to get and return predicted values and metrics after training and evaluation.
    #     """
    #     if self._num_ensemble_members == 1:
    #         # Single model case
    #         with open(self._model / "test" / f"model_epoch{str(self._config.epochs).zfill(3)}" / "test_results.p", "rb") as fp:
    #             results = pickle.load(fp)
    #             self._test_observed = results['Tuler']['1D']['xr']['ReservoirInflowFLOW-OBSERVED_obs'].sel(
    #                 time_step=0)
    #             self._test_predictions = results['Tuler']['1D']['xr']['ReservoirInflowFLOW-OBSERVED_sim'].sel(
    #                 time_step=0)

    #     else:
    #         # Ensemble case
    #         self._test_observed = self._model['Tuler']['1D']['xr']['ReservoirInflowFLOW-OBSERVED_obs']
    #         self._test_predictions = self._model['Tuler']['1D']['xr']['ReservoirInflowFLOW-OBSERVED_sim']

    #     return
    def _get_predictions(self) -> dict:
        """
        Private method to get and return predicted values and metrics after training and evaluation.
        For the single ensemble case only.
        """
        if self._num_ensemble_members == 1:
            # Single model case
            with open(self._model / "test" / f"model_epoch{str(self._config.epochs).zfill(3)}" / "test_results.p", "rb") as fp:
                results = pickle.load(fp)

            # Dynamically get the basin name
            self._basin_name = next(iter(results.keys()))  # Get the first basin key dynamically
            print(f"Using basin: {self._basin_name}")

            # Retrieve the target variable from the config
            self._target_variable = self._config.target_variables[0]  # Assuming single target variable for now
            print(f"Using target variable: {self._target_variable}")

            # Construct keys for observed and simulated data
            observed_key = f"{self._target_variable}_obs"
            simulated_key = f"{self._target_variable}_sim"

            # Check if keys exist
            if observed_key not in results[self._basin_name]['1D']['xr']:
                raise KeyError(f"Observed key '{observed_key}' not found in results for basin {self._basin_name}.")
            if simulated_key not in results[self._basin_name]['1D']['xr']:
                raise KeyError(f"Simulated key '{simulated_key}' not found in results for basin {self._basin_name}.")

            # Extract observed and simulated data
            self._test_observed = results[self._basin_name]['1D']['xr'][observed_key].sel(time_step=0)
            self._test_predictions = results[self._basin_name]['1D']['xr'][simulated_key].sel(time_step=0)

        return

    def _create_config(self) -> Config:
        """
        Private method to create Configuration object for training from user specifications.
        """
        if not self._yaml_path.exists():
            raise FileNotFoundError(f"YAML configuration file not found: {self._yaml_path}")

        # Load the base configuration from the provided YAML file
        config = Config(self._yaml_path)

        # config = Config(Path('./template_config.yaml'))

        if 'save_weights_every' not in self._hyperparams:
            self._hyperparams['save_weights_every'] = self._hyperparams['epochs']
        
        if self._dynamic_inputs is not None: 
            config.update_config({'dynamic_inputs': self._dynamic_inputs})
        
        config.update_config(self._hyperparams)
        config.update_config({'data_dir': self._data_dir})
        config.update_config({'physics_informed': self._physics_informed})
        if self._physics_informed:
            if self._physics_data_file:
                config.update_config({'physics_data_file': self._physics_data_file})
            else:
                raise ValueError("Physics-informed is enabled, but no physics data file was provided.")
        #config.update_config({'log_tensorboard': True}) 

        self._config = config

        if self._config.epochs % self._config.save_weights_every != 0:
            raise ValueError(
                "The 'save_weights_every' parameter must divide the 'epochs' parameter evenly. Ensure 'epochs' is a multiple of "
                "'save_weights_every' to use the most recent weights for the final model."
            )

        return

    # def _generate_obs_sim_plt(self):
    #     """
    #     Private method to plot observed and simulated values over time.
    #     """
    #     date_indexer = "date" if self._num_ensemble_members == 1 else "datetime"
    #     fig, ax = plt.subplots(figsize=(16, 10))
    #     ax.plot(self._test_observed[date_indexer],
    #             self._test_observed, label="Obs")
    #     ax.plot(self._test_predictions[date_indexer],
    #             self._test_predictions, label="Sim")
    #     ax.set_ylabel("ReservoirInflowFLOW-OBSERVED")
    #     ax.legend()
    #     plt.show()

    def _generate_obs_sim_plt(self):
        """
        Private method to plot observed and simulated values over time with improved aesthetics and dynamic labels.
        """
        if self._test_observed is None or self._test_predictions is None:
            print("[ERROR] Observed or predicted values are None. Cannot generate plot.")
            return

        # Create the plot
        fig, ax = plt.subplots(figsize=(16, 10))
        if self._physics_informed:
            simulated_label = 'HybridSimulation'
        else: simulated_label = 'Simulated'
        ax.plot(self._test_observed["date"], self._test_observed, label="Observed", linewidth=1.5)
        ax.plot(self._test_predictions["date"], self._test_predictions, label=simulated_label, linewidth=1.5)

        # Set dynamic labels and title using stored target variable and basin name
        ax.set_ylabel(f"{self._target_variable} (units)", fontsize=14)
        ax.set_xlabel("Date", fontsize=14)
        ax.set_title(f"{self._basin_name} - {self._target_variable} Over Time", fontsize=16)

        # Add legend and grid
        ax.legend(fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.7)

        # Improve date formatting
        fig.autofmt_xdate()

        # Final adjustments and show
        plt.tight_layout()
        plt.show()

    def _plot_day_of_year_average(self):
        """
        Private method to plot day-of-year averages of observed and predicted values.
        """
        if self._test_observed is None or self._test_predictions is None:
            print("[ERROR] Observed or predicted values are None. Cannot generate plot.")
            return

        date_indexer = "date" if self._num_ensemble_members == 1 else "datetime"

        observed_series = pd.Series(self._test_observed.values, index=self._test_observed[date_indexer].values)
        predicted_series = pd.Series(self._test_predictions.values, index=self._test_predictions[date_indexer].values)
        observed_doy_avg = observed_series.groupby(observed_series.index.dayofyear).mean()
        predicted_doy_avg = predicted_series.groupby(predicted_series.index.dayofyear).mean()

        fig, ax = plt.subplots(figsize=(16, 10))
        ax.plot(observed_doy_avg.index, observed_doy_avg, label="Observed DOY Avg")
        ax.plot(predicted_doy_avg.index, predicted_doy_avg, label="Predicted DOY Avg")
        ax.set_xlabel("Day of Year")
        ax.set_ylabel("Average Reservoir Inflow")
        ax.legend()
        plt.title("Day-of-Year Average Plot of Observed vs. Predicted")
        plt.show()

    def _plot_month_of_year_average(self):
        """
        Private method to plot month-of-year averages of observed and predicted values.
        """
        if self._test_observed is None or self._test_predictions is None:
            print("[ERROR] Observed or predicted values are None. Cannot generate plot.")
            return

        date_indexer = "date" if self._num_ensemble_members == 1 else "datetime"

        observed_series = pd.Series(self._test_observed.values, index=self._test_observed[date_indexer].values)
        predicted_series = pd.Series(self._test_predictions.values, index=self._test_predictions[date_indexer].values)

        observed_moy_avg = observed_series.groupby(observed_series.index.month).mean()
        predicted_moy_avg = predicted_series.groupby(predicted_series.index.month).mean()

        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        fig, ax = plt.subplots(figsize=(16, 10))
        ax.plot(observed_moy_avg.index, observed_moy_avg, label="Observed MOY Avg")
        ax.plot(predicted_moy_avg.index, predicted_moy_avg, label="Predicted MOY Avg")
        ax.set_xlabel("Month")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_names)
        ax.set_ylabel("Average Reservoir Inflow")
        ax.legend()
        plt.title("Month-of-Year Average Plot of Observed vs. Predicted")
        plt.show()

    def open_tensorboard(self, logdir: str, port: int = 6006):
        """
        Open TensorBoard and display logs.
        
        Args:
            logdir (str): Path to the directory containing TensorBoard event files.
            port (int): Port to host TensorBoard on (default: 6006).
        """
        logdir_path = Path(logdir)
        
        #check that the log directory exists
        if not logdir_path.exists():
            raise FileNotFoundError(f"Log directory {logdir} does not exist.")

        #check if event files exist in the log directory
        event_files = list(logdir_path.rglob("events.out.tfevents*"))
        if not event_files:
            raise FileNotFoundError(f"No TensorBoard event files found in log directory {logdir}.")

        #TensorBoard command
        tb_command = f"tensorboard --logdir={logdir} --port={port} --host=0.0.0.0"
        
        try:
            #start TensorBoard as a subprocess
            process = subprocess.Popen(tb_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(5)  # Allow TensorBoard some time to start
            
            #open TensorBoard in the default web browser
            url = f"http://localhost:{port}"
            webbrowser.open(url)
            print(f"TensorBoard started at {url} with logs from {logdir}")
        
        except Exception as e:
            raise Exception(f"Failed to start TensorBoard: {e}")

        return process
