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
    - add logging
    - add support for turning on and off physics based inputs
    - add more visualizations
    - make default args better
    - add percentiles to ensemble runs
'''

from pathlib import Path
import pickle
import matplotlib.pyplot as plt

from neuralhydrology.utils.config import Config
from neuralhydrology.training.train import start_training
from neuralhydrology.nh_run import eval_run
from neuralhydrology.utils.nh_results_ensemble import create_results_ensemble
from neuralhydrology.evaluation.metrics import calculate_all_metrics

class UCB_trainer:
    def __init__(self, path_to_csv_folder: Path, hyperparams: dict, num_ensemble_members: int = 1, physics_informed: bool = True, gpu: int = -1):
        """
        Initializes the UCB_trainer object.

        Args:
            hyperparams (dict): A dictionary of hyperparameters for the model.
            num_ensemble_members (int): The number of ensemble members.
            physics_informed (bool): Whether to use internal states of physical model as features.
        """
        self.hyperparams = hyperparams
        self.num_ensemble_members = num_ensemble_members
        self.physics_informed = physics_informed
        self.gpu = gpu
        self.data_dir = path_to_csv_folder
        
        self.config = None
        self.model = None
        self.test_predictions = None
        self.test_observed = None
        self.metrics = None

        self._create_config()

    def train(self):
        """
        Public method to handle the training and evaluating process for individual models or ensembles. Sets self.model.
        """
        if self.num_ensemble_members == 1:
            self.model = self._train_model() # returns run directory of single model
            self._eval_model(self.model)
        else:
            self.model = self._train_ensemble() # returns dict with predictions on test set and metrics
        return

    def results(self) -> dict:
        """
        Public method to return metrics and plot data visualizations of model preformance.
        """
        self._get_predictions()
        if self.num_ensemble_members == 1:
            self.metrics = calculate_all_metrics(self.test_observed.sel(time_step=0), self.test_predictions.sel(time_step=0))
        else: 
            self.metrics = calculate_all_metrics(self.test_observed, self.test_predictions)
        
        self._generate_obs_sim_plt()
        return self.metrics

    def _train_model(self) -> Path:
        """
        Private method to train an individual model. Returns the path to the model results.
        """
    
        # check if a GPU has been specified. If yes, overwrite config
        if self.gpu is not None and self.gpu >= 0:
            self.config.device = f"cuda:{self.gpu}"
        if self.gpu is not None and self.gpu < 0:
            self.config.device = "cpu"

        start_training(self.config)
        path = self.config.run_dir
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
        paths = [] #store the path of the results of the model
        for _ in range(self.num_ensemble_members):
            path = self._train_model()
            paths.append(path)

        #for each path evaluate the model    
        for p in paths:
            self._eval_model(run_directory=p, period=period)
            #self._eval_model(run_dir=p, period="validation") 

        ensemble_run = create_results_ensemble(paths, period=period)
        return ensemble_run

    def _get_predictions(self) -> dict:
        """
        Private method to get and return predicted values and metrics after training and evaluation.
        """
        if self.num_ensemble_members == 1:
            with open(self.model / "test" / f"model_epoch{str(self.config.epochs).zfill(3)}" / "test_results.p", "rb") as fp:
                results = pickle.load(fp)
                self.test_observed = results['Tuler']['1D']['xr']['ReservoirInflowFLOW-OBSERVED_obs']
                self.test_predictions = results['Tuler']['1D']['xr']['ReservoirInflowFLOW-OBSERVED_sim']

        else:
            self.test_observed = self.model['Tuler']['1D']['xr']['ReservoirInflowFLOW-OBSERVED_obs']
            self.test_predictions = self.model['Tuler']['1D']['xr']['ReservoirInflowFLOW-OBSERVED_sim']
        
        return

    def _create_config(self) -> Config:
        """
        Private method to create Configuration object for training from user specifications.
        """
        config = Config(Path('./template_config.yaml'))
        config.update_config(self.hyperparams)
        config.update_config({'data_dir': self.data_dir})
        self.config = config
        return

    def _generate_obs_sim_plt(self):
        """
        Private method to plot observed and simulated values over time.
        """
        date_indexer = "date" if self.num_ensemble_members == 1 else "datetime"
        fig, ax = plt.subplots(figsize=(16,10))
        ax.plot(self.test_observed[date_indexer], self.test_observed, label="Obs")
        ax.plot(self.test_predictions[date_indexer], self.test_predictions, label="Sim")
        ax.set_ylabel("ReservoirInflowFLOW-OBSERVED")
        ax.legend()
        plt.show()