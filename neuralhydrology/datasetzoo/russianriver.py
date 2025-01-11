from typing import List, Dict, Union
from pathlib import Path

import pandas as pd
import xarray
import logging

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config


class RussianRiver(BaseDataset):
    """To make this dataset available for model training, don't forget to add it to the `get_dataset()` function in 
    'neuralhydrology.datasetzoo.__init__.py'

    Parameters
    ----------
    cfg : Config
        The run configuration.
    is_train : bool 
        Defines if the dataset is used for training or evaluating. If True (training), means/stds for each feature
        are computed and stored to the run directory. If one-hot encoding is used, the mapping for the one-hot encoding 
        is created and also stored to disk. If False, a `scaler` input is expected and similarly the `id_to_int` input
        if one-hot encoding is used. 
    period : {'train', 'validation', 'test'}
        Defines the period for which the data will be loaded
    basin : str, optional
        If passed, the data for only this basin will be loaded. Otherwise the basin(s) are read from the appropriate
        basin file, corresponding to the `period`.
    additional_features : List[Dict[str, pd.DataFrame]], optional
        List of dictionaries, mapping from a basin id to a pandas DataFrame. This DataFrame will be added to the data
        loaded from the dataset, and all columns are available as 'dynamic_inputs', 'evolving_attributes' and
        'target_variables'
    id_to_int : Dict[str, int], optional
        If the config argument 'use_basin_id_encoding' is True in the config and period is either 'validation' or 
        'test', this input is required. It is a dictionary, mapping from basin id to an integer (the one-hot encoding).
    scaler : Dict[str, Union[pd.Series, xarray.DataArray]], optional
        If period is either 'validation' or 'test', this input is required. It contains the centering and scaling
        for each feature and is stored to the run directory during training (train_data/train_data_scaler.yml).

    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        # initialize parent class
        super(RussianRiver, self).__init__(cfg=cfg,
                                              is_train=is_train,
                                              period=period,
                                              basin=basin,
                                              additional_features=additional_features,
                                              id_to_int=id_to_int,
                                              scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load basin time series data
        
        This function is used to load the time series data (meteorological forcing, streamflow, etc.) and make available
        as time series input for model training later on. Make sure that the returned dataframe is time-indexed.
        
        Parameters
        ----------
        basin : str
            Basin identifier as string.

        Returns
        -------
        pd.DataFrame
            Time-indexed DataFrame, containing the time series data (e.g., forcings + discharge).
        """
        df = load_russian_river_data(self.cfg.data_dir, self.cfg.hourly) #daily or hourly df
        if self.cfg.physics_informed:
            physics_file = self.cfg.physics_data_file
            if Path(physics_file).exists():
                physics_df = load_hms_basin_data(physics_file, self.cfg.hourly)
                df = pd.merge(df, physics_df, left_index=True, right_index=True, how='outer') #add physics columns if physics informed
            else: raise FileNotFoundError(f"Physics data file not found: {physics_file}")

        return df
            

    def _load_attributes(self) -> pd.DataFrame:
        """Load dataset attributes
        
        This function is used to load basin attribute data (e.g. CAMELS catchments attributes) as a basin-indexed 
        dataframe with features in columns.
        
        Returns
        -------
        pd.DataFrame
            Basin-indexed DataFrame, containing the attributes as columns.
        """
        return load_russian_river_attributes(self.cfg.data_dir)
    
def load_hms_basin_data(physics_data_file: Path, hourly: bool) -> pd.DataFrame:
    logging.info(f"Loading data from file: {physics_data_file}, hourly: {hourly}")
    df = pd.read_csv(physics_data_file, low_memory=False)
    df.columns = df.iloc[0]
    df = df[3:]
    df.columns = df.columns.str.strip()
    df = df.drop(columns=['Ordinate'])
    df = df.rename(columns={'Date': 'Day', 'Time': 'Time'})
    #if hourly:
    #    df['Time'] = df['Time'] + ":00"
    df['Time'] = df['Time'].replace('24:00:00', '00:00:00')
    df['date'] = pd.to_datetime(df['Day'], format='%d-%b-%y') + pd.to_timedelta(df['Time'])
    df.set_index('date', inplace=True)
    return df

def load_russian_river_data(data_dir: Path, hourly: bool) -> pd.DataFrame:  
    if hourly:
        file_path = data_dir / 'hourly.csv'
    else:
        file_path = data_dir / 'daily.csv'
    df = pd.read_csv(file_path, low_memory=False)
    
    df.columns = df.iloc[0]
    df = df[3:]
    df.columns = df.columns.str.strip()
    df = df.drop(columns=['Ordinate'])
    df = df.rename(columns={'Date': 'Day', 'Time': 'Time'})
    #if hourly:
    #    df['Time'] = df['Time'] + ":00"
    df['Time'] = df['Time'].replace('24:00:00', '00:00:00')
    df['date'] = pd.to_datetime(df['Day'], format='%d-%b-%y') + pd.to_timedelta(df['Time'])
    df.set_index('date', inplace=True)
    return df

def load_russian_river_attributes(data_dir: Path) -> pd.DataFrame:
    # look into whether we need these or not
    return None

