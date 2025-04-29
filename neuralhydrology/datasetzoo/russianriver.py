from typing import List, Dict, Union
from pathlib import Path
import pandas as pd
import xarray
import logging
import matplotlib.pyplot as plt

from UCB_training.UCB_utils import clean_df
from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config

class RussianRiver(BaseDataset):
    """Multi-basin, multi-timescale dataset loader for the Russian River region.

    If cfg.is_mts=True, merges daily.csv + hourly.csv + daily physics + hourly physics
    for the given basin, upsampling daily data to hourly. Otherwise, single-frequency logic.
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        super(RussianRiver, self).__init__(cfg=cfg,
                                           is_train=is_train,
                                           period=period,
                                           basin=basin,
                                           additional_features=additional_features,
                                           id_to_int=id_to_int,
                                           scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load daily and hourly data (and physics if needed) for the given basin, with debug prints."""
        cfg_dict = self.cfg.as_dict()
        is_mts_data_flag = cfg_dict.get("is_mts_data", False)

        print(f"[DEBUG] => in _load_basin_data, cfg['is_mts_data'] = {is_mts_data_flag}")
        if is_mts_data_flag:
            return self._load_mts_data(basin)
        else:
            return self._load_single_freq(basin)

    def _load_mts_data(self, basin: str) -> pd.DataFrame:
        """Helper to load multi-timescale daily_mts.csv + hourly_mts.csv, plus physics if needed."""
        # daily_path = self.cfg.data_dir / "daily_mts.csv"
        daily_path = self.cfg.data_dir / "daily_mts_shift.csv"
        print(f"[DEBUG] => MTS: reading daily from {daily_path}")
        daily_df = pd.read_csv(daily_path, low_memory=False)
        print("[DEBUG] => daily_mts.csv shape =>", daily_df.shape)
        daily_df = clean_df(daily_df)
        print("[DEBUG] => daily_df shape AFTER clean =>", daily_df.shape)
        daily_df = daily_df.resample("1H").ffill()
        print("[DEBUG] => daily_df AFTER resample =>", daily_df.shape)

        hourly_path = self.cfg.data_dir / "hourly_mts.csv"
        print(f"[DEBUG] => MTS: reading hourly from {hourly_path}")
        hourly_df = pd.read_csv(hourly_path, low_memory=False)
        print("[DEBUG] => hourly_mts.csv shape =>", hourly_df.shape)
        hourly_df = clean_df(hourly_df)
        print("[DEBUG] => hourly_df shape AFTER clean =>", hourly_df.shape)

        df = pd.merge(hourly_df, daily_df, how="outer", left_index=True, right_index=True)
        print("[DEBUG] => MTS => after merging daily/hourly => df.shape =>", df.shape)

        if self.cfg.physics_informed:
            daily_phys_path = self.cfg.data_dir / f"{basin}_daily_mts_shift.csv" # --> ALWAYS LIKE THIS
            hourly_phys_path = self.cfg.data_dir / f"{basin}_hourly_mts.csv"

            if daily_phys_path.exists():
                print(f"[DEBUG] => MTS: reading daily physics from {daily_phys_path}")
                daily_phys_df = pd.read_csv(daily_phys_path, low_memory=False)
                print("[DEBUG] => daily_phys_df shape =>", daily_phys_df.shape)
                daily_phys_df = clean_df(daily_phys_df)
                print("[DEBUG] => daily_phys_df shape AFTER clean =>", daily_phys_df.shape)
                daily_phys_df = daily_phys_df.resample("1H").ffill()
                print("[DEBUG] => daily_phys_df AFTER resample =>", daily_phys_df.shape)

                df = pd.merge(df, daily_phys_df, how="outer", left_index=True, right_index=True)
                print("[DEBUG] => MTS => after merging daily_phys => df.shape =>", df.shape)

            if hourly_phys_path.exists():
                print(f"[DEBUG] => MTS: reading hourly physics from {hourly_phys_path}")
                hourly_phys_df = pd.read_csv(hourly_phys_path, low_memory=False)
                print("[DEBUG] => hourly_phys_df shape =>", hourly_phys_df.shape)
                hourly_phys_df = clean_df(hourly_phys_df)
                print("[DEBUG] => hourly_phys_df shape AFTER clean =>", hourly_phys_df.shape)

                df = pd.merge(df, hourly_phys_df, how="outer", left_index=True, right_index=True)
                print("[DEBUG] => MTS => after merging hourly_phys => df.shape =>", df.shape)

        print("[DEBUG] => final MTS df.shape =>", df.shape)
        return df

    def _load_single_freq(self, basin: str) -> pd.DataFrame:
        """Load single-frequency data (daily or hourly)."""
        if self.cfg.hourly:
            path = self.cfg.data_dir / "hourly.csv"
            freq_str = "HOURLY"
        else:
            # path = self.cfg.data_dir / "daily.csv"
            path = self.cfg.data_dir / "daily_shift.csv"
            freq_str = "DAILY"

        print(f"[DEBUG] => single-freq: reading {freq_str} from {path}")
        raw_df = pd.read_csv(path, low_memory=False)
        print(f"[DEBUG] => shape after read_csv => {raw_df.shape}")

        df = clean_df(raw_df)
        print(f"[DEBUG] => shape after clean_df => {df.shape}")

        if self.cfg.physics_informed and self.cfg.physics_data_file:
            physics_path = self.cfg.physics_data_file
            print(f"[DEBUG] => single-freq: reading PHYSICS from {physics_path}")
            phys_df = pd.read_csv(physics_path, low_memory=False)
            print(f"[DEBUG] => physics shape => {phys_df.shape}")
            phys_df = clean_df(phys_df)
            print(f"[DEBUG] => after clean => physics shape => {phys_df.shape}")

            df = pd.merge(df, phys_df, how='outer', left_index=True, right_index=True)
            print("[DEBUG] => after merging single-freq physics => df.shape =>", df.shape)
        else:
            if self.cfg.physics_informed:
                print(f"[WARNING] => Provided physics_data_file does not exist or was not set: {self.cfg.physics_data_file}")

        return df

    def _load_attributes(self) -> pd.DataFrame:
        # If you have static basin attributes, load them here
        return load_russian_river_attributes(self.cfg.data_dir)

def load_russian_river_attributes(data_dir: Path) -> pd.DataFrame:
    # if no static attributes are needed, just return None or empty DataFrame
    return pd.DataFrame()