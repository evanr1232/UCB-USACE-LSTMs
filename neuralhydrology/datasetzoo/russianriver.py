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
        cfg_dict = self.cfg.as_dict()
        is_mts_data_flag = cfg_dict.get("is_mts_data", False)

        print(f"[DEBUG:_load_basin_data] => basin={basin}, is_mts_data={is_mts_data_flag}, physics_informed={self.cfg.physics_informed}")

        if is_mts_data_flag:
            return self._load_mts_data(basin)
        else:
            return self._load_single_freq(basin)

    def _load_mts_data(self, basin: str) -> pd.DataFrame:
        # print("[DEBUG:_load_mts_data] => reading daily_mts_shift.csv and hourly_mts.csv")

        daily_path = self.cfg.data_dir / "daily_mts_shift.csv"
        hourly_path = self.cfg.data_dir / "hourly_mts.csv"

        # print(f"[DEBUG:_load_mts_data] => daily file: {daily_path}")
        daily_df = pd.read_csv(daily_path, low_memory=False)
        # print("[DEBUG:_load_mts_data] => daily_df shape BEFORE clean_df:", daily_df.shape)

        daily_df = clean_df(daily_df)
        # print("[DEBUG:_load_mts_data] => daily_df shape AFTER clean_df:", daily_df.shape)
        daily_df = daily_df.resample("1H").ffill()
        # print("[DEBUG:_load_mts_data] => daily_df shape AFTER resample('1H'):", daily_df.shape)
        # print("[DEBUG:_load_mts_data] => daily_df index[:5]:", daily_df.index[:5])

        # print(f"[DEBUG:_load_mts_data] => hourly file: {hourly_path}")
        hourly_df = pd.read_csv(hourly_path, low_memory=False)
        # print("[DEBUG:_load_mts_data] => hourly_df shape BEFORE clean_df:", hourly_df.shape)

        hourly_df = clean_df(hourly_df)
        # print("[DEBUG:_load_mts_data] => hourly_df shape AFTER clean_df:", hourly_df.shape)
        # print("[DEBUG:_load_mts_data] => hourly_df index[:5]:", hourly_df.index[:5])

        df = pd.merge(hourly_df, daily_df, how="outer", left_index=True, right_index=True)
        # print("[DEBUG:_load_mts_data] => shape after merging hourly + daily =>", df.shape)
        # print("[DEBUG:_load_mts_data] => df index[:5]:", df.index[:5])

        if self.cfg.physics_informed:
            # Additional merges for daily/hourly physics
            pass  # omitted for brevity but add similar debug prints

        # print("[DEBUG:_load_mts_data] => final df shape:", df.shape)
        return df

    def _load_single_freq(self, basin: str) -> pd.DataFrame:
        """Load single-frequency data (daily or hourly)."""
        if self.cfg.hourly:
            path = self.cfg.data_dir / "hourly.csv"
            freq_str = "HOURLY"
        else:
            path = self.cfg.data_dir / "daily_shift.csv"
            freq_str = "DAILY"

        # print(f"[DEBUG:_load_single_freq] => loading {freq_str} from {path}")
        raw_df = pd.read_csv(path, low_memory=False)
        # print("[DEBUG:_load_single_freq] => raw_df shape:", raw_df.shape)

        df = clean_df(raw_df)
        # print("[DEBUG:_load_single_freq] => shape AFTER clean_df:", df.shape)
        # print("[DEBUG:_load_single_freq] => first 5 index entries:", df.index[:5])

        # if len(df.index) > 1:
        #     print("[DEBUG:_load_single_freq] => index[1] - index[0] =", df.index[1] - df.index[0])

        if self.cfg.physics_informed and self.cfg.physics_data_file:
            physics_path = self.cfg.physics_data_file
            # print(f"[DEBUG:_load_single_freq] => reading PHYSICS from {physics_path}")
            phys_df = pd.read_csv(physics_path, low_memory=False)
            # print("[DEBUG:_load_single_freq] => phys_df shape BEFORE clean_df:", phys_df.shape)

            phys_df = clean_df(phys_df)
            # print("[DEBUG:_load_single_freq] => phys_df shape AFTER clean_df:", phys_df.shape)
            df = pd.merge(df, phys_df, how='outer', left_index=True, right_index=True)
            # print("[DEBUG:_load_single_freq] => shape after merging with physics =>", df.shape)
        else:
            if self.cfg.physics_informed:
                print(f"[WARNING:_load_single_freq] => No physics_data_file found, skipping merges.")

        return df

    def _load_attributes(self) -> pd.DataFrame:
        # If you have static basin attributes, load them here
        return load_russian_river_attributes(self.cfg.data_dir)

def load_russian_river_attributes(data_dir: Path) -> pd.DataFrame:
    # if no static attributes are needed, just return None or empty DataFrame
    return pd.DataFrame()