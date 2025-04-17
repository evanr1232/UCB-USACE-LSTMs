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
        if "is_mts_data" in cfg_dict and cfg_dict["is_mts_data"]:
            mts_flag = True
        else:
            mts_flag = False

        print(f"[DEBUG] => in _load_basin_data, cfg['is_mts_data'] = {mts_flag}")
        if mts_flag:
            daily_path = self.cfg.data_dir / "daily_mts.csv"
            print(f"[DEBUG] => MTS: reading daily from {daily_path}")
            daily_df = pd.read_csv(daily_path, low_memory=False)
            print("[DEBUG] => daily_mts columns after read_csv =>", daily_df.columns.tolist())
            daily_df = clean_df(daily_df)
            print("[DEBUG] => columns AFTER clean_df_mts =>", daily_df.columns.tolist())
            print("[DEBUG] => index AFTER clean_df_mts =>", daily_df.index)
            daily_df = daily_df.resample("1H").ffill()
            print("[DEBUG] => daily_df after resample('1H') => columns:", daily_df.columns.tolist())

            hourly_path = self.cfg.data_dir / "hourly_mts.csv"
            print(f"[DEBUG] => MTS: reading hourly from {hourly_path}")
            hourly_df = pd.read_csv(hourly_path, low_memory=False)
            print("[DEBUG] => hourly_mts columns after read_csv =>", hourly_df.columns.tolist())
            hourly_df = clean_df(hourly_df)
            print("[DEBUG] => hourly_df after clean_df => columns:", hourly_df.columns.tolist())

            df = pd.merge(hourly_df, daily_df, how="outer", left_index=True, right_index=True)
            print("[DEBUG] => after merging daily/hourly => df.columns =>", df.columns.tolist())

            # If physics_informed => also merge basin_daily_mts.csv + basin_hourly_mts.csv
            if self.cfg.physics_informed:
                daily_phys_path = self.cfg.data_dir / f"{basin}_daily_mts.csv"
                hourly_phys_path = self.cfg.data_dir / f"{basin}_hourly_mts.csv"

                if daily_phys_path.exists():
                    print(f"[DEBUG] => MTS: reading daily physics from {daily_phys_path}")
                    daily_phys_df = pd.read_csv(daily_phys_path, low_memory=False)
                    daily_phys_df = clean_df(daily_phys_df)
                    daily_phys_df = daily_phys_df.resample("1H").ffill()
                    df = pd.merge(df, daily_phys_df, how="outer", left_index=True, right_index=True)
                    print("[DEBUG] => after merging daily_phys => df.columns =>", df.columns.tolist())

                if hourly_phys_path.exists():
                    print(f"[DEBUG] => MTS: reading hourly physics from {hourly_phys_path}")
                    hourly_phys_df = pd.read_csv(hourly_phys_path, low_memory=False)
                    hourly_phys_df = clean_df(hourly_phys_df)
                    df = pd.merge(df, hourly_phys_df, how="outer", left_index=True, right_index=True)
                    print("[DEBUG] => after merging hourly_phys => df.columns =>", df.columns.tolist())

            print("[DEBUG] => final MTS df.columns =>", df.columns.tolist())
            return df

        else:
            return self._load_single_freq(basin)

    def _load_single_freq(self, basin: str) -> pd.DataFrame:
        """Load single-frequency data (daily or hourly)."""
        if self.cfg.hourly:
            path = self.cfg.data_dir / "hourly.csv"
            print(f"[DEBUG] => single-freq: reading HOURLY from {path}")
        else:
            path = self.cfg.data_dir / "daily.csv"
            print(f"[DEBUG] => single-freq: reading DAILY from {path}")

        df = pd.read_csv(path, low_memory=False)
        print("[DEBUG] => single-freq columns after read_csv =>", df.columns.tolist())
        df = clean_df(df)

        if self.cfg.physics_informed and self.cfg.physics_data_file:
            physics_path = Path(self.cfg.physics_data_file)
            print(f"[DEBUG] => single-freq: reading PHYSICS from {physics_path}")
            phys_df = pd.read_csv(physics_path, low_memory=False)
            print("[DEBUG] => physics columns after read_csv =>", phys_df.columns.tolist())
            phys_df = clean_df(phys_df)
            df = pd.merge(df, phys_df, how='outer', left_index=True, right_index=True)
            print("[DEBUG] => after merging single-freq physics => df.columns =>", df.columns.tolist())
        else:
            print(f"[WARNING] => Provided physics_data_file does not exist")
        return df
    def _load_attributes(self) -> pd.DataFrame:
        # If you have static basin attributes, load them here
        return load_russian_river_attributes(self.cfg.data_dir)

def load_russian_river_attributes(data_dir: Path) -> pd.DataFrame:
    # if no static attributes are needed, just return None or empty DataFrame
    return pd.DataFrame()