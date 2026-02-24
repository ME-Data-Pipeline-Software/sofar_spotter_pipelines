import os
from typing import Dict, Union
import xarray as xr
import pandas as pd
import numpy as np
from tsdat import DataReader


class GPSReader(DataReader):
    """Reads "LOC" filetype from spotter: GPS data"""

    def read(self, input_key: str) -> Union[xr.Dataset, Dict[str, xr.Dataset]]:

        df = pd.read_csv(input_key, delimiter=",", index_col=False)
        time = "GPS_Epoch_Time(s)"
        ds = xr.Dataset(
            data_vars={
                "lat": (
                    [time],
                    np.array(df["lat(deg)"] + df["lat(min*1e5)"] * 1e-5 / 60),
                ),
                "lon": (
                    [time],
                    np.array(df["long(deg)"] + df["long(min*1e5)"] * 1e-5 / 60),
                ),
            },
            coords={time: (time, df[time])},
        )
        return ds


class SSTReader(DataReader):
    """Reads "SST" filetype from spotter: sea surface temperature data"""

    def read(self, input_key: str) -> Union[xr.Dataset, Dict[str, xr.Dataset]]:

        df = pd.read_csv(input_key, delimiter=",", index_col=False)

        # Create timestamp
        time = "GPS_Epoch_Time(s)"
        # 2nd version of SST file includes timestamps, 1st does not
        if "millis" in df:
            if not os.path.isfile(input_key.replace("SST", "FLT")):
                raise FileNotFoundError(
                    "Cannot read SST file without associated FLT file"
                )
            df_flt = pd.read_csv(
                input_key.replace("SST", "FLT"), delimiter=",", index_col=False
            )

            # Get zero milliseconds from FLT file
            t_zero = df_flt[time][2] - df_flt["millis"][2] / 1000

            # Counter resets at 4294967296 (2**32, must be a 32 bit number)
            ms = df["millis"].values.copy()
            flag = np.diff(ms) < 0
            if sum(flag):
                idx = np.where(flag)[0][0]
                ms[idx + 1 :] += 4294967296

            time_data = t_zero + ms / 1000
            sst = "sst"
        else:
            time_data = df["timestamp (ticks/UTC)"].values
            sst = "temperature (C)"

        ds = xr.Dataset(
            data_vars={"sst": ([time], df[sst].values)},
            coords={time: (time, time_data)},
        )

        return ds
