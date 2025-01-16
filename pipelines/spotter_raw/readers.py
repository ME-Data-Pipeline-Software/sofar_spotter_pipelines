import warnings
from typing import Dict, Union
import xarray as xr
import pandas as pd
import numpy as np
from tsdat import DataReader


class GPSReader(DataReader):
    """Reads "LOC" filetype from spotter: GPS data"""

    def read(self, input_key: str) -> Union[xr.Dataset, Dict[str, xr.Dataset]]:

        df = pd.read_csv(input_key, delimiter=",", index_col=0)
        df["lat"] = np.array(df["lat(deg)"] + df["lat(min*1e5)"] * 1e-5 / 60)
        df["lon"] = np.array(df["long(deg)"] + df["long(min*1e5)"] * 1e-5 / 60)
        df.index.name = "time"

        return df.to_xarray()


class SSTReader(DataReader):
    """Reads "SST" filetype from spotter: sea surface temperature data"""

    def read(self, input_key: str) -> Union[xr.Dataset, Dict[str, xr.Dataset]]:

        df = pd.read_csv(input_key, delimiter=",", index_col=0)

        # 2nd version of SST file includes timestamps, 1st does not
        if df.index.name == "millis":
            warnings.warn("SST file missing timestamps. SST file will not be read.")
            return xr.Dataset()
        else:
            df.index.name = "time"
            return df.to_xarray()


class SpotterRawReader(DataReader):
    """Reads raw files from spotter that don't require special edits"""

    def read(self, input_key: str) -> Union[xr.Dataset, Dict[str, xr.Dataset]]:
        df = pd.read_csv(input_key, delimiter=",", index_col=0, engine="python")
        df.index.name = "time"
        return df.to_xarray()
