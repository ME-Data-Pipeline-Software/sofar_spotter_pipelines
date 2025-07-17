import warnings
from typing import Dict, Union
from pydantic import BaseModel, Extra
import xarray as xr
import pandas as pd
import numpy as np
from tsdat import DataReader


def dump_bad_files(data):
    # Fail files that have no timestamps
    if not data.size:
        return True
    # Fail files that were created before the GPS has a lock
    # (timestamps end in "t")
    elif "t" in str(data[0]):
        return True
    else:
        return False


class GPSReader(DataReader):
    """Reads "LOC" filetype from spotter: GPS data"""

    def read(self, input_key: str) -> Union[xr.Dataset, Dict[str, xr.Dataset]]:

        df = pd.read_csv(input_key, delimiter=",", index_col=0)
        df["lat"] = np.array(df["lat(deg)"] + df["lat(min*1e5)"] * 1e-5 / 60)
        df["lon"] = np.array(df["long(deg)"] + df["long(min*1e5)"] * 1e-5 / 60)
        df.index.name = "time"

        if dump_bad_files(df.index):
            return xr.Dataset()
        return df.to_xarray()


class SSTReader(DataReader):
    """Reads "SST" filetype from spotter: sea surface temperature data"""

    def read(self, input_key: str) -> Union[xr.Dataset, Dict[str, xr.Dataset]]:

        df = pd.read_csv(input_key, delimiter=",", index_col=0)

        # 2nd version of SST file includes timestamps, 1st does not
        if df.index.name == "millis":
            warnings.warn("SST file missing timestamps. SST file will not be read.")
            return xr.Dataset()
        elif dump_bad_files(df.index):
            return xr.Dataset()
        else:
            df.index.name = "time"
            return df.to_xarray()


class SpotterRawReader(DataReader):
    """Reads raw files from spotter that don't require special edits"""

    def read(self, input_key: str) -> Union[xr.Dataset, Dict[str, xr.Dataset]]:
        df = pd.read_csv(input_key, delimiter=",", index_col=0, engine="python")
        df.index.name = "time"

        if dump_bad_files(df.index):
            return xr.Dataset()
        return df.to_xarray()
