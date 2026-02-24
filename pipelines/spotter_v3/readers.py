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


class FLTReader(DataReader):
    """Reads "FLT" filetype from spotter: motion data"""

    def read(self, input_key: str) -> Union[xr.Dataset, Dict[str, xr.Dataset]]:
        try:
            df = pd.read_csv(
                input_key,
                delimiter=",",
                index_col="time",
                names=["ms", "time", "x", "y", "z", ""],
                header=0,
                on_bad_lines="skip",
                na_values=[
                    "millis",
                    "GPS_Epoch_Time(s)",
                    "outx(mm)",
                    "outy(mm)",
                    "outz(mm)",
                ],
                low_memory=False,
            )
            # Drop rows where the index is NaN
            df = df.loc[df.index.notna()]
        except:
            return {}

        if dump_bad_files(df.index):
            return {}
        return df.to_xarray()


class GPSReader(DataReader):
    """Reads "LOC" filetype from spotter: GPS data"""

    def read(self, input_key: str) -> Union[xr.Dataset, Dict[str, xr.Dataset]]:
        try:
            df = pd.read_csv(input_key, delimiter=",", index_col=0)
            df["lat"] = np.array(df["lat(deg)"] + df["lat(min*1e5)"] * 1e-5 / 60)
            df["lon"] = np.array(df["long(deg)"] + df["long(min*1e5)"] * 1e-5 / 60)
            df.index.name = "time_gps"
        except:
            return {}

        if dump_bad_files(df.index):
            return {}
        return df.to_xarray()


class SpotterRawReader(DataReader):
    """Reads raw files from spotter that don't require special edits"""

    class Parameters(BaseModel, extra=Extra.forbid):
        time_var: str = "time"

    parameters: Parameters = Parameters()

    def read(self, input_key: str) -> Union[xr.Dataset, Dict[str, xr.Dataset]]:
        try:
            df = pd.read_csv(input_key, delimiter=",", index_col=0, engine="python")
            df.index.name = self.parameters.time_var
        except:
            return {}

        if dump_bad_files(df.index):
            return {}
        return df.to_xarray()
