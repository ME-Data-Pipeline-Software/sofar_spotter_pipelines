from typing import Dict, Union
from pydantic import BaseModel, Extra
import xarray as xr
import pandas as pd
import numpy as np
from tsdat import DataReader


class GPSReader(DataReader):
    """---------------------------------------------------------------------------------
    Custom DataReader that can be used to read data from a specific format.

    Built-in implementations of data readers can be found in the
    [tsdat.io.readers](https://tsdat.readthedocs.io/en/latest/autoapi/tsdat/io/readers)
    module.

    ---------------------------------------------------------------------------------"""

    class Parameters(BaseModel, extra=Extra.forbid):
        """If your CustomDataReader should take any additional arguments from the
        retriever configuration file, then those should be specified here.

        e.g.,:
        custom_parameter: float = 5.0

        """

    parameters: Parameters = Parameters()
    """Extra parameters that can be set via the retrieval configuration file. If you opt
    to not use any configuration parameters then please remove the code above."""

    def read(self, input_key: str) -> Union[xr.Dataset, Dict[str, xr.Dataset]]:
        # Reads "LOC" filetype from spotter: GPS data
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
