from typing import Dict, Union
from pydantic import BaseModel, Extra
import xarray as xr
import numpy as np

import dolfyn
from dolfyn.adv import api
from tsdat import DataReader


class ADVReader(DataReader):
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

        magnetic_declination: float = 0
        correlation_threshold: float = 30
        bin_averaging_window: int = 300  # ensemble size in seconds

    parameters: Parameters = Parameters()
    """Extra parameters that can be set via the retrieval configuration file. If you opt
    to not use any configuration parameters then please remove the code above."""

    def read(self, input_key: str) -> Union[xr.Dataset, Dict[str, xr.Dataset]]:
        """-------------------------------------------------------------------
        Classes derived from the FileHandler class can implement this method.
        to read a custom file format into a xr.Dataset object.

        Args:
            filename (str): The path to the ADCP file to read in.

        Returns:
            xr.Dataset: An xr.Dataset object
        -------------------------------------------------------------------"""

        ds = dolfyn.read(input_key)

        # Select specific time interval
        t0 = np.datetime64("2021-07-01T13:30:00.0")
        ds = ds.sel(time=slice(t0, ds.time[-1]))

        # Clean out low correlation data
        ds.velds.rotate2("beam")
        ds["vel"] = ds["vel"].where(
            ds["corr"].values > 100 - self.parameters.correlation_threshold
        )
        ds.velds.rotate2("inst")

        # Set declination and rotate to earth coordinates
        dolfyn.set_declination(ds, self.parameters.magnetic_declination)
        dolfyn.rotate2(ds, "earth")

        # Calculate principal heading and rotate to principal coordinates
        ds.attrs["principal_heading"] = dolfyn.calc_principal_heading(ds.vel)
        dolfyn.rotate2(ds, "principal")

        # Velocity magnitude and direction in degrees from N
        ds["U_mag"] = ds.velds.U_mag
        ds["U_dir"] = ds.velds.U_dir
        ds.U_dir.values = dolfyn.tools.misc.convert_degrees(ds.U_dir.values)

        # Bin velocity data and calculate basic turbulence statistics
        bnr = dolfyn.adv.api.ADVBinner(
            n_bin=self.parameters.bin_averaging_window * ds.fs, fs=ds.fs
        )
        bdat = bnr.do_avg(ds)
        bdat["U_mag"] = bdat.velds.U_mag
        bdat["U_dir"] = bdat.velds.U_dir
        bdat["tke_vec"] = bnr.calc_tke(ds.vel)
        bdat["stress_vec"] = bnr.calc_stress(ds.vel)
        bdat["TKE"] = bdat.velds.tke
        bdat["TI"] = bdat.velds.I

        # Dropping the detailed configuration stats because netcdf can't save it
        for key in list(bdat.attrs.keys()):
            if "config" in key:
                bdat.attrs.pop(key)

        return bdat
