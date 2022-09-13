from typing import Dict, Union
from pydantic import BaseModel, Extra
import xarray as xr

import dolfyn
from dolfyn.adp import api
from tsdat import DataReader


class UpFacingADCPReader(DataReader):
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

        depth_offset: float = 0.5
        salinity: float = 35
        magnetic_declination: float = 0
        correlation_threshold: float = 30

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

        # The ADCP transducers were measured to be 0.6 m from the feet of the lander
        api.clean.set_range_offset(ds, self.parameters.depth_offset)

        # Locate surface using pressure data and remove data above it
        api.clean.find_surface_from_P(ds, salinity=self.parameters.salinity)
        ds = api.clean.nan_beyond_surface(ds)

        # Clean out low correlation data
        ds = api.clean.correlation_filter(
            ds, thresh=self.parameters.correlation_threshold
        )

        # Set declination and rotate to earth coordinates if not already
        dolfyn.set_declination(ds, self.parameters.magnetic_declination)
        dolfyn.rotate2(ds, "earth")

        # Velocity magnitude and direction in degrees from N
        ds["U_mag"] = ds.velds.U_mag
        ds["U_dir"] = ds.velds.U_dir
        ds.U_dir.values = dolfyn.tools.misc.convert_degrees(ds.U_dir.values)

        # Dropping the detailed configuration stats because netcdf can't save it
        for key in list(ds.attrs.keys()):
            if "config" in key:
                ds.attrs.pop(key)

        return ds


class DownFacingADCPReader(DataReader):
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

        depth_offset: float = 0.5
        amplitude_threshold: float = 40
        magnetic_declination: float = 0
        correlation_threshold: float = 30

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

        api.clean.set_range_offset(ds, self.parameters.depth_offset)

        # Locate surface using pressure data and remove data above it
        api.clean.find_surface(ds, thresh=self.parameters.amplitude_threshold)
        ds = api.clean.nan_beyond_surface(ds)

        # Rotate to Earth coordinates
        dolfyn.set_declination(ds, self.parameters.magnetic_declination)
        dolfyn.rotate2(ds, "earth")

        ds = api.clean.correlation_filter(
            ds, thresh=self.parameters.correlation_threshold
        )

        # Velocity magnitude and direction in degrees from N
        ds["U_mag"] = ds.velds.U_mag
        ds["U_dir"] = ds.velds.U_dir
        ds.U_dir.values = dolfyn.tools.misc.convert_degrees(ds.U_dir.values)

        # Dropping the detailed configuration stats because netcdf can't save it
        for key in list(ds.attrs.keys()):
            if "config" in key:
                ds.attrs.pop(key)

        return ds
