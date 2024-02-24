import numpy as np
from pydantic import BaseModel, Extra
import xarray as xr
from numpy.typing import NDArray
from tsdat import QualityChecker, QualityHandler
from dolfyn.adv.clean import GN2002, clean_fill


class GoringNikora2002(QualityChecker):
    """----------------------------------------------------------------------------
    The Goring & Nikora 2002 'despiking' method, with Wahl2003 correction.
    Returns a logical vector that is true where spikes are identified.

    Args:
        variable_name (str): array (1D or 3D) to clean.
        n_points (int) : The number of points over which to perform the method.

    Returns:
        mask [np.ndarray]: Logical vector with spikes labeled as 'True'

    ----------------------------------------------------------------------------"""

    class Parameters(BaseModel, extra=Extra.forbid):
        n_points: int = 5000

    parameters: Parameters = Parameters()
    """Extra parameters that can be set via the quality configuration file. If you opt
    to not use any configuration parameters then please remove the code above."""

    def run(self, dataset: xr.Dataset, variable_name: str) -> NDArray[np.bool8]:

        return GN2002(dataset[variable_name], npt=self.parameters.n_points)


class CubicSplineInterp(QualityHandler):
    """----------------------------------------------------------------------------
    Interpolate over mask values in timeseries data using the specified method

    Parameters
    ----------
    variable_name : xarray.DataArray
        The dataArray to clean.
    mask : bool
        Logical tensor of elements to "nan" out and replace
    npt : int
        The number of points on either side of the bad values that
    interpolation occurs over
    method : string
        Interpolation scheme to use (linear, cubic, pchip, etc)
    max_gap : int
        Max number of consective nan's to interpolate across, must be <= npt/2

    Returns
    -------
    da : xarray.DataArray
        The dataArray with nan's filled in
    ----------------------------------------------------------------------------"""

    class Parameters(BaseModel, extra=Extra.forbid):
        npt: int = 12
        method: str = "cubic"

    parameters: Parameters = Parameters()
    """Extra parameters that can be set via the quality configuration file. If you opt
    to not use any configuration parameters then please remove the code above."""

    def run(
        self, dataset: xr.Dataset, variable_name: str, failures: NDArray[np.bool8]
    ) -> xr.Dataset:

        if failures.any():
            dataset[variable_name] = clean_fill(
                dataset[variable_name],
                mask=failures,
                npt=self.parameters.npt,
                method=self.parameters.method,
            )
        return dataset
