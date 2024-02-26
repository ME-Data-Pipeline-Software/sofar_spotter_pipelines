import numpy as np
import xarray as xr
from numpy.typing import NDArray
from tsdat import QualityChecker, QualityHandler


class WaveCheckFactor(QualityChecker):
    """----------------------------------------------------------------------------
    Checks for where the wave factor is nan or negative and returns a
    mask where bad values are labeled as True.
    (This function runs on a variable by variable basis.)
    ----------------------------------------------------------------------------"""

    def run(self, dataset: xr.Dataset, variable_name: str) -> NDArray[np.bool8]:

        if 'frequency' in dataset[variable_name].dims:
            check_factor = dataset['wave_check_factor']
        else:
            check_factor = dataset['wave_check_factor'].median('frequency')

        mask = check_factor.isnull() + (check_factor < 0.5) 

        return mask
