from typing import Any, Optional
import xarray as xr
from tsdat import DataConverter, DatasetConfig, RetrievedDataset
from mhkit.dolfyn.time import epoch2dt64


class EpochTimeConverter(DataConverter):
    """---------------------------------------------------------------------------------
    Converts Spotter GPS time to datetime64
    ---------------------------------------------------------------------------------"""

    def convert(
        self,
        data: xr.DataArray,
        variable_name: str,
        dataset_config: DatasetConfig,
        retrieved_dataset: RetrievedDataset,
        **kwargs: Any,
    ) -> Optional[xr.DataArray]:
        """----------------------------------------------------------------------------
        Simple function to convert epoch time recorded by the Spotter buoy to
        numpy.datetime64 format.

        Args:
            data (xr.DataArray): The DataArray corresponding with the retrieved data
                variable to convert.
            variable_name (str): The name of the variable to convert.
            dataset_config (DatasetConfig): The output dataset configuration.
            retrieved_dataset (RetrievedDataset): The retrieved dataset structure.

        Returns:
            Optional[xr.DataArray]: The converted data as an xr.DataArray, or None if
                the conversion was done in place.
        ----------------------------------------------------------------------------"""

        data = epoch2dt64(data)
        data = data.assign_coords({variable_name: data})

        return data
