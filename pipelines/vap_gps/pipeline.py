import numpy as np
import xarray as xr
from typing import Dict
import matplotlib.pyplot as plt

from tsdat import TransformationPipeline


class VapGPS(TransformationPipeline):
    """---------------------------------------------------------------------------------
    This is an example pipeline meant to demonstrate how one might set up a
    pipeline using this template repository.

    ---------------------------------------------------------------------------------"""

    def hook_customize_input_datasets(self, input_datasets) -> Dict[str, xr.Dataset]:
        # Code hook to customize any input datasets prior to datastreams being combined
        # and data converters being run.
        return input_datasets

    def hook_customize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        # (Optional) Use this hook to modify the dataset before qc is applied
        
        # Because we set bin average alignment to "LEFT", we need to add 5
        # minutes to change this to "CENTER". We did this because the waves
        # data is being process with a 5 minute offset as well.
        time = dataset['time'].values
        time += np.timedelta64(300, 's')
        time = xr.DataArray(
            time, coords={"time": time}, attrs=dataset["time"].attrs
        )
        dataset = dataset.assign_coords({"time": time})
        
        return dataset

    def hook_finalize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        # (Optional) Use this hook to modify the dataset after qc is applied
        # but before it gets saved to the storage area
        return dataset

    def hook_plot_dataset(self, dataset: xr.Dataset):
        # (Optional, recommended) Create plots.

        # Set the format of the x-axis tick labels
        plt.style.use("default")  # clear any styles that were set before
        plt.style.use("shared/styling.mplstyle")

        ## Plot GPS
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(dataset["lon"], dataset["lat"])
        ax.set(ylabel="Latitude [deg N]", xlabel="Longitude [deg E]")
        ax.ticklabel_format(axis="both", style="plain", useOffset=False)
        ax.set(
            xlim=(dataset.geospatial_lon_min, dataset.geospatial_lon_max),
            ylim=(dataset.geospatial_lat_min, dataset.geospatial_lat_max),
        )

        plot_file = self.get_ancillary_filepath(title="location")
        fig.savefig(plot_file)
        plt.close(fig)
