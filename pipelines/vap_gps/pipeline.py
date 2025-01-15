import xarray as xr
from typing import Dict
import matplotlib.pyplot as plt

from tsdat import TransformationPipeline


class VapGPS(TransformationPipeline):
    """---------------------------------------------------------------------------------
    VAP pipeline for time-averaging GPS data into 10 minute windows.
    ---------------------------------------------------------------------------------"""

    def hook_customize_input_datasets(self, input_datasets) -> Dict[str, xr.Dataset]:
        # Code hook to customize any input datasets prior to datastreams being combined
        # and data converters being run.
        return input_datasets

    def hook_customize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        # (Optional) Use this hook to modify the dataset before qc is applied
        return dataset

    def hook_finalize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        # (Optional) Use this hook to modify the dataset after qc is applied
        # but before it gets saved to the storage area
        return dataset

    def hook_plot_dataset(self, dataset: xr.Dataset):
        # (Optional, recommended) Create plots.
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
        # Set grid below
        ax.set_axisbelow(True)
        ax.grid()

        plt.xticks(rotation=45)
        plot_file = self.get_ancillary_filepath(title="location")
        fig.savefig(plot_file)

        # Timeseries plots of lat/lon
        fig, ax = plt.subplots()
        ax.plot(dataset["time"], dataset["lat"], "C0")
        ax.tick_params(axis="y", color="C0", labelcolor="C0")
        ax.set_ylabel("Latitude [degN]", color="C0")
        ax2 = ax.twinx()
        ax2.plot(dataset["time"], dataset["lon"], "C1")
        ax2.tick_params(axis="y", color="C1", labelcolor="C1")
        ax2.set_ylabel("Longitude [degE]", color="C1")
        ax2.spines["left"].set_color("C0")
        ax2.spines["right"].set_color("C1")

        plot_file = self.get_ancillary_filepath(title="timeseries")
        fig.savefig(plot_file)
        plt.close("all")
