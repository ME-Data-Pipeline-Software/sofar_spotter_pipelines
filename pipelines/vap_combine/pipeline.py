import xarray as xr
from typing import Dict
import matplotlib.pyplot as plt
from cmocean.cm import amp_r, dense, haline

from tsdat import TransformationPipeline


class VapWaveStats(TransformationPipeline):
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

        # Wave stats
        fig, axs = plt.subplots(nrows=3)

        # Plot Wave Heights
        c2 = amp_r(0.50)
        dataset["wave_hs"].plot(ax=axs[0], c=c2, label=r"H$_{sig}$")
        axs[0].legend(bbox_to_anchor=(1, -0.10), ncol=3)
        axs[0].set_ylabel("Wave Height (m)")

        # Plot Wave Periods
        c1, c2 = dense(0.3), dense(0.6)
        dataset["wave_ta"].plot(ax=axs[1], c=c1, label=r"T$_{mean}$")
        dataset["wave_tp"].plot(ax=axs[1], c=c2, label=r"T$_{peak}$")
        axs[1].legend(bbox_to_anchor=(1, -0.10), ncol=3)
        axs[1].set_ylabel("Wave Period (s)")

        # Plot Wave Directions
        c1 = haline(0.5)
        dataset["wave_dp"].plot(ax=axs[2], c=c1, label=r"D$_{peak}$")
        axs[2].legend(bbox_to_anchor=(1, -0.10), ncol=2)
        axs[2].set_ylabel("Wave Direction (deg)")

        # c1 = haline(0.9)
        # ds["sst"].plot(ax=axs[3], c=c1, label=r"Sea Surface$")
        # axs[3].legend(bbox_to_anchor=(1, -0.10), ncol=2)
        # axs[3].set_ylabel("Temperature (deg C)")

        for i in range(len(axs)):
            axs[i].set_xlabel("Time (UTC)")

        plot_file = self.get_ancillary_filepath(title="wave_stats")
        fig.savefig(plot_file)
        plt.close(fig)