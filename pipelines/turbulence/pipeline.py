import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from tsdat import IngestPipeline, get_start_date_and_time_str, get_filename
import dolfyn
from utils import format_time_xticks


class Turbulence(IngestPipeline):
    """---------------------------------------------------------------------------------
    This is an example ingestion pipeline meant to demonstrate how one might set up a
    pipeline using this template repository.

    ---------------------------------------------------------------------------------"""

    def hook_customize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        # (Optional) Use this hook to modify the dataset before qc is applied
        return dataset

    def hook_finalize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        # (Optional) Use this hook to modify the dataset after qc is applied
        # but before it gets saved to the storage area
        return dataset

    def hook_plot_dataset(self, ds: xr.Dataset):
        location = self.dataset_config.attrs.location_id
        datastream: str = self.dataset_config.attrs.datastream

        date, time = get_start_date_and_time_str(ds)

        plt.style.use("default")  # clear any styles that were set before
        plt.style.use("shared/styling.mplstyle")

        with self.storage.uploadable_dir(datastream) as tmp_dir:

            fig, ax = plt.subplots(5, figsize=(15, 10))
            ax[0].plot(ds.time, ds.velocity[0])
            ax[0].set(xlabel="Time", ylabel="Velocity {} [m/s]".format(ds.dir[0].data))

            ax[1].plot(ds.time, ds.velocity[1])
            ax[1].set(xlabel="Time", ylabel="Velocity {} [m/s]".format(ds.dir[1].data))

            ax[2].plot(ds.time, ds.velocity[2])
            ax[2].set(xlabel="Time", ylabel="Velocity {} [m/s]".format(ds.dir[2].data))

            ax[3].plot(ds.time, ds.turbulent_kinetic_energy)
            ax[3].set(xlabel="Time", ylabel="TKE [m^2/s^2]", yscale="log")

            ax[4].plot(ds.time, ds.turbulence_intensity)
            ax[4].set(xlabel="Time", ylabel="Turbulence Intensity [%]")
            ax[4].set(ylim=[0, 1])

            plot_file = get_filename(ds, title="velocity", extension="png")
            fig.savefig(tmp_dir / plot_file)
            plt.close(fig)
