import numpy as np
import xarray as xr
from matplotlib import pyplot as plt, dates as mpldt
import dolfyn
from tsdat import IngestPipeline, get_start_date_and_time_str, get_filename


class WaveStats(IngestPipeline):
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

    def hook_plot_dataset(self, dataset: xr.Dataset):
        # (Optional, recommended) Create plots.
        location = self.dataset_config.attrs.location_id
        datastream: str = self.dataset_config.attrs.datastream

        date, time = get_start_date_and_time_str(dataset)

        plt.style.use("default")  # clear any styles that were set before
        plt.style.use("shared/styling.mplstyle")

        with self.storage.uploadable_dir(datastream) as tmp_dir:
            t = dolfyn.time.dt642date(dataset.time)

            fig, ax = plt.subplots()
            ax.loglog(
                dataset["frequency"],
                dataset["wave_energy_density"].mean("time"),
                label="vertical",
            )
            m = -4
            x = np.logspace(-1, 0.5)
            y = 10 ** (-5) * x**m
            ax.loglog(x, y, "--", c="black", label="f^-4")
            ax.set(
                ylim=(0.00001, 10),
                xlabel="Frequency [Hz]",
                ylabel="Energy Density [m^2/Hz]",
            )
            plot_file = get_filename(
                dataset, title="elevation_spectrum", extension="png"
            )
            fig.savefig(tmp_dir / plot_file)
            plt.close(fig)

            fig, ax = plt.subplots(2, figsize=(15, 10))
            ax[0].scatter(t, dataset["wave_hs"])
            ax[0].set_xlabel("Time")
            ax[0].xaxis.set_major_formatter(mpldt.DateFormatter("%D"))
            ax[0].set_ylabel("Significant Wave Height [m]")

            ax[1].scatter(t, dataset["wave_te"])
            ax[1].set_xlabel("Time")
            ax[1].xaxis.set_major_formatter(mpldt.DateFormatter("%D"))
            ax[1].set_ylabel("Energy Period [s]")
            plot_file = get_filename(dataset, title="wave_stats", extension="png")
            fig.savefig(tmp_dir / plot_file)
            plt.close(fig)

            fig, ax = plt.subplots(1, figsize=(20, 10))
            ax.scatter(t, dataset["wave_dp"], label="Wave direction")
            ax.scatter(t, dataset["wave_spread"], label="Wave spread")
            ax.set_xlabel("Time")
            ax.xaxis.set_major_formatter(mpldt.DateFormatter("%D"))
            ax.set_ylabel("deg")
            ax.legend()
            plot_file = get_filename(dataset, title="wave_direction", extension="png")
            fig.savefig(tmp_dir / plot_file)
            plt.close(fig)
