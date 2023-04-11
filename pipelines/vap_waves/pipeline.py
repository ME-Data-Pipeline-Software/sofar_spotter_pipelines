import xarray as xr
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from tsdat import TransformationPipeline, get_start_date_and_time_str, get_filename


class VapWaves(TransformationPipeline):
    """---------------------------------------------------------------------------------
    This is an example pipeline meant to demonstrate how one might set up a
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
        ds = dataset
        location = self.dataset_config.attrs.location_id
        datastream: str = self.dataset_config.attrs.datastream

        date, time = get_start_date_and_time_str(dataset)

        # Set the format of the x-axis tick labels
        time_format = mdates.DateFormatter("%D")
        plt.style.use("default")  # clear any styles that were set before
        plt.style.use("shared/styling.mplstyle")

        with self.storage.uploadable_dir(datastream) as tmp_dir:
            fig, ax = plt.subplots()
            ax.plot(ds.time, ds["x"], label="surge")
            ax.plot(ds.time, ds["y"], label="sway")
            ax.plot(ds.time, ds["z"], label="heave")
            ax.set_title("")  # Remove bogus title created by xarray
            ax.legend(ncol=2, bbox_to_anchor=(1, -0.05))
            ax.set_ylabel("Buoy Displacement [m]")
            ax.set_xlabel("Time [UTC]")
            ax.xaxis.set_major_formatter(time_format)
            plot_file = get_filename(dataset, title="displacement", extension="png")
            fig.savefig(tmp_dir / plot_file)
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.scatter(ds["lon"], ds["lat"])
            ax.set_title("")  # Remove bogus title created by xarray
            ax.set_ylabel("Latitude [deg N]")
            ax.set_xlabel("Longitude [deg E]")
            plot_file = get_filename(dataset, title="location", extension="png")
            fig.savefig(tmp_dir / plot_file)
            plt.close(fig)

        # TODO: Better x-axis ticks:
        # Set the x-axis to have ticks spaced by the hour
        # hours = mdates.HourLocator(interval=1)
        # ax.xaxis.set_major_locator(hours)
