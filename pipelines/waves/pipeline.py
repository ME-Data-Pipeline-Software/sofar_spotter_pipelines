import xarray as xr
import matplotlib.pyplot as plt

from tsdat import IngestPipeline, get_start_date_and_time_str, get_filename


class Waves(IngestPipeline):
    """--------------------------------------------------------------------------------
    SPOTTER_BUOY INGESTION PIPELINE

    Wave data taken in Clallam Bay over a month-long deployment in Aug-Sep 2021

    --------------------------------------------------------------------------------"""

    def hook_customize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        # (Optional) Use this hook to modify the dataset before qc is applied
        return dataset

    def hook_finalize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        # (Optional) Use this hook to modify the dataset after qc is applied
        # but before it gets saved to the storage area
        return dataset

    def hook_plot_dataset(self, dataset: xr.Dataset):
        ds = dataset
        location = self.dataset_config.attrs.location_id
        datastream: str = self.dataset_config.attrs.datastream

        date, time = get_start_date_and_time_str(dataset)

        plt.style.use("default")  # clear any styles that were set before
        plt.style.use("shared/styling.mplstyle")

        with self.storage.uploadable_dir(datastream) as tmp_dir:

            if 'waves' in ds.qualifier:
                fig, ax = plt.subplots()

                ax.plot(ds.time, ds["x"], label="surge")
                ax.plot(ds.time, ds["y"], label="sway")
                ax.plot(ds.time, ds["z"], label="heave")

                ax.set_title("")  # Remove bogus title created by xarray
                ax.legend(ncol=2, bbox_to_anchor=(1, -0.05))
                ax.set_ylabel("Buoy Displacement [m]")
                ax.set_xlabel("Time [UTC]")

                plot_file = get_filename(
                    dataset, title="buoy_displacement", extension="png"
                )
                fig.savefig(tmp_dir / plot_file)
                plt.close(fig)

            elif 'gps' in ds.qualifier:
                fig, ax = plt.subplots()

                ax.scatter(ds['lon'], ds["lat"])

                ax.set_title("")  # Remove bogus title created by xarray
                ax.set_ylabel("Latitude [deg N]")
                ax.set_xlabel("Longitude [deg E]")

                plot_file = get_filename(
                    dataset, title="location", extension="png"
                )
                fig.savefig(tmp_dir / plot_file)
                plt.close(fig)
