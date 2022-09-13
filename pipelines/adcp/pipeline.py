import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from tsdat import IngestPipeline, get_filename


class Current(IngestPipeline):
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
        def add_colorbar(ax, plot, label):
            cb = plt.colorbar(plot, ax=ax, pad=0.01)
            cb.ax.set_ylabel(label, fontsize=12)
            cb.outline.set_linewidth(1)
            cb.ax.tick_params(size=0)
            cb.ax.minorticks_off()
            return cb

        datastream: str = self.dataset_config.attrs.datastream
        date = pd.to_datetime(ds.time.values)

        plt.style.use("default")  # clear any styles that were set before
        plt.style.use("shared/styling.mplstyle")

        with self.storage.uploadable_dir(datastream) as tmp_dir:

            fig, ax = plt.subplots(
                nrows=2, ncols=1, figsize=(14, 8), constrained_layout=True
            )

            magn = ax[0].pcolormesh(
                date, ds.range, ds.flow_speed, cmap="Blues", shading="nearest"
            )
            ax[0].plot(date, ds.depth)
            ax[0].set_xlabel("Time (UTC)")
            ax[0].set_ylabel(r"Range [m]")
            ax[0].set_ylim([0, 11])
            add_colorbar(ax[0], magn, r"Speed [m/s]")

            dirc = ax[1].pcolormesh(
                date, ds.range, ds.flow_direction, cmap="twilight", shading="nearest"
            )
            ax[1].plot(date, ds.depth)
            ax[1].set_xlabel("Time (UTC)")
            ax[1].set_ylabel(r"Range [m]")
            ax[1].set_ylim([0, 11])
            add_colorbar(ax[1], dirc, r"Direction [deg from N]")

            plot_file = get_filename(ds, title="current", extension="png")
            fig.savefig(tmp_dir / plot_file)
            plt.close(fig)

        with self.storage.uploadable_dir(datastream) as tmp_dir:

            fig, ax = plt.subplots(
                nrows=ds.n_beams, ncols=1, figsize=(14, 8), constrained_layout=True
            )

            for beam in range(ds.n_beams):
                amp = ax[beam].pcolormesh(
                    date, ds.range, ds.amplitude[beam], shading="nearest"
                )
                ax[beam].set_title("Beam " + str(beam + 1))
                ax[beam].set_xlabel("Time (UTC)")
                ax[beam].set_ylabel(r"Range [m]")
                ax[beam].set_ylim([0, 11])
                add_colorbar(ax[beam], amp, "Amplitude [dB]")

            plot_file = get_filename(ds, title="amplitude", extension="png")
            fig.savefig(tmp_dir / plot_file)
            plt.close(fig)

        with self.storage.uploadable_dir(datastream) as tmp_dir:

            fig, ax = plt.subplots(
                nrows=ds.n_beams, ncols=1, figsize=(14, 8), constrained_layout=True
            )

            for beam in range(ds.n_beams):
                amp = ax[beam].pcolormesh(
                    date,
                    ds.range,
                    ds.correlation[beam],
                    cmap="copper",
                    shading="nearest",
                )
                ax[beam].set_title("Beam " + str(beam + 1))
                ax[beam].set_xlabel("Time (UTC)")
                ax[beam].set_ylabel(r"Range [m]")
                ax[beam].set_ylim([0, 11])
                add_colorbar(ax[beam], amp, "Correlation [%]")

            plot_file = get_filename(ds, title="correlation", extension="png")
            fig.savefig(tmp_dir / plot_file)
            plt.close(fig)
