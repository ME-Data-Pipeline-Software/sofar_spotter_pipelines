import numpy as np
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

        # Need to write in direction coordinate that will be used later
        for key in input_datasets:
            if "wave" in key:
                directions = np.arange(0, 360, 2.0)
                input_datasets[key] = input_datasets[key].assign_coords(
                    {"direction": directions}
                )
                return input_datasets

    def hook_customize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        # (Optional) Use this hook to modify the dataset before qc is applied

        # Calculate directional wave spectrum
        a0 = dataset["wave_energy_density"] / np.pi
        r1 = (1 / a0) * np.sqrt(
            dataset["wave_a1_value"] ** 2 + dataset["wave_b1_value"] ** 2
        )
        r2 = (1 / a0) * np.sqrt(
            dataset["wave_a2_value"] ** 2 + dataset["wave_b2_value"] ** 2
        )
        # dir1 (+/- pi) and dir2 (+/- pi/2) are CCW from East, "to" convention
        dir1 = np.arctan2(dataset["wave_b1_value"], dataset["wave_a1_value"])
        dir2 = 0.5 * np.arctan2(dataset["wave_b2_value"], dataset["wave_a2_value"])

        # Spreading function
        # Subtract dataset variable to get dimensions right
        D = (1 / np.pi) * (
            0.5
            + r1 * np.cos(-1 * (dir1 - np.deg2rad(dataset["direction"])))
            + r2 * np.cos(-2 * (dir2 - np.deg2rad(dataset["direction"])))
        )

        # Wave energy density is units of Hz and degrees
        dataset["wave_dir_energy_density"].values = dataset[
            "wave_energy_density"
        ] * np.rad2deg(D)

        # Reset direction coordinate so that the spreading function D corresponds
        # to CW from North, "from" convention, instead of CCW from East, "to" convention.
        dir_from_N = (270 - dataset["direction"]) % 360
        dirN = xr.DataArray(
            dir_from_N,
            coords={"direction": dir_from_N},
            attrs=dataset["direction"].attrs,
        )
        dataset = dataset.assign_coords({"direction": dirN})
        # sort direction properly so that it runs 0 - 360
        dataset = dataset.sortby(dataset["direction"])

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
        fig, ax = plt.subplots(nrows=4)
        # Plot Wave Heights
        c2 = amp_r(0.50)
        dataset["wave_hs"].plot(ax=ax[0], c=c2, label=r"H$_{sig}$")
        ax[0].legend(bbox_to_anchor=(1, -0.10), ncol=3)
        ax[0].set(ylabel="Wave Height [m]", xlabel="", xticklabels=[], ylim=(0, 2))

        # Plot Wave Periods
        c1, c2 = dense(0.3), dense(0.6)
        dataset["wave_ta"].plot(ax=ax[1], c=c1, label=r"T$_{mean}$")
        dataset["wave_tp"].plot(ax=ax[1], c=c2, label=r"T$_{peak}$")
        ax[1].legend(bbox_to_anchor=(1, -0.10), ncol=3)
        ax[1].set(ylabel="Wave Period [s]", xlabel="", xticklabels=[], ylim=(0, 22))

        # Plot Wave Direction
        c1 = haline(0.5)
        dataset["wave_dp"].plot(ax=ax[2], c=c1, label=r"D$_{peak}$")
        ax[2].legend(bbox_to_anchor=(1, -0.10), ncol=2)
        ax[2].set(
            ylabel="Wave Direction [deg]", xlabel="", xticklabels=[], ylim=(-180, 180)
        )
        c1 = haline(0.9)
        dataset["sst"].plot(ax=ax[3], c=c1, label=r"SST$")
        ax[3].legend(bbox_to_anchor=(1, -0.10), ncol=2)
        ax[3].set(ylabel="Temperature [deg C]", xlabel="Time (UTC)", ylim=(5, 20))

        plot_file = self.get_ancillary_filepath(title="wave_stats")
        fig.savefig(plot_file)
        plt.close(fig)

        # Plot directional spectra
        fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        spectrum = dataset["wave_dir_energy_density"].mean("time")
        a, f = np.meshgrid(np.deg2rad(dataset["direction"]), dataset["frequency"])
        color_level_max = np.ceil(np.max(spectrum.data) * 10) / 10
        levels = np.linspace(0, color_level_max, 11)
        c = ax.contourf(a, f, spectrum, levels=levels)
        cbar = plt.colorbar(c)
        cbar.set_label("Spectrum [m^2 s/deg]", rotation=270, labelpad=20)
        ylabels = ax.get_yticklabels()
        ylabels = [ilabel.get_text() for ilabel in ax.get_yticklabels()]
        ylabels = [ilabel + "Hz" for ilabel in ylabels]
        ticks_loc = ax.get_yticks()
        ax.set_yticks(ticks_loc)
        ax.set_yticklabels(ylabels)

        plot_file = self.get_ancillary_filepath(title="directional_spectra")
        fig.savefig(plot_file)
        plt.close(fig)
