import numpy as np
import xarray as xr
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from cmocean.cm import amp_r, dense, haline
from mhkit.tidal import graphics
from tsdat import TransformationPipeline


class VapWaveStats(TransformationPipeline):
    """---------------------------------------------------------------------------------
    VAP pipeline for combining Sofar Spotter data products.
    ---------------------------------------------------------------------------------"""

    def hook_customize_input_datasets(self, input_datasets) -> Dict[str, xr.Dataset]:
        # Code hook to customize any input datasets prior to datastreams being combined
        # and data converters being run.

        # Need to write in direction coordinate that will be used later
        for key in input_datasets:
            if "wave" in key:
                directions = np.arange(0, 360, 2.0).astype("float32")
                input_datasets[key] = input_datasets[key].assign_coords(
                    {"direction": directions}
                )
                return input_datasets

    def hook_customize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        # (Optional) Use this hook to modify the dataset before qc is applied
        dataset.attrs.pop("description")

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
        plt.style.use("default")  # clear any styles that were set before

        # Wave stats
        if "air_pressure" in dataset:
            n = 5
        else:
            n = 4

        fig, ax = plt.subplots(n, 1, figsize=(11, 7), constrained_layout=True)
        ax[0].plot(
            dataset["time"],
            dataset["wave_hs"],
            ".-",
            label="Significant Wave Height",
            color=amp_r(0.10),
        )
        ax[0].set(ylabel="Height [m]")

        ax[1].plot(
            dataset["time"],
            dataset["wave_ta"],
            ".-",
            label="Mean Period",
            color=dense(0.15),
        )
        ax[1].plot(
            dataset["time"],
            dataset["wave_tp"],
            ".-",
            label="Peak Period",
            color=dense(0.35),
        )
        ax[1].plot(
            dataset["time"],
            dataset["wave_te"],
            ".-",
            label="Energy Period",
            color=dense(0.65),
        )
        ax[1].plot(
            dataset["time"],
            dataset["wave_tz"],
            ".-",
            label="Zero Crossing Period",
            color=dense(0.95),
        )
        ax[1].set(ylabel="Period [s]")

        ax[2].plot(
            dataset["time"],
            dataset["wave_dp"],
            ".-",
            label="Peak Direction",
            color=haline(0.10),
        )
        ax[2].plot(
            dataset["time"],
            dataset["wave_spread"],
            ".-",
            label="Peak Spread",
            color=haline(0.50),
        )
        ax[2].set(ylabel="Direction [deg]")

        ax[3].plot(
            dataset["time"],
            dataset["sst"],
            ".-",
            label="Sea Surface Temperature",
            color="black",
        )
        ax[3].set(ylabel="Temperature\n[deg C]")

        if "air_pressure" in dataset:
            ax[4].plot(
                dataset["time"],
                dataset["air_pressure"],
                ".-",
                label="Air Pressure",
                color="black",
            )
            ax[4].set(ylabel="Pressure [hPa]")

        for a in ax:
            a.legend(loc="upper left", bbox_to_anchor=[1.01, 1.0], handlelength=1.5)
        for a in ax[:-1]:
            a.set(xticklabels=[])
        ax[-1].tick_params(labelrotation=45)
        ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%D %H"))
        ax[-1].set(xlabel="Time (UTC)")

        plot_file = self.get_ancillary_filepath(title="wave_stats")
        fig.savefig(plot_file)

        ## Plot GPS
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.subplots_adjust(left=0.16, right=0.95, top=0.95, bottom=0.17)
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
        ax.tick_params(labelrotation=45)

        plot_file = self.get_ancillary_filepath(title="location")
        fig.savefig(plot_file)

        # Plot wave roses
        fig, ax = plt.subplots(
            figsize=(8, 6), subplot_kw={"projection": "polar"}, constrained_layout=True
        )
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        # Use 360 degrees
        dp = dataset["wave_dp"].copy(deep=True).values
        dp = dp % 360
        # Calculate the 2D histogram
        H, dir_edges, vel_edges = graphics._histogram(dp, dataset["wave_hs"], 10, 0.5)
        # Determine number of bins
        dir_bins = H.shape[0]
        h_bins = H.shape[1]
        # Create the angles
        thetas = np.arange(0, 2 * np.pi, 2 * np.pi / dir_bins)
        # Set bar color based on wind speed
        colors = plt.cm.Wistia(np.linspace(0, 1.0, h_bins))
        # Set the current speed bin label names
        # Calculate the 2D histogram
        labels = [f"{i:.1f}-{j:.1f}" for i, j in zip(vel_edges[:-1], vel_edges[1:])]
        # Initialize the vertical-offset (polar radius) for the stacked bar chart.
        r_offset = np.zeros(dir_bins)
        for h_bin in range(h_bins):
            # Plot fist set of bars in all directions
            ax.bar(
                thetas,
                H[:, h_bin],
                width=(2 * np.pi / dir_bins),
                bottom=r_offset,
                color=colors[h_bin],
                label=labels[h_bin],
            )
            # Increase the radius offset in all directions
            r_offset = r_offset + H[:, h_bin]
        # Add the a legend for current speed bins
        plt.legend(loc="best", title="Hs [m]", bbox_to_anchor=(1.29, 1.00), ncol=1)
        # Get the r-ticks (polar y-ticks)
        yticks = plt.yticks()
        # Format y-ticks with  units for clarity
        rticks = [f"{y:.1f}%" for y in yticks[0]]
        # Set the y-ticks
        ax.set_yticks(yticks[0], rticks)

        plot_file = self.get_ancillary_filepath(title="wave_rose")
        fig.savefig(plot_file)

        # Plot directional spectra
        fig, ax = plt.subplots(
            figsize=(8, 6), subplot_kw=dict(projection="polar"), constrained_layout=True
        )
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        # Use frequencies up to 0.5 Hz
        spectrum = dataset["wave_dir_energy_density"].mean("time")
        # Create grid and plot
        a, f = np.meshgrid(np.deg2rad(spectrum["direction"]), 1 / spectrum["frequency"])
        color_level_max = np.ceil(np.max(spectrum.values) * 10) / 10
        levels = np.linspace(0, color_level_max, 11)
        c = ax.contourf(a, f, spectrum, levels=levels, cmap="Blues")

        cbar = plt.colorbar(c)
        cbar.set_label("ESD [m$^2$ s/deg]", rotation=270, labelpad=20)
        ax.set_ylim(1, 12)
        ylabels = ax.get_yticklabels()
        ylabels = [ilabel.get_text() for ilabel in ax.get_yticklabels()]
        ylabels = [ilabel + " s" for ilabel in ylabels]
        ticks_loc = ax.get_yticks()
        ax.set_yticks(ticks_loc)
        ax.set_yticklabels(ylabels)

        plot_file = self.get_ancillary_filepath(title="directional_spectra")
        fig.savefig(plot_file)
        plt.close("all")
