import numpy as np
import xarray as xr
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from cmocean.cm import amp_r, dense, haline
from mhkit.tidal import graphics
from mhkit import wave, dolfyn
from tsdat import TransformationPipeline


fs = 2.5  # Hz, Spotter sampling frequency
wat = 1800  # s, window averaging time
freq_slc = [0.0455, 1]  # 22 to 1 s periods


class VapWaveStats(TransformationPipeline):
    """---------------------------------------------------------------------------------
    VAP pipeline for combining Sofar Spotter data products.
    ---------------------------------------------------------------------------------"""

    def hook_customize_input_datasets(self, input_datasets) -> Dict[str, xr.Dataset]:
        # Code hook to customize any input datasets prior to datastreams being combined
        # and data converters being run.

        # Need to write in frequency and direction coordinates that will be used later
        # Create FFT frequency vector
        nfft = fs * wat // 6
        f = np.fft.fftfreq(int(nfft), 1 / fs)
        # Use only positive frequencies
        freq = np.abs(f[1 : int(nfft / 2.0 + 1)])
        # Trim frequency vector to > 0.0455 Hz (wave periods between 1 and 22 s)
        freq = freq[np.where((freq > freq_slc[0]) & (freq <= freq_slc[1]))]
        directions = np.arange(0, 360, 10).astype("float32")

        for key in input_datasets:
            input_datasets[key] = input_datasets[key].assign_coords(
                {"frequency": freq.astype("float32")}
            )
            input_datasets[key] = input_datasets[key].assign_coords(
                {"direction": directions}
            )
        return input_datasets

    def hook_customize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        # (Optional) Use this hook to modify the dataset before qc is applied
        dataset.attrs.pop("description")
        # Update buoy name in qualifier and datastream
        spotter_id = dataset.attrs["inputs"].split("-")[-1].split(".")[0]
        dataset.attrs["qualifier"] = spotter_id
        dataset.attrs["datastream"] = dataset.attrs["datastream"].replace(
            "XXXXX", spotter_id
        )

        # Create 2D tensor for spectral analysis
        disp = xr.DataArray(
            data=np.array(
                [
                    dataset["x"],
                    dataset["y"],
                    dataset["z"],
                ]
            ),
            coords={"dir": ["x", "y", "z"], "time": dataset["time"]},
        )

        ## Using dolfyn to create spectra
        nbin = fs * wat
        fft_tool = dolfyn.adv.api.ADVBinner(
            n_bin=nbin, fs=fs, n_fft=nbin // 6, n_fft_coh=nbin // 6
        )
        # Trim frequency vector to > 0.0455 Hz (wave periods smaller than 22 s)
        slc_freq = slice(freq_slc[0], freq_slc[1])

        # Auto-spectra
        psd = fft_tool.power_spectral_density(disp, freq_units="Hz")
        psd = psd.sel(freq=slc_freq)
        Sxx = psd.sel(S="Sxx")
        Syy = psd.sel(S="Syy")
        Szz = psd.sel(S="Szz")

        # Cross-spectra
        csd = fft_tool.cross_spectral_density(disp, freq_units="Hz")
        csd = csd.sel(coh_freq=slc_freq)
        Cxz = csd.sel(C="Cxz").real
        Cxy = csd.sel(C="Cxy").real
        Cyz = csd.sel(C="Cyz").real

        ## Wave height and period
        pd_Szz = Szz.T.to_pandas()
        Hs = wave.resource.significant_wave_height(pd_Szz)
        Te = wave.resource.energy_period(pd_Szz)
        Ta = wave.resource.average_wave_period(pd_Szz)
        Tp = wave.resource.peak_period(pd_Szz)
        Tz = wave.resource.average_zero_crossing_period(pd_Szz)

        # Check factor: generally should be around 1
        k = np.sqrt((Sxx + Syy) / Szz)

        # Calculate peak wave direction and spread
        a1 = Cxz.values / np.sqrt((Sxx + Syy) * Szz)
        b1 = Cyz.values / np.sqrt((Sxx + Syy) * Szz)
        a2 = (Sxx - Syy) / (Sxx + Syy)
        b2 = 2 * Cxy.values / (Sxx + Syy)
        theta = np.rad2deg(np.arctan2(b1, a1))  # degrees CCW from East, "to" convention
        phi = np.rad2deg(np.sqrt(2 * (1 - np.sqrt(a1**2 + b1**2))))

        # Get peak frequency - fill nan slices with 0
        peak_idx = psd[2].fillna(0).argmax("freq")
        # degrees CW from North ("from" convention)
        direction = (270 - theta[:, peak_idx]) % 360
        # Set direction from -180 to 180
        direction[direction > 180] -= 360
        spread = phi[:, peak_idx]

        # Create averaged dataset
        ds = dataset.copy()
        mean_time = fft_tool.mean(ds["time"].values)
        # Trim time length to averaged time
        ds = ds.isel(time=slice(None, mean_time.size))
        # Set time coordinates
        mean_time = xr.DataArray(
            mean_time, coords={"time": mean_time}, attrs=dataset["time"].attrs
        )
        ds = ds.assign_coords({"time": mean_time})
        # Slice timestamps for auxillary data
        ds = ds.sel(time_aux=slice(dataset["time"][0], dataset["time"][-1]))

        # Make sure mhkit vars are set to float32
        ds["wave_energy_density"].values = Szz
        ds["wave_hs"].values = Hs.to_xarray()["Hm0"].astype("float32")
        ds["wave_te"].values = Te.to_xarray()["Te"].astype("float32")
        ds["wave_tp"].values = Tp.to_xarray()["Tp"].astype("float32")
        ds["wave_ta"].values = Ta.to_xarray()["Tm"].astype("float32")
        ds["wave_tz"].values = Tz.to_xarray()["Tz"].astype("float32")
        ds["wave_check_factor"].values = k
        ds["wave_a1_value"].values = a1
        ds["wave_b1_value"].values = b1
        ds["wave_a2_value"].values = a2
        ds["wave_b2_value"].values = b2
        ds["wave_dp"].values = direction
        ds["wave_spread"].values = spread

        ds = ds.drop(("x", "y", "z"))

        # Calculate directional wave spectrum
        a0 = ds["wave_energy_density"] / np.pi
        r1 = (1 / a0) * np.sqrt(ds["wave_a1_value"] ** 2 + ds["wave_b1_value"] ** 2)
        r2 = (1 / a0) * np.sqrt(ds["wave_a2_value"] ** 2 + ds["wave_b2_value"] ** 2)
        # dir1 (+/- pi) and dir2 (+/- pi/2) are CCW from East, "to" convention
        dir1 = np.arctan2(ds["wave_b1_value"], ds["wave_a1_value"])
        dir2 = 0.5 * np.arctan2(ds["wave_b2_value"], ds["wave_a2_value"])

        # Spreading function
        # Subtract dataset variable to get dimensions right
        D = (1 / np.pi) * (
            0.5
            + r1 * np.cos(-1 * (dir1 - np.deg2rad(ds["direction"])))
            + r2 * np.cos(-2 * (dir2 - np.deg2rad(ds["direction"])))
        )

        # Wave energy density is units of Hz and degrees
        ds["wave_dir_energy_density"] = ds["wave_dir_energy_density"].expand_dims(
            dim={"time": ds["time"]}, axis=0
        )
        ds["wave_dir_energy_density"].values = ds["wave_energy_density"] * np.rad2deg(D)

        # Reset direction coordinate so that the spreading function D corresponds
        # to CW from North, "from" convention, instead of CCW from East, "to" convention.
        dir_from_N = (270 - ds["direction"]) % 360
        dirN = xr.DataArray(
            dir_from_N,
            coords={"direction": dir_from_N},
            attrs=ds["direction"].attrs,
        )
        ds = ds.assign_coords({"direction": dirN})
        # sort direction properly so that it runs 0 - 360
        ds = ds.sortby(dataset["direction"])

        return ds

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
            dataset["time_aux"],
            dataset["sea_surface_temperature"],
            ".-",
            label="Sea Surface Temperature",
            color=haline(0.15),
        )
        ax[3].set(ylabel="Temperature\n[deg C]")

        if "air_temperature" in dataset:
            ax[3].plot(
                dataset["time_aux"],
                dataset["air_temperature"],
                ".-",
                label="Air Temperature",
                color="black",
            )

        if "air_pressure" in dataset:
            ax[4].plot(
                dataset["time_aux"],
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
        ax.scatter(dataset["longitude"], dataset["latitude"])
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
