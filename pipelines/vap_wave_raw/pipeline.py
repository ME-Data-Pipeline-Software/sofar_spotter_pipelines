import numpy as np
import xarray as xr
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mhkit import wave, dolfyn
from cmocean.cm import amp_r, dense, haline
from tsdat import TransformationPipeline


fs = 2.5  # Hz, Spotter sampling frequency
wat = 1800  # s, window averaging time
freq_slc = [0.0455, 1]  # 22 to 1 s periods


class VapWaves(TransformationPipeline):
    """---------------------------------------------------------------------------------
    VAP pipeline for calculating wave statistics from a Sofar Spotter wave buoy data.
    ---------------------------------------------------------------------------------"""

    def hook_customize_input_datasets(self, input_datasets) -> Dict[str, xr.Dataset]:
        # Code hook to customize any input datasets prior to datastreams being combined
        # and data converters being run.

        # Need to write in frequency coordinate that will be used later
        for key in input_datasets:
            if "pos" in key:
                # Create FFT frequency vector
                nfft = fs * wat // 6
                f = np.fft.fftfreq(int(nfft), 1 / fs)
                # Use only positive frequencies
                freq = np.abs(f[1 : int(nfft / 2.0 + 1)])
                # Trim frequency vector to > 0.0455 Hz (wave periods between 1 and 22 s)
                freq = freq[np.where((freq > freq_slc[0]) & (freq <= freq_slc[1]))]
                input_datasets[key] = input_datasets[key].assign_coords(
                    {"frequency": freq.astype("float32")}
                )

                return input_datasets

    def hook_customize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        # (Optional) Use this hook to modify the dataset before qc is applied

        ds = dataset.copy()

        # Fill small gps so we can calculate a wave spectrum
        for key in ["x", "y", "z"]:
            ds[key] = ds[key].interpolate_na(
                dim="time", method="linear", max_gap=np.timedelta64(5, "s")
            )

        # Create 2D tensor for spectral analysis
        disp = xr.DataArray(
            data=np.array(
                [
                    ds["x"],
                    ds["y"],
                    ds["z"],
                ]
            ),
            coords={"dir": ["x", "y", "z"], "time": ds["time"]},
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

        # Trim dataset length
        ds = ds.isel(time=slice(None, len(psd["time"])))
        # Set time coordinates
        time = xr.DataArray(
            psd["time"], coords={"time": psd["time"]}, attrs=ds["time"].attrs
        )
        ds = ds.assign_coords({"time": time})
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

        return ds.drop(("x", "y", "z"))

    def hook_finalize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        # (Optional) Use this hook to modify the dataset after qc is applied
        # but before it gets saved to the storage area
        return dataset

    def hook_plot_dataset(self, dataset: xr.Dataset):
        # (Optional, recommended) Create plots.
        plt.style.use("default")  # clear any styles that were set before

        # Wave spectrum
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.subplots_adjust(left=0.14, right=0.95, top=0.95, bottom=0.1)
        ax.loglog(
            dataset["frequency"],
            dataset["wave_energy_density"].mean("time"),
            label="vertical",
        )
        m = -4
        x = np.logspace(-1, 0)
        y = 10 ** (-4) * x**m
        ax.loglog(x, y, "--", c="black", label="f^-4")
        ax.set(
            ylim=(0.0001, 10),
            xlabel="Frequency [Hz]",
            ylabel="Energy Density [m^2/Hz]",
        )
        plot_file = self.get_ancillary_filepath(title="elevation_spectrum")
        fig.savefig(plot_file)

        # Wave stats
        fig, ax = plt.subplots(3, 1, figsize=(10, 7), constrained_layout=True)
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

        for a in ax:
            a.legend(loc="upper left", bbox_to_anchor=[1.01, 1.0], handlelength=1.5)
        for a in ax[:-1]:
            a.set(xticklabels=[])
        date = dataset.time[0].values.astype(str).split("T")[0]
        ax[0].set(title=f"{dataset.datastream} on {date}")
        ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax[-1].set(xlabel="Time (UTC)")

        plot_file = self.get_ancillary_filepath(title="wave_stats")
        fig.savefig(plot_file)
        plt.close("all")
