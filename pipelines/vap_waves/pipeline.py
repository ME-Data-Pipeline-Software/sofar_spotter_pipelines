import numpy as np
import xarray as xr
from typing import Dict
import matplotlib.pyplot as plt
from mhkit import wave, dolfyn
from cmocean.cm import amp_r, dense, haline

from tsdat import TransformationPipeline


class VapWaves(TransformationPipeline):
    """---------------------------------------------------------------------------------
    This is an example pipeline meant to demonstrate how one might set up a
    pipeline using this template repository.

    ---------------------------------------------------------------------------------"""

    def hook_customize_input_datasets(self, input_datasets) -> Dict[str, xr.Dataset]:
        # Code hook to customize any input datasets prior to datastreams being combined
        # and data converters being run.

        # Need to write in frequency coordinate that will be used later
        for key in input_datasets:
            if "pos" in key:
                # Create FFT frequency vector
                fs = 2.5
                nfft = fs * 600 // 3
                f = np.fft.fftfreq(int(nfft), 1 / fs)
                # Use only positive frequencies
                freq = np.abs(f[1 : int(nfft / 2.0 + 1)])
                # Trim frequency vector to > 0.0455 Hz (wave periods smaller than 22 s)
                freq = freq[np.where(freq > 0.0455)]
                input_datasets[key] = input_datasets[key].assign_coords(
                    {"frequency": freq}
                )

                return input_datasets

    def hook_customize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        # (Optional) Use this hook to modify the dataset before qc is applied
        ds = dataset.copy()
        
        # Fill nans so we can calculate a wave spectrum
        for key in ["x", "y", "z"]:
            ds[key] = ds[key].interpolate_na(
                dim="time", method="cubic", max_gap=np.timedelta64(10, "s")
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
            coords={"dir": ["x", "y", "z"], "time": ds.time},
        )

        ## Using dolfyn to create spectra
        fs = 2.5
        nbin = fs * 600
        fft_tool = dolfyn.adv.api.ADVBinner(
            n_bin=nbin, fs=fs, n_fft=nbin // 3, n_fft_coh=nbin // 3
        )
        # Trim frequency vector to > 0.0455 Hz (wave periods smaller than 22 s)
        slc_freq = slice(0.0455, None)
        
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

        # Check that k is less than or equal to 1
        k = np.sqrt((Sxx + Syy) / Szz)

        ## Wave direction and spread
        a1 = Cxz.values / np.sqrt((Sxx + Syy) * Szz).values
        b1 = Cyz.values / np.sqrt((Sxx + Syy) * Szz).values
        a2 = (Sxx - Syy) / (Sxx + Syy)
        b2 = 2 * Cxy.values / (Sxx + Syy)

        theta = np.arctan(b1 / a1)
        phi = np.sqrt(2 * (1 - np.sqrt(a1**2 + b1**2)))
        theta = np.nan_to_num(theta)  # fill missing data with zeroes
        phi = np.nan_to_num(phi)  # fill missing data with zeroes

        direction = np.arange(len(Tp))
        spread = np.arange(len(Tp))
        for i in range(len(Tp)):
            direction[i] = np.rad2deg(
                np.trapz(theta[i], psd.freq)
            )  # degrees CW from North
            spread[i] = np.rad2deg(np.trapz(phi[i], psd.freq))

        # Trim dataset length
        ds = ds.isel(time=slice(None, len(psd["time"])))
        # Set time coordinates
        time = xr.DataArray(
            psd["time"], coords={"time": psd["time"]}, attrs=ds["time"].attrs
        )
        ds = ds.assign_coords({"time": time})
        ds["wave_energy_density"].values = Szz
        ds["wave_hs"].values = Hs.to_xarray()["Hm0"]
        ds["wave_te"].values = Te.to_xarray()["Te"]
        ds["wave_tp"].values = Tp.to_xarray()["Tp"]
        ds["wave_ta"].values = Ta.to_xarray()["Tm"]
        ds["wave_tz"].values = Tz.to_xarray()["Tz"]
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

        # Set the format of the x-axis tick labels
        plt.style.use("default")  # clear any styles that were set before
        plt.style.use("shared/styling.mplstyle")

        # Plot wave spectra
        t = dolfyn.time.dt642date(dataset.time)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.loglog(
            dataset["frequency"],
            dataset["wave_energy_density"].mean("time"),
            label="vertical",
        )
        m = -4
        x = np.logspace(-1, 0.5)
        y = 10 ** (-4) * x**m
        ax.loglog(x, y, "--", c="black", label="f^-4")
        ax.set(
            ylim=(0.00001, 10),
            xlabel="Frequency [Hz]",
            ylabel="Energy Density [m^2/Hz]",
        )
        plot_file = self.get_ancillary_filepath(title="elevation_spectrum")
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
