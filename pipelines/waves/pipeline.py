#import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from tsdat import IngestPipeline, get_start_date_and_time_str, get_filename
#import dolfyn
#from mhkit import wave


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

        ## Analysis for VAP
        # ds = dataset
        # disp = xr.DataArray(
        #     data=np.array(
        #         [
        #             ds["surge"],
        #             ds["sway"],
        #             ds["heave"],
        #         ]
        #     ),
        #     coords={"dir": ["x", "y", "z"], "time": ds.time},
        # )
        # ## Using dolfyn to create spectra
        # fft_tool = dolfyn.adv.api.ADVBinner(
        #     ds.nbin, ds.fs, n_fft=ds.nbin / 3, n_fft_coh=ds.nbin / 3
        # )
        # psd = fft_tool.calc_psd(ds.disp, freq_units="Hz")
        # csd = fft_tool.calc_csd(ds.disp, freq_units="Hz")

        # psd = psd.sel(freq=slice(0.0455, None))
        # Sxx = psd.sel(S="Sxx")
        # Syy = psd.sel(S="Syy")
        # Szz = psd.sel(S="Szz")
        # pd_Szz = Szz.T.to_pandas()

        # ## Wave analysis using MHKiT
        # Hs = wave.resource.significant_wave_height(pd_Szz)
        # Te = wave.resource.energy_period(pd_Szz)

        # ## Wave direction and spread
        # Cxz = csd.sel(C="Cxz").real
        # Cyz = csd.sel(C="Cyz").real

        # # Check that k is less than 1
        # k = np.sqrt(
        #     (Sxx + Syy) / Szz
        # )  # wavenumber approx - need to take into account shallow water
        # plt.figure()
        # plt.loglog(psd['freq'], k.mean("time"))
        # plt.xlabel("Frequency [Hz]")
        # plt.ylabel("k [nondim]")

        # a = Cxz / (k * Szz)
        # b = Cyz / (k * Szz)
        # theta = np.arctan(b / a) * (180 / np.pi)  # degrees CCW from East
        # # theta = dolfyn.tools.misc.convert_degrees(theta) # degrees CW from North
        # phi = np.sqrt(2 * (1 - np.sqrt(a**2 + b**2))) * (180 / np.pi)
        # phi = phi.fillna(0)  # fill missing data

        # direction = np.arange(len(ds['time']))
        # spread = np.arange(len(ds['time']))
        # for i in range(len(ds['time'])):
        #     direction[i] = np.trapz(theta[i], psd['freq'])
        #     spread[i] = np.trapz(phi[i], psd['freq'])

        return dataset

    def hook_plot_dataset(self, dataset: xr.Dataset):
        ds = dataset
        location = self.dataset_config.attrs.location_id
        datastream: str = self.dataset_config.attrs.datastream

        date, time = get_start_date_and_time_str(dataset)

        plt.style.use("default")  # clear any styles that were set before
        plt.style.use("shared/styling.mplstyle")

        with self.storage.uploadable_dir(datastream) as tmp_dir:
            fig, ax = plt.subplots()

            ax.plot(ds.time, ds["surge"], label="surge")
            ax.plot(ds.time, ds["sway"], label="sway")
            ax.plot(ds.time, ds["heave"], label="heave")

            ax.set_title("")  # Remove bogus title created by xarray
            ax.legend(ncol=2, bbox_to_anchor=(1, -0.05))
            ax.set_ylabel("Buoy Displacement [m]")
            ax.set_xlabel("Time [UTC]")
            # format_time_xticks(ax, date_format="%Y-%m-%d %H:%M")
            plt.legend()

            plot_file = get_filename(
                dataset, title="buoy_displacement", extension="png"
            )
            fig.savefig(tmp_dir / plot_file)
            plt.close(fig)
        pass
