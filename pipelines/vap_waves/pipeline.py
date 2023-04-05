import numpy as np
import xarray as xr
# import matplotlib.dates as mdates
# import matplotlib.pyplot as plt

from tsdat import TransformationPipeline
from mhkit import wave
import dolfyn


class VapWaves(TransformationPipeline):
    """---------------------------------------------------------------------------------
    This is an example pipeline meant to demonstrate how one might set up a
    pipeline using this template repository.

    ---------------------------------------------------------------------------------"""

    def hook_customize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        # DEVELOPER: (Optional) Use this hook to modify the dataset before qc is applied

        ## Analysis for VAP
        slc_freq = slice(0.0455, 1)
        ds = dataset
        disp = xr.DataArray(
            data=np.array(
                [
                    ds["surge"],
                    ds["sway"],
                    ds["heave"],
                ]
            ),
            coords={"dir": ["x", "y", "z"], "time": ds.time},
        )
        ## Using dolfyn to create spectra
        fft_tool = dolfyn.adv.api.ADVBinner(ds.nbin, ds.fs, n_fft=ds.nbin/3, n_fft_coh=ds.nbin/3)
        psd = fft_tool.calc_psd(disp, freq_units='Hz')
        psd = psd.sel(freq=slc_freq)
        csd = fft_tool.calc_csd(disp, freq_units='Hz')
        csd = csd.sel(coh_freq=slc_freq)
        t = dolfyn.time.dt642date(psd.time)

        Sxx = psd.sel(S='Sxx')
        Syy = psd.sel(S='Syy')
        Szz = psd.sel(S='Szz')
        Cxz = csd.sel(C='Cxz').real
        Cyz = csd.sel(C='Cyz').real

        ## Wave height and period
        pd_Szz = Szz.T.to_pandas()
        Hs = wave.resource.significant_wave_height(pd_Szz)
        Te = wave.resource.energy_period(pd_Szz)

        ## Wave direction and spread
        Cxz = csd.sel(C="Cxz").real
        Cyz = csd.sel(C="Cyz").real

        # Check that k is less than 1
        # Wavenumber approx - need to take into account shallow water
        k = np.sqrt((Sxx + Syy) / Szz)

        ## Wave direction and spread
        a = Cxz.values / np.sqrt((Sxx+Syy)*Szz).values
        b = Cyz.values / np.sqrt((Sxx+Syy)*Szz).values
        theta = np.arctan(b/a)
        phi = np.sqrt(2*(1 - np.sqrt(a**2 + b**2)))
        phi = np.nan_to_num(phi) # fill missing data

        direction = np.arange(len(t))
        spread = np.arange(len(t))
        for i in range(len(t)):
            direction[i] = 90 - np.rad2deg(np.trapz(theta[i], psd.freq)) # degrees CW from North
            spread[i] = np.rad2deg(np.trapz(phi[i], psd.freq))

        # ds_avg = xr.Dataset()
        # ds_avg['Sxx'] = Sxx
        # ds_avg['Syy'] = Syy
        # ds_avg['Szz'] = Szz
        # ds_avg['Hs'] = Hs.to_xarray()['Hm0']
        # ds_avg['Te'] = Te.to_xarray()['Te']
        # ds_avg['Cxz'] = Cxz
        # ds_avg['Cyz'] = Cyz
        # ds_avg['a1'] = xr.DataArray(a, dims=['time', 'freq'])
        # ds_avg['b1'] = xr.DataArray(b, dims=['time', 'freq'])
        # ds_avg['direction'] = xr.DataArray(direction, dims=['time'])
        # ds_avg['spread'] = xr.DataArray(spread, dims=['time'])

        return dataset

    def hook_finalize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        # DEVELOPER: (Optional) Use this hook to modify the dataset after qc is applied
        # but before it gets saved to the storage area
        return dataset

    def hook_plot_dataset(self, dataset: xr.Dataset):
        # DEVELOPER: (Optional, recommended) Create plots.
        # location = self.dataset_config.attrs.location_id
        # datastream: str = self.dataset_config.attrs.datastream

        # date, time = get_start_date_and_time_str(dataset)

        # plt.style.use("default")  # clear any styles that were set before
        # plt.style.use("shared/styling.mplstyle")

        # with self.storage.uploadable_dir(datastream) as tmp_dir:

        #     fig, ax = plt.subplots()
        #     dataset["example_var"].plot(ax=ax, x="time")  # type: ignore
        #     fig.suptitle(f"Example Variable at {location} on {date} {time}")
        #     plot_file = get_filename(dataset, title="example_plot", extension="png")
        #     fig.savefig(tmp_dir / plot_file)
        #     plt.close(fig)
        # TODO: Better x-axis ticks:
        # Set the x-axis to have ticks spaced by the hour
        # hours = mdates.HourLocator(interval=1)
        # ax.xaxis.set_major_locator(hours)

        # # Set the format of the x-axis tick labels
        # time_format = mdates.DateFormatter('%H:%M')
        # ax.xaxis.set_major_formatter(time_format)
        pass
