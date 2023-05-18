import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mpldt

from mhkit import wave
import dolfyn


ds = xr.open_dataset(
    "storage/root/data/clallam.vap_pos.b1/clallam.vap_pos.b1.20210824.180425.nc"
)

# Trim frequency vector from 0.0455 to 1 Hz (wave period of 22 to 1 s)
slc_freq = slice(0.0455, 1)

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
fft_tool = dolfyn.adv.api.ADVBinner(
    ds.fs * 600, ds.fs, n_fft=ds.fs * 600 / 3, n_fft_coh=ds.fs * 600 / 3
)
# Auto-spectra
psd = fft_tool.calc_psd(disp, freq_units="Hz")
psd = psd.sel(freq=slc_freq)
Sxx = psd.sel(S="Sxx")
Syy = psd.sel(S="Syy")
Szz = psd.sel(S="Szz")

# Cross-spectra
csd = fft_tool.calc_csd(disp, freq_units="Hz")
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
    direction[i] = 90 - np.rad2deg(
        np.trapz(theta[i], psd.freq)
    )  # degrees CW from North
    spread[i] = np.rad2deg(np.trapz(phi[i], psd.freq))


# Save to a new dataset
ds_avg = xr.Dataset()
ds_avg["Szz"] = Szz
ds_avg["Hs"] = Hs.to_xarray()["Hm0"]
ds_avg["Te"] = Te.to_xarray()["Te"]
ds_avg["Tp"] = Tp.to_xarray()["Tp"]
ds_avg["Ta"] = Ta.to_xarray()["Tm"]
ds_avg["Tz"] = Tz.to_xarray()["Tz"]
ds_avg["k"] = k
ds_avg["a1"] = xr.DataArray(a1, dims=["time", "freq"])
ds_avg["b1"] = xr.DataArray(b1, dims=["time", "freq"])
ds_avg["a2"] = xr.DataArray(a2, dims=["time", "freq"])
ds_avg["b2"] = xr.DataArray(b2, dims=["time", "freq"])
ds_avg["direction"] = xr.DataArray(direction, dims=["time"])
ds_avg["spread"] = xr.DataArray(spread, dims=["time"])

ds_avg["lat"] = xr.DataArray(fft_tool.mean(ds["lat"].values), dims=["time"])
ds_avg["lon"] = xr.DataArray(fft_tool.mean(ds["lon"].values), dims=["time"])

ds_avg.to_netcdf(
    "storage/root/data/clallam.vap_pos.c0/clallam.pos.c0.20210824.180425.nc"
)
