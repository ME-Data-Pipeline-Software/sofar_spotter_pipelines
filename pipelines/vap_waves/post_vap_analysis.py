import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mpldt

from mhkit import wave
import dolfyn


ds = xr.open_dataset(
    "storage/root/data/clallam.vap_waves.b1/clallam.vap_waves.b1.20210824.180425.nc"
)
slc_freq = slice(0.0455, 1)

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
psd = fft_tool.calc_psd(disp, freq_units="Hz")
psd = psd.sel(freq=slc_freq)
csd = fft_tool.calc_csd(disp, freq_units="Hz")
csd = csd.sel(coh_freq=slc_freq)
t = dolfyn.time.dt642date(psd.time)

Sxx = psd.sel(S="Sxx")
Syy = psd.sel(S="Syy")
Szz = psd.sel(S="Szz")
Cxz = csd.sel(C="Cxz").real
Cyz = csd.sel(C="Cyz").real

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
a = Cxz.values / np.sqrt((Sxx + Syy) * Szz).values
b = Cyz.values / np.sqrt((Sxx + Syy) * Szz).values
theta = np.arctan(b / a)
phi = np.sqrt(2 * (1 - np.sqrt(a**2 + b**2)))
theta = np.nan_to_num(theta)  # fill missing data
phi = np.nan_to_num(phi)  # fill missing data

direction = np.arange(len(t))
spread = np.arange(len(t))
for i in range(len(t)):
    direction[i] = 90 - np.rad2deg(
        np.trapz(theta[i], psd.freq)
    )  # degrees CW from North
    spread[i] = np.rad2deg(np.trapz(phi[i], psd.freq))

ds_avg = xr.Dataset()
ds_avg["Sxx"] = Sxx
ds_avg["Syy"] = Syy
ds_avg["Szz"] = Szz
ds_avg["Hs"] = Hs.to_xarray()["Hm0"]
ds_avg["Te"] = Te.to_xarray()["Te"]
ds_avg["Cxz"] = Cxz
ds_avg["Cyz"] = Cyz
ds_avg["a1"] = xr.DataArray(a, dims=["time", "freq"])
ds_avg["b1"] = xr.DataArray(b, dims=["time", "freq"])
ds_avg["direction"] = xr.DataArray(direction, dims=["time"])
ds_avg["spread"] = xr.DataArray(spread, dims=["time"])

plt.figure()
plt.loglog(psd.freq, pd_Szz.mean(axis=1), label="vertical")
m = -4
x = np.logspace(-1, 0.5)
y = 10 ** (-5) * x**m
plt.loglog(x, y, "--", c="black", label="f^-4")
plt.ylim((0.00001, 10))
plt.xlabel("Frequency [Hz]")
plt.ylabel("Energy Density [m^2/Hz]")
plt.savefig("storage/root/ancillary/clallam.vap_waves.b1/wave_spectra.png")

fig, ax = plt.subplots(2, figsize=(15, 10))
ax[0].scatter(t, Hs)
ax[0].set_xlabel("Time")
ax[0].xaxis.set_major_formatter(mpldt.DateFormatter("%D"))
ax[0].set_ylabel("Significant Wave Height [m]")

ax[1].scatter(t, Te)
ax[1].set_xlabel("Time")
ax[1].xaxis.set_major_formatter(mpldt.DateFormatter("%D"))
ax[1].set_ylabel("Energy Period [s]")
plt.savefig("storage/root/ancillary/clallam.vap_waves.b1/wave_stats.png")

plt.figure()
plt.loglog(psd.freq, k.mean("time"))
plt.xlabel("Frequency [Hz]")
plt.ylabel("k [nondim]")
plt.savefig("storage/root/ancillary/clallam.vap_waves.b1/wave_check.png")

ax = plt.figure(figsize=(20, 10)).add_axes([0.14, 0.14, 0.8, 0.74])
ax.scatter(t, direction, label="Wave direction (towards)")
ax.scatter(t, spread, label="Wave spread")
ax.set_xlabel("Time")
ax.xaxis.set_major_formatter(mpldt.DateFormatter("%D"))
ax.set_ylabel("deg")
plt.savefig("storage/root/ancillary/clallam.vap_waves.b1/wave_direction.png")
