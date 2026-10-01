import numpy as np
import xarray as xr
from scipy import stats
import pycwt
from mhkit import wave, dolfyn

dir_bin_width = 5  # degrees, directional distribution histogram bin width
dir_bins = np.arange(-180, 180 + dir_bin_width, dir_bin_width)
constants = dict(
    fs=2.5,  # Hz, Spotter sampling frequency
    wat=1800,  # s, window averaging time
    fft_decimation=6,  # 1/fraction of bin length for FFT vector
    pct_overlap=0.5,
    freq_slc=[0.045, 0.5],
    dir_centers=((dir_bins[:-1] + dir_bins[1:]) / 2).astype("float32"),
)


def wave_analysis(dataset, wavelet_basic_stats=False, directional_spectra=False):
    # Fill small gaps so we can calculate a wave spectrum
    for key in ["x", "y", "z"]:
        if not dataset[key].size:
            raise ValueError(f"dataset[{key}] is empty")
        dataset[key] = dataset[key].interpolate_na(
            dim="time", method="linear", max_gap=np.timedelta64(5, "s")
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
    nbin = constants["fs"] * constants["wat"]
    fft_tool = dolfyn.adv.api.ADVBinner(
        n_bin=nbin,
        fs=constants["fs"],
        n_fft=nbin // constants["fft_decimation"],
        n_fft_coh=nbin // constants["fft_decimation"],
    )
    # Trim frequency vector to > 0.0455 Hz (wave periods smaller than 22 s)
    slc_freq = slice(constants["freq_slc"][0], constants["freq_slc"][1])

    # Auto-spectra
    psd = fft_tool.power_spectral_density(
        disp, freq_units="Hz", pct_overlap=constants["pct_overlap"]
    )
    psd = psd.sel(freq=slc_freq)
    Sxx = psd.sel(S="Sxx")
    Syy = psd.sel(S="Syy")
    Szz = psd.sel(S="Szz")

    # Cross-spectra
    csd = fft_tool.cross_spectral_density(
        disp, freq_units="Hz", pct_overlap=constants["pct_overlap"]
    )
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

    # Calculate wave direction and spread
    a1 = Cxz.values / np.sqrt((Sxx + Syy) * Szz)
    b1 = Cyz.values / np.sqrt((Sxx + Syy) * Szz)
    a2 = (Sxx - Syy) / (Sxx + Syy)
    b2 = 2 * Cxy.values / (Sxx + Syy)

    ## Mean wave direction and spread
    # Integrate first fourier coefficients to get average energy
    a1_int = np.trapezoid(a1.values, a1["freq"].values, axis=-1)
    b1_int = np.trapezoid(b1.values, b1["freq"].values, axis=-1)
    # Find angles
    theta_mean = np.rad2deg(np.arctan2(b1_int, a1_int))
    phi_mean = np.rad2deg(np.sqrt(2 * (1 - np.sqrt(a1_int**2 + b1_int**2))))
    # convert "degrees CCW from East" ("to" convention) to "degrees CW from North" ("from" convention)
    dir_mean = (270 - theta_mean) % 360
    # Set direction from -180 to 180
    dir_mean[dir_mean > 180] -= 360

    ## Peak wave direction and spread
    # (Use first moment fourier coefficients to keep it simple)
    theta = np.rad2deg(np.arctan2(b1, a1))
    phi = np.rad2deg(np.sqrt(2 * (1 - np.sqrt(a1**2 + b1**2))))
    # Get peak frequency - fill nan slices with 0
    peak_idx = psd[2].fillna(0).argmax("freq")
    phi_peak = phi[:, peak_idx]
    # degrees CW from North ("from" convention)
    dir_peak = (270 - theta[:, peak_idx]) % 360
    # Set direction from -180 to 180
    dir_peak[dir_peak > 180] -= 360

    if directional_spectra:
        ## Discrete Fourier Transform (DFT) Method for Directional Wave Spectrum
        # Calculate directional wave spectrum
        r1 = np.sqrt(a1**2 + b1**2)
        r2 = np.sqrt(a2**2 + b2**2)
        # dir1 (+/- pi) and dir2 (+/- pi/2) are CCW from East, "to" convention
        dir1 = np.arctan2(b1, a1)  # "mean wave direction"
        dir2 = 0.5 * np.arctan2(b2, a2)  # "principal wave direction"

        # ds["direction"] is CW from North, "from" convention (shared with the wavelet
        # method) - convert to CCW from East, "to" convention to match dir1/dir2
        # (this transform is self-inverse, so the same formula converts both ways)
        theta = np.deg2rad((270 - dataset["direction"]) % 360)

        # Spreading function (parametric estimate of the directional distribution function)
        # Subtract dataset variable to get dimensions right
        spread_func = (1 / np.pi) * (
            0.5 + r1 * np.cos(-1 * (dir1 - theta)) + r2 * np.cos(-2 * (dir2 - theta))
        )
        # Convert D from density per radian to density per degree (direction coord is in degrees)
        # "spread_func" DataArray inherits the original degree convention,
        # so no need to remap it to CW-North
        spread_func *= np.pi / 180

    ## Wavelets
    # Method specifically derived from Pelaez-Zapata et al (2024). They use
    # velocity instead of position to estimate direction, but the results
    # appear equivalent.
    w0 = 6  # According to Farge (1992), a commonly used value for the Morlet wavelet
    mother = pycwt.Morlet(w0)
    freq_target = psd["freq"].values
    # Remove NaN values
    mask = disp.isnull()
    disp = disp.fillna(0)
    # Wavelet analysis
    Wx_values, _, freq_out, _, _, _ = pycwt.cwt(
        disp[0].values, 1 / constants["fs"], wavelet=mother, freqs=freq_target
    )
    Wy_values, _, _, _, _, _ = pycwt.cwt(
        disp[1].values, 1 / constants["fs"], wavelet=mother, freqs=freq_target
    )
    Wz_values, _, _, _, _, _ = pycwt.cwt(
        disp[2].values, 1 / constants["fs"], wavelet=mother, freqs=freq_target
    )
    # Set as xarray dataarrays
    Wx = xr.DataArray(
        Wx_values,
        coords={"freq": freq_out, "time": dataset["time"].values},
        dims=["freq", "time"],
    )
    Wy = xr.DataArray(
        Wy_values,
        coords={"freq": freq_out, "time": dataset["time"].values},
        dims=["freq", "time"],
    )
    Wz = xr.DataArray(
        Wz_values,
        coords={"freq": freq_out, "time": dataset["time"].values},
        dims=["freq", "time"],
    )
    # Remask data
    Wx = Wx.where(~mask[0].values)
    Wy = Wy.where(~mask[1].values)
    Wz = Wz.where(~mask[2].values)

    # pycwt.cwt implements the exact discrete Torrence & Compo (1998)
    # normalization, so |W|^2 is directly comparable (shape and magnitude)
    # to the Fourier PSD (m^2/Hz)
    Wzz_psd = abs(Wz) ** 2
    # Estimate wave direction from X and Y wavelet components
    # Cross wavelet transform: magnitude gives cross-wavelet power, angle gives relative phase
    Wyz = Wy * np.conj(Wz)
    Wxz = Wx * np.conj(Wz)
    # Find wave direction matrix and convert from "CCW from E" ("from" convention) to "CW from N" ("to" convention)
    direction_cwt = (270 - np.rad2deg(np.arctan2(Wyz.real, Wxz.real))) % 360
    # Set to +/-180 degrees
    direction_cwt = ((direction_cwt + 180) % 360) - 180

    # Bin-average wavelet-based parameters to reduce noise
    step = int((1 - constants["pct_overlap"]) * fft_tool.n_bin)
    Wzz_psd_avg = fft_tool.mean(Wzz_psd, axis=-1, step=step)

    # Average wave direction
    direction_cwt_reshaped = fft_tool.reshape(direction_cwt, step=step)
    direction_cwt_avg = stats.circmean(
        direction_cwt_reshaped, low=-180, high=180, axis=-1
    )

    ## Directional distribution function
    # Histogram of direction within each cwt_tool time bin (same windows as
    # Wzz_psd_cwt/D_cwt above), normalized so it integrates to 1 over direction,
    # for each time/frequency. This is an empirical (non-parametric) estimate
    def _hist_per_freq(x, bins):
        x = x[~np.isnan(x)]
        if x.size == 0:
            return np.full(bins.size - 1, np.nan)
        counts, _ = np.histogram(x, bins=bins, density=True)
        return counts

    # Reshape the fast-time direction estimate into cwt_tool's windows: (freq, time, n_bin)
    dir_samples = fft_tool.reshape(direction_cwt.values, step=step)
    dir_distr_func = np.apply_along_axis(_hist_per_freq, -1, dir_samples, dir_bins)
    # (freq, time, direction) -> (time, freq, direction)
    dir_distr_func = np.moveaxis(dir_distr_func, 0, 1)

    if wavelet_basic_stats:
        # Wavelet wave stats
        # Note: Hs appears overestimated compared to PSD method, others are comparable
        Hs_cwt = wave.resource.significant_wave_height(Wzz_psd)
        Te_cwt = wave.resource.energy_period(Wzz_psd)
        Ta_cwt = wave.resource.average_wave_period(Wzz_psd)
        Tp_cwt = wave.resource.peak_period(Wzz_psd)
        Tz_cwt = wave.resource.average_zero_crossing_period(Wzz_psd)
        # Find peak wave direction
        peak_idx_ctw = abs(Wzz_psd).fillna(0).argmax("freq")
        Dp_cwt = direction_cwt.isel(freq=peak_idx_ctw)

        Hs_cwt = fft_tool.mean(Hs_cwt, step=step)
        Te_cwt = fft_tool.mean(Te_cwt, step=step)
        Ta_cwt = fft_tool.mean(Ta_cwt, step=step)
        Tp_cwt = fft_tool.mean(Tp_cwt, step=step)
        Tz_cwt = fft_tool.mean(Tz_cwt, step=step)
        Dp_cwt = fft_tool.mean(Dp_cwt, step=step)

    ## Create output dataset
    # Trim dataset length
    ds = dataset.isel(
        time=slice(None, len(psd["time_psd"])),
    )
    # Set time coordinates
    time = xr.DataArray(
        psd["time_psd"].values,
        coords={"time": psd["time_psd"].values},
        attrs=dataset["time"].attrs,
    )

    ds = ds.assign_coords({"time": time})
    # Make sure mhkit vars are set to float32
    ds["wave_energy_density"].values = Szz
    if time.size > 1:
        ds["wave_hs"].values = Hs.to_xarray().astype("float32")
        ds["wave_te"].values = Te.to_xarray().astype("float32")
        ds["wave_tp"].values = Tp.to_xarray().astype("float32")
        ds["wave_ta"].values = Ta.to_xarray().astype("float32")
        ds["wave_tz"].values = Tz.to_xarray().astype("float32")
    else:
        ds["wave_hs"].values = np.array([Hs]).astype("float32")
        ds["wave_te"].values = np.array([Te]).astype("float32")
        ds["wave_tp"].values = np.array([Tp]).astype("float32")
        ds["wave_ta"].values = np.array([Ta]).astype("float32")
        ds["wave_tz"].values = np.array([Tz]).astype("float32")
    ds["wave_check_factor"].values = k
    ds["wave_dp"].values = dir_peak.astype("float32")
    ds["wave_dm"].values = dir_mean.astype("float32")
    ds["wave_sp"].values = phi_peak.astype("float32")
    ds["wave_sm"].values = phi_mean.astype("float32")

    ds["wave_direction"].values = direction_cwt_avg.T
    ds["wavelet_energy_density"].values = Wzz_psd_avg.T
    ds["directional_distr_func"].values = dir_distr_func

    if wavelet_basic_stats:
        ds["wave_hs_cwt"].values = Hs_cwt.astype("float32")
        ds["wave_te_cwt"].values = Te_cwt.astype("float32")
        ds["wave_tp_cwt"].values = Tp_cwt.astype("float32")
        ds["wave_ta_cwt"].values = Ta_cwt.astype("float32")
        ds["wave_tz_cwt"].values = Tz_cwt.astype("float32")
        ds["wave_dp_cwt"].values = Dp_cwt.astype("float32")

    if directional_spectra:
        ds["wave_a1_value"].values = a1
        ds["wave_b1_value"].values = b1
        ds["wave_a2_value"].values = a2
        ds["wave_b2_value"].values = b2
        ds["spreading_func"].values = spread_func

    return ds
