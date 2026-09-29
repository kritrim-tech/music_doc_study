import mne
import numpy as np
import copy

def compute_psd_mne(data, band_ranges, overlap, segment_size, f_min, f_max, power_type="relative"):

    # if data.ndim == 1:
    #     data = data[np.newaxis, :]  # Convert 1D to 2D for consistency

    # Compute power spectral density (PSD) using Welch's method
    #freqs, psd = welch(data, sfreq, nperseg=int(sfreq * 2))  # nperseg = 2-second window (larger=better frequency resoltion), window = Defaults to a Hann window, noverlap=If None, noverlap = nperseg // 2

    picks_eeg = mne.pick_types(data.info, eeg=True)
    # print("Segment Size: ", segment_size)
    # print("Overlap: ", overlap)
    psd_data = data.compute_psd(method='welch', fmin=f_min, fmax=f_max, picks=picks_eeg, n_per_seg=segment_size, n_fft=segment_size, n_overlap=overlap, average=False)
    psd, freqs = psd_data.get_data(return_freqs=True)

    # psd = 10 * np.log10(psd) # Added on 9/1/2026 Actuallly, not needed since fooof requires non logged data and also when we look at the difference between 2 conditions the significance graphs show the same results
    # so it does not matter with which scale we calculate the psds. For an example, check the plots in sub07 where we have an example with and without this line of code, and the significance plots are the same. 

    psds_raw = copy.deepcopy(psd)
    freqs_raw = copy.deepcopy(freqs)
    psd = np.mean(psd, axis=2) #PSD is 3d (channels, freq_bins, segments), mean over segments is computed to then integrate over the freqs
    
    # Compute absolute band power
    abs_band_power = {}
    for band, (fmin, fmax) in band_ranges.items():
        # print(fmin, fmax)
        idx_band = np.logical_and(freqs >= fmin, freqs < fmax)  # Frequency indices for the band
        # print(idx_band)
        abs_band_power[band] = np.trapezoid(psd[:, idx_band], freqs[idx_band], axis=1)  # Compute power using integration

    # If only absolute power is needed, return it
    if power_type == "absolute":
        return abs_band_power, psds_raw, freqs_raw

    # Compute relative band power
    total_power = np.sum(np.array(list(abs_band_power.values())), axis=0)  # Sum across all bands

    rel_band_power = {band: abs_band_power[band] / total_power for band in abs_band_power}

    # Return relative power only if specified
    if power_type == "relative":
        return rel_band_power, psds_raw, freqs_raw

    # If both are needed, return both in a dictionary
    return {"absolute": abs_band_power, "relative": rel_band_power}