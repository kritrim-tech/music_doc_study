import copy



def apply_mne_butterworth(raw_eeg, apply_notch=False, notch_freq=50, notch_order=4, filter_type='bandpass', cutoff_freq_low=0.5, cutoff_freq_high=80, order_low=4, order_high=4, phase='zero-double', picks='eeg', sfreq=None):
    """
    Apply an IIR Butterworth filter to an MNE Raw object.
    
    Parameters:
    - raw: mne.io.Raw object to filter.
    - filter_type: 'lowpass', 'highpass', or 'bandpass'.
    - cutoff_freq: Cutoff frequency (float for low/high-pass, tuple (l_freq, h_freq) for bandpass).
    - order: Filter order (int, default 4).
    - phase: 'zero-double' (zero-phase, forward-backward) or 'forward' (causal).
    - picks: Channels to filter (e.g., 'eeg', list of channel names).
    - sfreq: Sampling frequency (Hz, default from raw.info['sfreq']).
    
    Returns:
    - raw_filtered: Filtered Raw object.
    """

    eeg_copy = copy.deepcopy(raw_eeg)  # Ensure the original EEG data remains unchanged


    if apply_notch:
        eeg_copy.notch_filter(freqs=notch_freq, trans_bandwidth=notch_order)
    


    if sfreq is None:
        sfreq = eeg_copy.info['sfreq']
    
    # Set l_freq and h_freq based on filter_type
    if filter_type == 'lowpass':
        l_freq, h_freq = cutoff_freq_low, None

        # Apply Butterworth filter
        eeg_copy.filter(
            l_freq=l_freq,
            h_freq=h_freq,
            method='iir',
            iir_params=dict(ftype='butter', order=order_low),
            picks=picks,
            phase=phase,
            verbose=True
        )

    elif filter_type == 'highpass':
        l_freq, h_freq = None, cutoff_freq_high
        eeg_copy.filter(l_freq=None, l_trans_bandwidth=order_high, h_freq=h_freq)


    elif filter_type == 'bandpass':
        l_freq, h_freq = cutoff_freq_low, cutoff_freq_high

        # Apply Butterworth filter
        eeg_copy.filter(
            l_freq=l_freq,
            h_freq=None,
            method='iir',
            iir_params=dict(ftype='butter', order=order_low),
            picks=picks,
            phase=phase,
            verbose=True
        )

        # eeg_copy.filter(
        #     l_freq=None,
        #     h_freq=h_freq,
        #     method='iir',
        #     iir_params=dict(ftype='butter', order=order_high),
        #     picks=picks,
        #     phase=phase,
        #     verbose=True
        # )

        eeg_copy.filter(h_freq=h_freq, h_trans_bandwidth=order_high, l_freq=None)


    elif filter_type == 'bandpassp4':
        l_freq, h_freq = cutoff_freq_low, cutoff_freq_high

        # Apply Butterworth filter
        eeg_copy.filter(
            l_freq=l_freq,
            h_freq=None,
            method='iir',
            iir_params=dict(ftype='butter', order=order_low),
            picks=picks,
            phase=phase,
            verbose=True
        )

        eeg_copy.filter(
            l_freq=None,
            h_freq=h_freq,
            method='iir',
            iir_params=dict(ftype='butter', order=order_high),
            picks=picks,
            phase=phase,
            verbose=True
        )

    else:
        raise ValueError("filter_type must be 'lowpass', 'highpass', or 'bandpass'")
    
    return eeg_copy


