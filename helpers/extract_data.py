import numpy as np
import mne
import sys
import copy
print("Python Executable Path: ")
print(sys.executable)





def find_closest_index(array, value):
    """
    Find the index of the closest value in an array.

    Parameters:
    - array: The array to search.
    - value: The value to find the closest to.

    Returns:
    - Index of the closest value.
    """
    array = np.asarray(array)
    closest_index = (np.abs(array - value)).argmin()
    return closest_index



def extract_data_from_streams(data_streams):
    """
    Extracts time_series, time_stamps, and channel names (if BioSemi stream) for all streams.
    Also processes 'MusicStream' to get data between "RecordingStarted" and "RecordingFinished".

    Parameters:
    - data_streams: List of data stream dictionaries.

    Returns:
    - stream_data: Dictionary with all the data from all streams.
    - filtered_data: Dictionary with data filtered between "RecordingStarted" and "RecordingFinished" for all streams.
    """
    stream_data = {}
    filtered_data = {}
    recording_start_time = None
    recording_end_time = None

    # Step 1: Process MusicStream to get the recording start and end times
    stream_indx = 0
    for stream in data_streams:
        if stream['info']['name'][0] == 'MusicStream':
            music_time_series = stream['time_series']
            music_time_stamps = stream['time_stamps']

            for index, trigger_music_stream in enumerate(music_time_series):
                if trigger_music_stream[0] == 'RecordingStarted':
                    start_index = index
                elif trigger_music_stream[0] == 'RecordingFinished':
                    end_index = index

            try:
                recording_start_time = music_time_stamps[start_index]
                recording_end_time = music_time_stamps[end_index]

                # Print the time difference
                recording_duration = recording_end_time - recording_start_time
                print(f"Recording duration: {recording_duration:.2f} seconds")
            except:
                print(f"MusicStream markers 'RecordingStarted' and 'RecordingFinished' not found, Stream Index: {stream_indx} and Stream Name: {stream['info']['name'][0]}")
        stream_indx += 1

    if recording_start_time is None or recording_end_time is None:
        raise ValueError("Recording start and end times could not be determined.")

    # Step 2: Process all other streams
    for stream in data_streams:
        stream_name = stream['info']['name'][0]
        time_series = stream['time_series']
        time_stamps = stream['time_stamps']
        # print(stream_name, stream['info'])

        # Add all data to the dictionary
        stream_data[stream_name] = {
            'time_series': time_series,
            'time_stamps': time_stamps
        }

        print(f"{stream_name} Original Data Length: {len(time_stamps)}")

        if len(time_stamps) > 0:
            # Filter data within the recording window
            start_idx = find_closest_index(time_stamps, recording_start_time)
            end_idx = find_closest_index(time_stamps, recording_end_time)

            filtered_data[stream_name] = {
                'time_series': time_series[start_idx:end_idx + 1],
                'time_stamps': time_stamps[start_idx:end_idx + 1]
            }

            filtered_length = len(filtered_data[stream_name]['time_stamps'])
            sampling_frequency = filtered_length / recording_duration

            print(f"{stream_name} Filtered Data Length: {filtered_length}")
            print(f"{stream_name} Filtered Sampling Frequency: {sampling_frequency:.2f} Hz", "\n")

            # Special handling for BioSemi stream
            if stream_name == 'BioSemi':
                channel_names = []
                channel_n = 137
                for ch_indx in range(channel_n):
                    channel_label = stream['info']['desc'][0]['channels'][0]['channel'][ch_indx]['label']
                    channel_names.append(channel_label)
                stream_data[stream_name]['channel_names'] = channel_names
                filtered_data[stream_name]['channel_names'] = channel_names

    return stream_data, filtered_data, recording_duration









def extract_eeg_segments(data_stream_dict):
    """
    Extract EEG segments corresponding to music trigger segments,
    print duration checks, and remove non-EEG channels 
    (first channel + last 24 channels).

    Args:
        data_stream_dict (dict): Dictionary containing LSL streams.
            - Keys: stream names (e.g., 'MusicTriggers', 'BioSemi')
            - Values: [time_series, time_stamps]

    Returns:
        dict: {audio_name: eeg_segment (np.array of samples, only EEG channels)}
    """
    eeg_data, eeg_timestamps = data_stream_dict["BioSemi"]
    sfreq = 1.0 / np.mean(np.diff(eeg_timestamps))
    triggers, trigger_timestamps = data_stream_dict["MusicTriggers"]

    # remove first channel and last 24 channels
    eeg_data = eeg_data[:, 1:-24]

    eeg_segments = {}
    start_time = None
    start_label = None
    c = 1

    ts_list= []

    for trig, trig_time in zip(triggers, trigger_timestamps):
        trig = trig[0]  # unpack ['START_xxx.wav'] → "START_xxx.wav"

        if trig.startswith("START_"):
            start_time = trig_time
            start_label = trig.replace("START_", "").replace(".wav", "")
            ts_list.append(start_time)

        elif trig.startswith("END_") and start_label is not None:
            end_time = trig_time
            duration = end_time - start_time
            # ts_list.append(end_time)

            print(f"Segment '{c} {start_label}' duration: {duration:.3f} s")

            # Find EEG samples between start and end
            mask = (eeg_timestamps >= start_time) & (eeg_timestamps <= end_time)
            seg_data = eeg_data[mask, :]

            # Ensure unique key
            key = start_label
            suffix = 1
            
            while key in eeg_segments:
                key = f"{start_label}_{suffix}"
                suffix += 1

            eeg_segments[key] = seg_data

            start_time, start_label = None, None  # reset
            c += 1

    print("TOTAL Segments: ", len(eeg_segments))

    return eeg_segments, sfreq, ts_list



















def split_and_concatenate_eeg_segments(eeg_segments, sfreq, montage="biosemi128", epoch_len_sec=30.0, n_reward_songs=2):
    """
    Concatenate EEG segments into two sets:
      1) eeg_segs_all: all except the last two
      2) eeg_reward: only the last two
    Also create an MNE Raw object (only all-but-last-2, with annotations)
    and an EpochsArray (only all-but-last-2, fixed 30s).

    Args:
        eeg_segments (dict): {audio_name: eeg_data (samples x channels)}
        sfreq (float): Sampling frequency
        montage (str): EEG montage (default: "biosemi64")
        epoch_len_sec (float): Desired epoch length in seconds

    Returns:
        tuple:
            - eeg_segs_all (np.ndarray)
            - eeg_reward (np.ndarray)
            - raw (mne.io.RawArray): concatenated raw (all but last 2, with annotations)
            - epochs (mne.EpochsArray): epoched object (fixed 30s, excludes last 2)
    """

    # --- Split into two sets ---
    seg_keys = list(eeg_segments.keys())
    all_but_last2_keys = seg_keys[:-n_reward_songs]
    last2_keys = seg_keys[-n_reward_songs:]

    eeg_segs_all = np.vstack([eeg_segments[k] for k in all_but_last2_keys])
    eeg_reward = np.vstack([eeg_segments[k] for k in last2_keys])

    print(f"eeg_segs_all shape: {eeg_segs_all.shape}")
    print(f"eeg_reward shape: {eeg_reward.shape}")

    # --- Create Raw object (exclude last 2) ---
    all_data = np.vstack([eeg_segments[k] for k in all_but_last2_keys])  # (samples, channels)
    n_samples, n_channels = all_data.shape

    ch_names = mne.channels.make_standard_montage(montage).ch_names[:n_channels]
    ch_types = ["eeg"] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    raw = mne.io.RawArray(all_data.T, info)
    raw.set_montage(montage)
    print(f"Created Raw object: {n_channels} ch, {n_samples/sfreq:.2f}s long")

    # --- Add annotations at boundaries ---
    onsets = []
    durations = []
    descriptions = []
    offset = 0

    for i, k in enumerate(all_but_last2_keys):
        seg_len = eeg_segments[k].shape[0] / sfreq
        offset += seg_len
        if i < len(all_but_last2_keys) - 1:  # no annotation after the last
            onsets.append(offset)
            durations.append(0.5)  # instantaneous marker
            descriptions.append(f"Boundary_{i+1}_{k}")

    annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
    raw.set_annotations(annotations)

    print(f"Added {len(onsets)} annotations at segment boundaries")

    # --- Create EpochsArray (30s fixed length, exclude last 2) ---
    target_len = int(epoch_len_sec * sfreq)
    epochs_data = []

    for k in all_but_last2_keys:
        seg = eeg_segments[k]
        seg_len = seg.shape[0]
        if seg_len > target_len:
            seg_fixed = seg[:target_len, :]
        elif seg_len < target_len:
            print("Segment Length Smaller than the Taget Length: ", target_len)
            pad_len = target_len - seg_len
            seg_fixed = np.vstack([seg, np.zeros((pad_len, seg.shape[1]))])
        else:
            seg_fixed = seg
        epochs_data.append(seg_fixed.T)  # (channels, samples)

    epochs_data = np.stack(epochs_data, axis=0)  # (n_epochs, n_channels, n_times)

    # Dummy events
    n_epochs = len(epochs_data)
    events = np.column_stack([
        np.arange(0, n_epochs * target_len, target_len),
        np.zeros(n_epochs, dtype=int),
        np.arange(1, n_epochs + 1)
    ])
    event_id = {f"seg_{i+1}": i+1 for i in range(n_epochs)}

    epochs = mne.EpochsArray(epochs_data, info=info, events=events,
                             event_id=event_id, tmin=0.0, baseline=None)

    print(f"Created EpochsArray with {n_epochs} epochs (excl. last 2), each {epoch_len_sec}s long")

    return eeg_segs_all, eeg_reward, raw, epochs








def ndarray_to_raw(data, sfreq=2048.0, montage_name="biosemi128"):
    # BioSemi 128 montage (standard 10-5 system)
    montage = mne.channels.make_standard_montage(montage_name)
    ch_names = montage.ch_names
    
    # # Safety check
    # if data.shape[0] != 128:
    #     raise ValueError(f"Data must have 128 channels, got {data.shape[0]}")
    
    # Create info and Raw object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    raw.set_montage(montage)
    
    print(f"Raw created: {raw}")
    print(f"   Duration: {raw.times[-1]:.2f} s")
    print(f"   Channels: {len(raw.ch_names)} EEG")
    
    return raw








def extract_resting_eeg(data_stream_dict, montage='biosemi128'):

    """
    Extracts rest_init / rest_mid / rest_end → one MNE RawArray.
    - Uses correct BioSemi 128 channel names from montage
    - Adds annotations in ONE call
    - Perfect montage matching (no warnings)
    
    Returns
    -------
    raw : mne.io.RawArray
        Concatenated resting EEG with 6 annotations
    """
    # ------------------------------------------------------------------ #
    # 1. Triggers
    # ------------------------------------------------------------------ #
    trigger_labels = [lbl[0] for lbl in data_stream_dict['ExpConditionTriggers'][0]]
    trigger_times  = data_stream_dict['ExpConditionTriggers'][1]

    # ------------------------------------------------------------------ #
    # 2. EEG: (n_samples, n_channels)
    # ------------------------------------------------------------------ #
    eeg_data       = data_stream_dict['BioSemi'][0]      # (n_samples, n_channels)
    eeg_timestamps = data_stream_dict['BioSemi'][1]      # (n_samples,)
    sfreq = 1.0 / np.mean(np.diff(eeg_timestamps))

    n_samples_total, n_channels_total = eeg_data.shape
    n_eeg_channels = n_channels_total - 25  # drop first + last 24

    print("=== RESTING-STATE → MNE RAW ===\n")
    print(f"EEG shape      : {eeg_data.shape}")
    print(f"Sampling rate  : {sfreq:.2f} Hz")
    print(f"EEG channels   : 1:{n_channels_total-24} → {n_eeg_channels} ch\n")

    # ------------------------------------------------------------------ #
    # 3. Extract segments
    # ------------------------------------------------------------------ #
    segments = []
    block_names = []

    for cond in ['rest_init', 'rest_mid', 'rest_end']:
        start_lbl = f'ConditionStart:{cond}'
        end_lbl   = f'ConditionEnd:{cond}'

        if start_lbl not in trigger_labels or end_lbl not in trigger_labels:
            print(f"ERROR: {cond} missing\n")
            continue

        t_start = trigger_times[trigger_labels.index(start_lbl)]
        t_end   = trigger_times[trigger_labels.index(end_lbl)]

        # Closest EEG timestamps
        closest_start = eeg_timestamps[np.argmin(np.abs(eeg_timestamps - t_start))]
        closest_end   = eeg_timestamps[np.argmin(np.abs(eeg_timestamps - t_end))]

        s_start = np.argmin(np.abs(eeg_timestamps - closest_start))
        s_end   = np.argmin(np.abs(eeg_timestamps - closest_end))

        s_start = max(0, s_start)
        s_end   = min(n_samples_total, s_end)

        if s_end <= s_start:
            print(f"{cond.upper()}: no samples\n")
            continue

        seg = eeg_data[s_start:s_end, 1:-24]  # (samples, eeg_ch)
        duration_sec = seg.shape[0] / sfreq

        segments.append(seg)
        block_names.append(cond)

        print(f"{cond.upper()}: {seg.shape[0]} samples → {duration_sec:.1f}s")

    if not segments:
        raise ValueError("No valid rest segments found.")

    # ------------------------------------------------------------------ #
    # 4. Concatenate
    # ------------------------------------------------------------------ #
    data_concat = np.vstack(segments)  # (total_samples, n_eeg_channels)

    # ------------------------------------------------------------------ #
    # 5. Create montage + channel names
    # ------------------------------------------------------------------ #
    print("Creating MNE RawArray...")
    montage_obj = mne.channels.make_standard_montage(montage)
    ch_names = montage_obj.ch_names[:n_eeg_channels]  # Use exact montage names

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data_concat.T, info)  # (n_channels, total_samples)

    # ------------------------------------------------------------------ #
    # 6. Apply montage → perfect match
    # ------------------------------------------------------------------ #
    raw.set_montage(montage_obj)
    print(f"Montage '{montage}' applied successfully!")

    # ------------------------------------------------------------------ #
    # 7. Annotations in ONE call
    # ------------------------------------------------------------------ #
    onset_list = []
    duration_list = []
    desc_list = []
    cum_time = 0.0

    for name, seg in zip(block_names, segments):
        dur = seg.shape[0] / sfreq
        onset_list.extend([cum_time, cum_time + dur])
        duration_list.extend([0.0, 0.0])
        desc_list.extend([f'{name}_start', f'{name}_end'])
        cum_time += dur

    raw.set_annotations(mne.Annotations(
        onset=onset_list,
        duration=duration_list,
        description=desc_list,
        orig_time=None
    ))

    print(f"\nSUCCESS: Raw created")
    print(f"   Duration    : {raw.times[-1]:.1f}s")
    print(f"   Channels    : {raw.info['nchan']}")
    print(f"   Annotations : {len(desc_list)} (start/end for {len(segments)} blocks)")

    return raw
















def split_music_vs_pinknoise(raw_segs_corr, montage="biosemi128"):
    """
    Split a Raw object into music and pink-noise segments based on boundary annotations.
    Preserves and correctly places original annotations in the new concatenated raws.

    Returns
    -------
    raw_music : mne.io.Raw | None
    raw_pink : mne.io.Raw | None
    """
    raw = raw_segs_corr.copy()
    all_anns = raw.annotations

    # Filter valid boundary annotations
    boundary_anns = [
        a for a in all_anns
        if "BAD" not in a["description"] and "EDGE" not in a["description"] and "event" not in a["description"]]

    if len(boundary_anns) == 0:
        raise RuntimeError("No valid boundary annotations found.")

    music_segments = []
    pink_segments = []

    t_start = 0.0

    music_last_time = 0
    pinknoise_last_time = 0

    onset_music_list = []
    onset_pink_list = []

    duration_music_list = []
    duration_pink_list = []

    description_music_list = []
    description_pink_list = []
    
    last_ann = len(boundary_anns)
    count_ann = 0

    for a in boundary_anns:
        t_end = a["onset"]
        desc = a["description"]

        if t_end <= t_start:
            continue

        print(f"Segment: {desc}, {t_start:.2f}s to {t_end:.2f}s")

        # Crop the segment
        seg = raw.copy().crop(tmin=t_start, tmax=t_end, include_tmax=False)

        # Assign to correct list
        if "pink_noise" in desc.lower():
            print("  -> Pink Noise Segment")
            pink_segments.append(seg)

            onset_pink_list.append(pinknoise_last_time)
            duration_pink_list.append(a['duration'])
            description_pink_list.append(desc)

            pinknoise_last_time += seg.get_data().shape[1] / seg.info['sfreq']

        else:
            print("  -> Music Segment")
            music_segments.append(seg)
            onset_music_list.append(music_last_time)
            duration_music_list.append(a['duration'])
            description_music_list.append(desc)

            music_last_time += seg.get_data().shape[1] / seg.info['sfreq']
        
        count_ann += 1

        if count_ann == last_ann:
            # Handle last segment, from last annotation to end of raw (which is assumed to be a Noisy Segment)
            t_start =  a["onset"]
            t_end = raw.times[-1]

            if t_end > t_start:
                print(f"Segment: END, {t_start:.2f}s to {t_end:.2f}s")
                seg = raw.copy().crop(tmin=t_start, tmax=t_end, include_tmax=True)

                print("  -> Last Pink Noise Segment")
                pink_segments.append(seg)

                onset_pink_list.append(pinknoise_last_time)
                duration_pink_list.append(a['duration'])
                description_pink_list.append("END_pink_noise")
                # pinknoise_last_time += seg.get_data().shape[1] / seg.info['sfreq']

        t_start = t_end

    info = copy.deepcopy(raw.info)

    music_annots = mne.annotations.Annotations(
        onset=onset_music_list,
        duration=duration_music_list,
        description=description_music_list)
    
    music_raw = mne.concatenate_raws(music_segments)
    music_raw.info = info
    music_raw.set_montage(montage)
    music_raw.set_annotations(music_annots)

    pink_annots = mne.annotations.Annotations(
        onset=onset_pink_list,
        duration=duration_pink_list,
        description=description_pink_list)
    
    info = copy.deepcopy(raw.info)

    pink_raw = mne.concatenate_raws(pink_segments)
    pink_raw.info = info
    pink_raw.set_montage(montage)
    pink_raw.set_annotations(pink_annots)

    return music_raw, pink_raw









def trim_raw_seconds(raw, seconds, from_end=True):
    """
    Remove a given number of seconds from the start or end of an MNE Raw object.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw object to trim.
    seconds : float
        Number of seconds to remove.
    from_end : bool (default=True)
        If True → remove from end of recording.
        If False → remove from start.

    Returns
    -------
    raw_trimmed : mne.io.Raw
        A **new** Raw object with the trimmed duration.
    """

    # Get original time boundaries
    tmin = raw.times[0]          # usually 0
    tmax = raw.times[-1]         # last time point

    if from_end:
        # remove seconds from end
        new_tmax = tmax - seconds
        if new_tmax <= tmin:
            raise ValueError("Cannot trim more seconds than recording duration.")
        raw_trimmed = raw.copy().crop(tmin=tmin, tmax=new_tmax)

    else:
        # remove seconds from start
        new_tmin = tmin + seconds
        if new_tmin >= tmax:
            raise ValueError("Cannot trim more seconds than recording duration.")
        raw_trimmed = raw.copy().crop(tmin=new_tmin, tmax=tmax)

    return raw_trimmed











def trim_segments(segments_raw, trim_sec=1.0, montage="biosemi128"):

    segments = segments_raw.copy()
    all_anns = segments.annotations
    onset_list = []
    duration_list = []
    description_list = []
    segment_list = []

    last_ann = len(all_anns)
    count_ann = 0
    current_annot = all_anns[0]
    prev_annot = current_annot
    segments_length = 0
    offset = segments.first_time # Sometimes the offset can be different than 0
    if offset < 0:
        print("Offset is negative")
        return 

    for ann in all_anns:
        count_ann += 1
        current_annot = ann
        print(f"Processing annotation {count_ann}/{last_ann}: {current_annot['description']}")

        if current_annot != prev_annot:
            t_start = prev_annot["onset"] - offset
            t_end = current_annot["onset"] - offset
            desc = prev_annot["description"]
            dur = prev_annot["duration"]

            seg = segments.copy().crop(tmin=t_start, tmax=t_end, include_tmax=False)
            seg_trim_start = trim_raw_seconds(seg, trim_sec, from_end=False)
            seg_trim_end = trim_raw_seconds(seg_trim_start, trim_sec, from_end=True)

            seg_data = seg_trim_end.get_data()
            segment_list.append(seg_data)
            onset_list.append(segments_length)
            description_list.append(desc)
            duration_list.append(dur)

            segments_length += seg_trim_end.get_data().shape[1] / seg.info['sfreq']

        if count_ann == last_ann:
            t_start = current_annot["onset"] - offset
            t_end = segments.times[-1]
            desc = current_annot["description"]
            dur = current_annot["duration"]
            print("Processing last annotation segment...", t_start, t_end)

            seg = segments.copy().crop(tmin=t_start, tmax=t_end, include_tmax=True)
            seg_trim_start = trim_raw_seconds(seg, trim_sec, from_end=False)
            seg_trim_end = trim_raw_seconds(seg_trim_start, trim_sec, from_end=True)
            seg_data = seg_trim_end.get_data()
            segment_list.append(seg_data)
            onset_list.append(segments_length)
            description_list.append(desc)
            duration_list.append(dur)

        prev_annot = current_annot

    n_channels = np.shape(segment_list[0])[0]
    ch_names = mne.channels.make_standard_montage(montage).ch_names[:n_channels]

    print("Number of channels after trimming: ", n_channels)
    print("Data type: ", type(segment_list[0]))
    sfreq = segments.info['sfreq']
    ch_types = ["eeg"] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    seg_annots = mne.annotations.Annotations(
        onset=onset_list,
        duration=duration_list,
        description=description_list)
    
    # seg_raw = mne.concatenate_raws(segment_list)
    #seg_raw.info = info
    all_data = np.concatenate(segment_list, axis=1)
    print("All data shape after trimming: ", all_data.shape)
    seg_raw = mne.io.RawArray(all_data, info)
    seg_raw.set_montage(montage)
    seg_raw.set_annotations(seg_annots)
    
    return seg_raw








def create_epochs_for_source(raw_eeg, epoch_length=1, overlap_fraction=0):
    raw_epochs = mne.make_fixed_length_epochs(
    raw_eeg,
    duration=epoch_length,               # epoch length in seconds
    preload=True,
    overlap=epoch_length * overlap_fraction,                # no overlap
    reject_by_annotation=False,
    verbose=True
    )

    print(f"Created {len(raw_epochs)} epochs of {raw_epochs.times[-1]:.1f} s each")

    return raw_epochs




def create_epochs(raw_eeg, plot=False):
    sfreq = raw_eeg.info['sfreq']
    duration_sec = raw_eeg.n_times / sfreq

    event_interval = 5

    # Maximum start time so that even the last epoch fits completely
    max_start_sec = duration_sec - event_interval

    # Generate start times
    start_times_sec = np.arange(0, max_start_sec + 1e-6, event_interval)  # +epsilon to include boundary

    # Convert to samples (round down to avoid going over)
    event_samples = np.floor(start_times_sec * sfreq).astype(int)

    # Build events
    events = np.column_stack((
        event_samples,
        np.zeros(len(event_samples), dtype=int),
        np.ones(len(event_samples), dtype=int)
    ))

    print(f"Created {len(events)} safe epochs (last starts at {event_samples[-1]/sfreq:.1f} s)")

    # Create epochs
    raw_epochs = mne.Epochs(
    raw_eeg,
    events=events,
    event_id={'segment': 1},
    tmin=0,# With 0 the epochs become non overlapping and the duration is indicated by tmax.
    tmax=8.0,
    reject_by_annotation=False,
    preload=True,          # can stay False
    baseline=None,
    reject=None,
    flat=None,
    reject_tmin=None,
    reject_tmax=None
    )

    if plot:
        raw_epochs.plot(n_channels=128, scalings='auto', butterfly=0)
    return raw_epochs







def crop_to_fixed_length(raw, duration, random_segment=False, start_time=None):
    """
    Crop Raw EEG data to a fixed duration with flexible start options.

    Parameters
    ----------
    raw : mne.io.Raw
        Input Raw EEG data
    duration : float
        Desired duration in seconds (must be ≤ total length)
    random_segment : bool, default False
        If True → pick a random continuous segment of `duration` seconds
    start_time : float or None, default None
        Exact start time in seconds. Overrides random_segment if provided.

    Returns
    -------
    cropped_raw : mne.io.Raw
        Cropped Raw object with exactly `duration` seconds

    Raises
    ------
    ValueError
        If duration is too long or start_time is invalid
    """

    # Total duration of the recording (in seconds)
    total_duration = raw.times[-1] + 1/raw.info['sfreq']  # precise

    if duration > total_duration:
        raise ValueError(f"Requested duration ({duration:.2f}s) > data length ({total_duration:.2f}s)")

    # ——————————————————————————————————————————————————————————————
    # 1. User explicitly gives start time → highest priority
    # ——————————————————————————————————————————————————————————————
    if start_time is not None:
        if start_time < 0:
            raise ValueError("start_time cannot be negative")
        if start_time + duration > total_duration:
            raise ValueError(f"start_time + duration ({start_time + duration:.2f}s) "
                             f"exceeds data end ({total_duration:.2f}s)")

        tmin = start_time
        tmax = start_time + duration
        mode = f"fixed start ({tmin:.2f}s)"

    # ——————————————————————————————————————————————————————————————
    # 2. Random segment requested
    # ——————————————————————————————————————————————————————————————
    elif random_segment:
        max_possible_start = total_duration - duration
        tmin = np.random.uniform(0, max_possible_start)
        tmax = tmin + duration
        mode = f"random segment ({tmin:.2f}s)"

    # ——————————————————————————————————————————————————————————————
    # 3. Default: crop from beginning
    # ——————————————————————————————————————————————————————————————
    else:
        tmin = 0.0
        tmax = duration
        mode = "from beginning"

    # ——————————————————————————————————————————————————————————————
    # Perform the crop
    # ——————————————————————————————————————————————————————————————
    cropped_raw = raw.copy().crop(tmin=tmin, tmax=tmax, include_tmax=False)

    actual_duration = cropped_raw.times[-1] - cropped_raw.times[0]
    print(f"Cropped → {mode} → {tmin:.2f}s to {tmax:.2f}s "
          f"(duration = {actual_duration:.3f}s)")

    return cropped_raw



def remove_all_annotations(raw):
    """
    Remove all annotations from an MNE Raw object.

    Args:
        raw (mne.io.Raw): Raw object with annotations.

    Returns:
        mne.io.Raw: Same Raw object with annotations removed.
    """
    raw.set_annotations(mne.Annotations([], [], []))
    return raw