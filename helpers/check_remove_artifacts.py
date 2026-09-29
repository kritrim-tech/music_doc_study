import matplotlib.pyplot as plt
import mne



def remove_bad_segments(raw):
    """
    Remove segments marked as BAD from raw data and return corrected raw.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw data object containing BAD annotations
        
    Returns
    -------
    corr_raw : mne.io.Raw
        The corrected raw data with BAD segments removed
        Returns None if all data is marked as BAD
    """
    # Extract sampling frequency for time calculations
    sfreq = raw.info['sfreq']
    total_duration = raw.times[-1]

    # Get all BAD annotations only
    bad_annots = [annot for annot in raw.annotations 
                 if annot['description'].startswith('BAD')]
    bad_annots = sorted(bad_annots, key=lambda x: x['onset'])

    # Prepare list to store good segments
    good_segments = []
    current_time = 0.0  # Start from beginning
    offset = raw.first_time #Added during the rest_mid of Sub 7 Preprocessing

    if offset < 0:
        print("Offset Below 0")
        return 
    # Iterate through each bad segment
    for bad in bad_annots:
        onset = bad['onset'] - offset
        duration = bad['duration']
        
        # Add good data segment before this bad segment
        if onset > current_time:
            good_seg = raw.copy().crop(tmin=current_time, tmax=onset, include_tmax=True) #Added include_tmax=True for p7 raw segs
            good_segments.append(good_seg)
        
        # Update current_time to after bad segment
        current_time = max(current_time, onset + duration)

    # Add remaining good data after last bad segment
    if current_time < total_duration:
        good_seg = raw.copy().crop(tmin=current_time, tmax=total_duration)
        good_segments.append(good_seg)

    # Concatenate all good segments to create cleaned raw
    if len(good_segments) > 0:
        corr_raw = mne.concatenate_raws(good_segments)
    else:
        corr_raw = None
        print("No good segments found; all data marked as BAD.")

    # Print length comparison for sanity check
    print(f"Original duration: {total_duration:.2f} sec")
    if corr_raw:
        print(f"Corrected duration: {corr_raw.times[-1]:.2f} sec")
    
    return corr_raw