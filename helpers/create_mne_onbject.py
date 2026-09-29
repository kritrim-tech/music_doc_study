import mne
import copy

def create_mne_raw_with_start_time_restarted(raw_obj, montage="biosemi128", channel_n=128):
    raw = raw_obj.copy()
    # sfreq = raw.info['sfreq']
    info = copy.deepcopy(raw.info)

    montage_obj = mne.channels.make_standard_montage(montage)
    # ch_names = montage_obj.ch_names[:channel_n]  # Use exact montage names
    # info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

    raw_data = raw.get_data()

    raw_new = mne.io.RawArray(raw_data, info) 
    raw_new.set_montage(montage_obj)
    
    offset = raw.first_time

    onset_list = []
    duration_list = []
    desc_list = []
    # cum_time = 0.0

    for ann in raw.annotations:
        onset_list.append(ann["onset"] - offset)
        duration_list.append(ann["duration"])
        desc_list.append(ann["description"])

    raw_new.set_annotations(mne.Annotations(
        onset=onset_list,
        duration=duration_list,
        description=desc_list,
        orig_time=None
    ))

    return raw_new