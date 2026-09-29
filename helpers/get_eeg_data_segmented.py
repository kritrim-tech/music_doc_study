import mne

def reset_annotations(raw, paradigm):
    onset_list = []
    duration_list = []
    description_list = []

    duration = 0.5
    
    onset_start = 0.0
    onset_end = raw.times[-1]

    onset_list = [onset_start, onset_end]
    duration_list = [duration, duration]

    description_start = paradigm + "_start"
    description_end = paradigm + "_end"
    description_list = [description_start, description_end]

    annots = mne.annotations.Annotations(
        onset=onset_list,
        duration=duration_list,
        description=description_list)

    
    return annots


def get_segmented_data(data_cleaned_all, subs):
    segmented_data_all = {sub: {} for sub in subs}

    for sub in subs:    
        t_start = 0
        t_end = 0

        # t_start_segs = 0
        # t_end_segs = 0

        count_ann = 0
        all_annots = data_cleaned_all[sub].annotations

        offset = data_cleaned_all[sub].first_time

        if offset < 0:
            raise ValueError("Offset is negative!")

        segmented_data_all[sub]["pink_segs"] = []
        segmented_data_all[sub]["music_segs"] = []

        for ann in all_annots:
            t_end = ann["onset"] - offset
            seg = data_cleaned_all[sub].copy().crop(tmin=t_start, tmax=t_end, include_tmax=False)
            paradigm_name = ann["description"]
            annots_reset = reset_annotations(seg, paradigm_name)
            seg.set_annotations(annots_reset)

            if "end" not in paradigm_name:
                if "pink" in paradigm_name:
                    segmented_data_all[sub]["pink_segs"].append(seg)
                
                else:
                    segmented_data_all[sub]["music_segs"].append(seg)

            else:
                paradigm_name_split = paradigm_name.split("_")
                paradigm_name_shortened = paradigm_name_split[0] + "_" + paradigm_name_split[1]
                segmented_data_all[sub][paradigm_name_shortened] = seg


            # if count_ann >= len(all_annots):
            #     seg_segs = data_cleaned_all.copy().crop(tmin=t_start_segs, tmax=t_end_segs, include_tmax=False)
            #     segmented_data_all[sub]["segs"] = seg_segs

            count_ann += 1
            t_start = t_end
        
        segmented_data_all[sub]["pink_segs"] = mne.concatenate_raws(segmented_data_all[sub]["pink_segs"])
        segmented_data_all[sub]["music_segs"] = mne.concatenate_raws(segmented_data_all[sub]["music_segs"])

    return segmented_data_all




def clean_segments(segmented_data_all):
    segmented_data_all_annots_cleaned = {}

    for sub in segmented_data_all:
        segmented_data_all_annots_cleaned[sub] = {}

        for paradigm in segmented_data_all[sub]:
            onset_list = []
            description_list = []
            duration_list = []

            raw = segmented_data_all[sub][paradigm]
            raw_offset = raw.first_time

            if raw_offset < 0:
                raise ValueError("Offset is negative!")
            
            annots = raw.annotations

            for ann in annots:
                if "BAD" not in ann["description"] and "EDGE" not in ann["description"]:
                        onset_list.append(ann["onset"] - raw_offset)
                        description_list.append(ann["description"])
                        duration_list.append(ann["duration"])

            
            segmented_data_all_annots_cleaned[sub][paradigm] = raw.copy().set_annotations(
                                                                        mne.Annotations(onset=onset_list, 
                                                                        duration=duration_list, description=description_list))
            
    return segmented_data_all_annots_cleaned

