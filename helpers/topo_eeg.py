import numpy as np
import matplotlib.pyplot as plt
from mne.viz import plot_topomap
from statsmodels.stats.multitest import multipletests
from matplotlib.colors import ListedColormap
import mne
from scipy.stats import ttest_rel
from matplotlib.colorbar import ColorbarBase
from pathlib import Path
from collections import defaultdict







def get_fdr_p_values(p_vals_all, p_vals_per_band=128):
    fdr_corrected_p_vals_bands = {}

    for sub in p_vals_all:
        # Flatten this subject's p-values across all bands only
        p_vals_sub = []
        for band in p_vals_all[sub]:
            for p_val in p_vals_all[sub][band]:
                p_vals_sub.append(p_val)

        # FDR correction performed independently for this subject
        fdr_corrected = multipletests(
            p_vals_sub,
            alpha=0.05,
            method='fdr_bh',
            is_sorted=False,
            returnsorted=False
        )
        fdr_corrected_p_vals = fdr_corrected[1]

        fdr_corrected_p_vals_bands[sub] = {}
        count = 0
        for band in p_vals_all[sub]:
            fdr_corrected_p_vals_bands[sub][band] = fdr_corrected_p_vals[count:count + p_vals_per_band]
            count += p_vals_per_band

    return fdr_corrected_p_vals_bands












def plot_topomap_(band_powers_all, raw_info, pat_state, bands_range, save_path, save_path_cbar=None, show_bands=True,  title=None, 
                  cmap=ListedColormap(["#7DB17B", "#EDF4F6"]), power_units_heading="Power Diference"):
    """
    Plot EEG band power difference (passive - rest) for all subjects in a single figure.

    Parameters:
    - band_powers_all (dict): 
    - eeg_data_all (dict): Dictionary containing EEG data for all subjects.
    - title (str): Title for the entire figure (default: None).
    - cmap (str): Colormap for visualization (default: "viridis").
    """
    subjects = list(band_powers_all.keys())  # Extract subject names
    subjects.sort()  # Sort subjects alphabetically
    band_names = list(band_powers_all[subjects[0]].keys())  # Extract band names
    print(band_names)  # <-- check this output to confirm exact key spelling

    # Exclude high-gamma sub-band (adjust key to match your actual data)
    band_names = [b for b in band_names if b != 'gamma high']

    # Display-name mapping, keyed by the REAL band name
    # greek_symbols = {
    #     'delta': r'$\delta$',
    #     'theta': r'$\theta$',
    #     'alpha': r'$\alpha$',
    #     'beta': r'$\beta$',
    #     'gamma low': r'$\gamma$',
    # }
    # band_display_names = {b: greek_symbols.get(b, b.upper()) for b in band_names}

    band_display_names = {
    'delta': r'$\mathbf{\delta}$',
    'theta': r'$\mathbf{\theta}$',
    'alpha': r'$\mathbf{\alpha}$',
    'beta':  r'$\mathbf{\beta}$',
    'gamma low': r'$\mathbf{\gamma}$',
    }

    print(band_display_names)
    
    # Create subplots (subjects on x-axis, bands on y-axis)
    fig, axes = plt.subplots(len(band_names), len(subjects), figsize=(4 * len(subjects), 3 * len(band_names)))
    vmin, vmax = 0, 1
    # Ensure axes is always 2D (even for single subject/band)
    if len(subjects) == 1:
        axes = np.expand_dims(axes, axis=1)  # Ensure shape (bands, subjects)
    
    if len(band_names) == 1:
        axes = np.expand_dims(axes, axis=0)

    # Add band labels on the left side (y-axis)
    if show_bands:
        for band_idx, band in enumerate(band_names):
            axes[band_idx, 0].set_ylabel(f"{band_display_names[band]}", 
                                         fontsize=24, rotation=0, ha="right", va="center", fontweight='bold')


    # Add subject labels on the top (x-axis)
    for subj_idx, subject in enumerate(subjects):
        sub_num = list(pat_state.keys()).index(subject) + 1 if pat_state and subject in pat_state else subj_idx + 1
        axes[0, subj_idx].set_title(f"Sub{sub_num} ({pat_state[subject]})",
                                    fontsize=20, pad=10, fontweight='bold')

    for subj_idx, subject in enumerate(subjects):
        # Select the appropriate Raw object for electrode positions
        # for prd in band_powers_all[subject]:
        #     prd_name = prd
            
        # raw_object_for_topomap = eeg_data_all[subject][prd_name]  # Use the first chunk: could be any
        band_powers = band_powers_all[subject]

        for band_idx, band in enumerate(band_names):

            p_values = band_powers[band]
            # fdr_corrected = multipletests(
            #                     p_values, 
            #                     alpha=0.05, 
            #                     method='fdr_bh',           # Benjamini-Hochberg (most common FDR method)
            #                     is_sorted=False,           # usually leave as False unless p_values are already sorted
            #                     returnsorted=False
            #                 )
            
            # Extract FDR-adjusted p-values
            # p_values_fdr = fdr_corrected[1]
            # significant_mask = p_values_fdr < 0.05


            # significant_mask = p_values < 0.05
            # p_values[significant_mask] = 0
            # p_values[p_values != 0] = 1
            p_plot = np.where(np.asarray(p_values, float) < 0.05, 0.0, 1.0)

            ax = axes[band_idx, subj_idx]
            im, _ = mne.viz.plot_topomap(
                p_plot, 
                raw_info, 
                axes=ax,
                sensors=True, 
                cmap=cmap, 
                show=False,
                vlim=(0, 1),
                contours=0,                       # remove contour lines
                extrapolate='local',              # or 'box'
                # interpolation='nearest'
            )


    # This is the most effective way to prevent overlap:
    plt.tight_layout(rect=[0, 0, 1, 0.88])   # Leaves ~12% space at the top for suptitle

    # Fine-tune if needed:
    # plt.subplots_adjust(top=0.85, hspace=0.3, wspace=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


    if save_path_cbar is not None:
        import matplotlib.patches as mpatches

        fig_cbar = plt.figure(figsize=(3, 1.5))
        ax_cbar = fig_cbar.add_axes([0, 0, 1, 1])
        ax_cbar.axis('off')

        # Get the two discrete colors from the colormap (index 0 = significant, index 1 = non-significant)
        color_significant = cmap(0.0)
        color_nonsignificant = cmap(1.0)

        legend_patches = [
            mpatches.Patch(facecolor=color_significant, edgecolor='black', label='Significant (p < 0.05)'),
            mpatches.Patch(facecolor=color_nonsignificant, edgecolor='black', label='Non-significant'),
        ]
        ax_cbar.legend(
            handles=legend_patches,
            loc='center',
            frameon=False,
            fontsize=16,
            ncol=2,
            handlelength=1.5,
            handleheight=1.5,
        )

        plt.savefig(save_path_cbar, dpi=300, bbox_inches='tight')
        plt.show()
    











def plot_topomap_opt(band_powers_all, raw_info, pat_state,
                   save_path=None, save_dir=None, filename_template="topomap_{band}.pdf",
                   save_path_cbar=None, show_bands=True, title=None,
                   cmap=ListedColormap(['black', 'white']), power_units_heading="Power Diference",
                   bands_to_plot=None, subjects_to_plot=None):
    """
    Usage modes
    -----------
    1) All bands, all subjects, ONE combined figure (original behavior):
        bands_to_plot=None, save_path="all.pdf"
    2) ONE band, all subjects, ONE figure:
        bands_to_plot=["delta"], save_path="delta_only.pdf"
    3) All bands, ONE subject, ONE combined figure:
        bands_to_plot=None, subjects_to_plot=["sub01"], save_path="sub01_only.pdf"
    4) MULTIPLE specific bands -> auto-split into SEPARATE files, one per band:
        bands_to_plot=["delta", "theta", "alpha"],
        save_dir="output_folder/", filename_template="rest_music_{band}.pdf"
    """

    def _plot_topomap_core(band_powers_all, raw_info, pat_state, subjects, band_names,
                        save_path, save_path_cbar, show_bands, cmap, power_units_heading,
                        is_first_band=True):
        """Original single-figure plotting logic, now parameterized by an
        already-resolved subject/band list rather than doing its own filtering."""

        greek_symbols = {
            'delta': r'$\delta$',
            'theta': r'$\theta$',
            'alpha': r'$\alpha$',
            'beta': r'$\beta$',
            'gamma low': r'$\gamma$',
        }
        band_display_names = {b: greek_symbols.get(b, b.upper()) for b in band_names}

        fig, axes = plt.subplots(len(band_names), len(subjects), figsize=(4 * len(subjects), 3 * len(band_names)))
        vmin, vmax = 0, 1
        if len(subjects) == 1:
            axes = np.expand_dims(axes, axis=1)
        if len(band_names) == 1:
            axes = np.expand_dims(axes, axis=0)

        if show_bands:
            for band_idx, band in enumerate(band_names):
                axes[band_idx, 0].set_ylabel(f"{band_display_names[band]}",
                                            fontsize=24, rotation=0, ha="right", va="center", fontweight='bold')

        if is_first_band:
            start_offset = list(pat_state.keys()).index(subjects[0])
            for subj_idx, subject in enumerate(subjects):
                axes[0, subj_idx].set_title(f"Sub{start_offset + subj_idx + 1} ({pat_state[subject]})", fontsize=20, pad=10, fontweight='bold')

        for subj_idx, subject in enumerate(subjects):
            band_powers = band_powers_all[subject]

            for band_idx, band in enumerate(band_names):
                p_values = band_powers[band]
                significant_mask = p_values < 0.05
                p_values[significant_mask] = 0
                p_values[p_values != 0] = 1

                ax = axes[band_idx, subj_idx]
                im, _ = mne.viz.plot_topomap(
                    p_values, raw_info, axes=ax, sensors=True, cmap=cmap,
                    show=False, vlim=(0, 1), contours=0, extrapolate='local',
                )

        plt.tight_layout(rect=[0, 0, 1, 0.88])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.show()
        plt.close(fig)

        if save_path_cbar is not None:
            import matplotlib.patches as mpatches

            fig_cbar = plt.figure(figsize=(3, 1.5))
            ax_cbar = fig_cbar.add_axes([0, 0, 1, 1])
            ax_cbar.axis('off')

            color_significant = cmap(0.0)
            color_nonsignificant = cmap(1.0)

            legend_patches = [
                mpatches.Patch(facecolor=color_significant, edgecolor='black', label='Significant (p < 0.05)'),
                mpatches.Patch(facecolor=color_nonsignificant, edgecolor='black', label='Non-significant'),
            ]
            ax_cbar.legend(handles=legend_patches, loc='center', frameon=False,
                        fontsize=16, handlelength=1.5, handleheight=1.5)

            plt.savefig(save_path_cbar, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close(fig_cbar)


    import os

    subjects = list(band_powers_all.keys())
    subjects.sort()

    if subjects_to_plot is not None:
        missing = [s for s in subjects_to_plot if s not in subjects]
        if missing:
            print(f"Warning: requested subjects not found and skipped: {missing}")
        subjects = [s for s in subjects_to_plot if s in subjects]
    if not subjects:
        raise ValueError("No subjects to plot after filtering.")

    band_names = list(band_powers_all[subjects[0]].keys())
    band_names = [b for b in band_names if b != 'gamma high']

    if bands_to_plot is not None:
        missing = [b for b in bands_to_plot if b not in band_names]
        if missing:
            print(f"Warning: requested bands not found and skipped: {missing}")
        band_names = [b for b in bands_to_plot if b in band_names]
    if not band_names:
        raise ValueError("No bands to plot after filtering.")

    print(band_names)

    # ── auto-split mode: multiple explicit bands -> one file per band ──────
    if bands_to_plot is not None and len(band_names) > 1:
        if save_dir is None:
            raise ValueError("save_dir must be provided when bands_to_plot has more than one band.")
        os.makedirs(save_dir, exist_ok=True)
        for band in band_names:
            band_save_path = os.path.join(save_dir, filename_template.format(band=band))
            _plot_topomap_core(band_powers_all, raw_info, pat_state, subjects, [band],
                               band_save_path, save_path_cbar, show_bands, cmap, power_units_heading,
                               is_first_band=(band == band_names[0]))
        return

    # ── single combined figure (modes 1-3) ──────────────────────────────────
    _plot_topomap_core(band_powers_all, raw_info, pat_state, subjects, band_names,
                       save_path, save_path_cbar, show_bands, cmap, power_units_heading,
                       is_first_band=True)




def plot_band_power_difference_all_subjects(band_powers_all, raw_info, pat_state, bands_range, save_path, save_path_cbar=None, show_all_cbars=False,
                                            show_bands=True, title=None, cmap="viridis", power_units_heading="Power Diference"):

    subjects = list(band_powers_all.keys())  # Extract subject names
    subjects.sort()  # Sort subjects alphabetically
    paradigm_names =  list(band_powers_all[subjects[0]].keys())
    bands = list(next(iter(band_powers_all.values()))[paradigm_names[0]].keys())  # Extract band names


    # Exclude high-gamma sub-band (adjust key to match your actual data)
    bands = [b for b in bands if b != 'gamma high']

    # Display-name mapping, keyed by the REAL band name
    greek_symbols = {
        'delta': r'$\delta$',
        'theta': r'$\theta$',
        'alpha': r'$\alpha$',
        'beta': r'$\beta$',
        'gamma low': r'$\gamma$',
    }

    band_display_names = {b: greek_symbols.get(b, b.upper()) for b in bands}

    band_display_names = {
    'delta': r'$\mathbf{\delta}$',
    'theta': r'$\mathbf{\theta}$',
    'alpha': r'$\mathbf{\alpha}$',
    'beta':  r'$\mathbf{\beta}$',
    'gamma low': r'$\mathbf{\gamma}$',
    }
    print("Band Display Names: ", band_display_names)

    # Create subplots (subjects on x-axis, bands on y-axis)
    fig, axes = plt.subplots(len(bands), len(subjects), figsize=(4 * len(subjects), 3 * len(bands)))


    # --- 1. Compute global vmin and vmax across all band_power_difference values ---
    all_diffs = []
    for subject in subjects:
        for band in bands:
            print("Using Following Paradigm Names and in the following order: ", paradigm_names)  

            passive_psd = np.array(band_powers_all[subject][paradigm_names[1]][band])
            rest_psd = np.array(band_powers_all[subject][paradigm_names[0]][band])
            diff = passive_psd - rest_psd
            all_diffs.append(diff)

    all_diffs = np.concatenate(all_diffs)
    vmin, vmax = np.min(all_diffs), np.max(all_diffs)


    # Ensure axes is always 2D (even for single subject/band)
    if len(subjects) == 1:
        axes = np.expand_dims(axes, axis=1)  # Ensure shape (bands, subjects)
    if len(bands) == 1:
        axes = np.expand_dims(axes, axis=0)

    # Add band labels on the left side (y-axis)
    if show_bands:
        for band_idx, band in enumerate(bands):
            axes[band_idx, 0].set_ylabel(f"{band_display_names[band]}", fontsize=24, rotation=0, ha="right", va="center")

    
    for subj_idx, subject in enumerate(subjects):
        sub_num = list(pat_state.keys()).index(subject) + 1 if pat_state and subject in pat_state else subj_idx + 1
        axes[0, subj_idx].set_title(f"Sub{sub_num} ({pat_state[subject]})",
                                    fontsize=20, pad=10, fontweight='bold')

    for subj_idx, subject in enumerate(subjects):
        band_power_difference = {} 

        p_values_band = {}

        for band in bands:
            passive_psd = np.array(band_powers_all[subject][paradigm_names[1]][band])
            rest_psd = np.array(band_powers_all[subject][paradigm_names[0]][band])
            band_power_difference[band] = (passive_psd - rest_psd) / rest_psd
            t_stat, p_value = ttest_rel(passive_psd, rest_psd)
            p_values_band[band] = p_value

        for band_idx, band in enumerate(bands):
            ax = axes[band_idx, subj_idx]
            band_p_value = p_values_band[band]

            
            im, _ = mne.viz.plot_topomap(
                band_power_difference[band], 
                raw_info, 
                axes=ax, 
                cmap=cmap, 
                show=False
            )

            # im.set_clim(vmin, vmax)

            if show_all_cbars or subj_idx == len(subjects) - 1:
                cbar = plt.colorbar(im, ax=ax, shrink=0.7, orientation="vertical", pad=0.05)
                # cbar.set_label(power_units_heading)
                cbar.ax.tick_params(labelsize=18)



    # This is the most effective way to prevent overlap:
    plt.tight_layout(rect=[0, 0, 1, 0.88])   # Leaves ~12% space at the top for suptitle

    # Fine-tune if needed:
    # plt.subplots_adjust(top=0.85, hspace=0.3, wspace=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()















def compare_paired_arrays(array1, array2, alpha=0.05, test_normality=False, normality_test='shapiro'):
    from scipy import stats

    """
    Compare two paired 1D arrays:
      - If test_normality=True: check normality first, then choose test
            → paired t-test (ttest_rel) if both normal
            → Wilcoxon signed-rank test otherwise
      - If test_normality=False: skip the normality check and always use
            the Wilcoxon signed-rank test (non-parametric, paired)

    Returns:
        test_name, statistic, p_value, both_normal
    """
    mask = ~np.isnan(array1) & ~np.isnan(array2)
    x = array1[mask]
    y = array2[mask]

    if len(x) < 3:
        return "not_enough_data", np.nan, np.nan, False

    if not test_normality:
        # Skip normality testing, go straight to the non-parametric test
        statistic, p_val = stats.wilcoxon(x, y, alternative='two-sided')
        return "Wilcoxon signed-rank", statistic, p_val, False

    # Normality test (Shapiro-Wilk is most common)
    if normality_test == 'shapiro':
        _, p1 = stats.shapiro(x)
        _, p2 = stats.shapiro(y)
    elif normality_test == 'normaltest':
        _, p1 = stats.normaltest(x)
        _, p2 = stats.normaltest(y)
    else:
        raise ValueError("Unsupported normality test")

    both_normal = (p1 > alpha) and (p2 > alpha)

    if both_normal:
        t_stat, p_val = stats.ttest_rel(x, y, alternative='two-sided')
        test_name = "paired t-test"
        statistic = t_stat
    else:
        statistic, p_val = stats.wilcoxon(x, y, alternative='two-sided')
        test_name = "Wilcoxon signed-rank"

    return test_name, statistic, p_val, both_normal









def get_channel_wise_fooof(freqs_psd, eeg_fooof, bands, channels_n=128):
    fooof_bands = {}
    fooof_bands_pvalues_rel = {}
    fooof_bands_pvalues_abs = {}
    channels = channels_n

    for pt in eeg_fooof:
        fooof_bands[pt] = {}
        fooof_bands_pvalues_rel[pt] = {}
        fooof_bands_pvalues_abs[pt] = {}
        
        paradigm_names = list(eeg_fooof[pt].keys())
        
        print(pt)
        print("Paradigm 0", paradigm_names[0])
        print("Paradigm 1", paradigm_names[1])
        
        for parad_indx in eeg_fooof[pt]:
            # parad_name = paradigms[parad_indx]
            fooof_bands[pt][parad_indx] = {}
            fooof_periodic_parad = eeg_fooof[pt][parad_indx][2]
            # freqs = eeg_fooof[pt][parad_indx][1]
            # freqs = freqs_psd#eeg_psd[pt][parad_indx][1]
            freqs = eeg_fooof[pt][parad_indx][3]
            # psds = eeg_psds[pt][parad_indx][0]
            # psds = eeg_fooof[pt][parad_indx][0]
            # fooof_band_powers_absolute = {band: np.zeros((psds.shape[0], psds.shape[2])) for band in bands}
            
            fooof_band_powers_absolute = {band: np.zeros((fooof_periodic_parad.shape[0], fooof_periodic_parad.shape[2])) for band in bands}
            # print(freqs)

            for band, (fmin, fmax) in bands.items():
                band_idx = np.where((freqs >= fmin) & (freqs < fmax))[0] #Min freq included and the maxmum freq excluded
                # fooof_bands[pt][parad_indx][band] = fooof_periodic_parad[:, band_idx, :]
                for seg in range(fooof_periodic_parad.shape[2]):
                    # print(fooof_periodic_parad[:, band_idx, seg].shape)
                    fooof_band_powers_absolute[band][:, seg] = np.trapezoid(fooof_periodic_parad[:, band_idx, seg], freqs[band_idx], axis=1)

            fooof_total_power_bands = np.sum(np.array(list(fooof_band_powers_absolute.values())), axis=0)  # np.sum is applied on an array of size (5 bands, 128 channels, 159 segments)
            fooof_band_powers_relative = {band: fooof_band_powers_absolute[band] / fooof_total_power_bands for band in fooof_band_powers_absolute} 
            fooof_bands[pt][parad_indx] = [fooof_band_powers_absolute, fooof_band_powers_relative]

        for band, (fmin, fmax) in bands.items():
            fooof_bands_pvalues_rel[pt][band] = []
            fooof_bands_pvalues_abs[pt][band] = []

            for ch in range(channels):
                paradigm_0_abs = fooof_bands[pt][paradigm_names[0]][0][band][ch, :] # Absolute Power
                paradigm_1_abs = fooof_bands[pt][paradigm_names[1]][0][band][ch, :] # Absolute Power

                paradigm_0_rel = fooof_bands[pt][paradigm_names[0]][1][band][ch, :] # Relative Power
                paradigm_1_rel = fooof_bands[pt][paradigm_names[1]][1][band][ch, :] # Relative Power

                # print(paradigm_0_abs.shape, paradigm_1_abs.shape)
                # t_stat_abs, p_value_abs = ttest_rel(paradigm_0_abs, paradigm_1_abs)
                # t_stat_rel, p_value_rel = ttest_rel(paradigm_0_rel, paradigm_1_rel)

                _, _, p_value_rel, _ = compare_paired_arrays(paradigm_0_rel, paradigm_1_rel, test_normality=False)
                _, _, p_value_abs, _ = compare_paired_arrays(paradigm_0_abs, paradigm_1_abs, test_normality=False)

                fooof_bands_pvalues_rel[pt][band].append(p_value_rel)
                fooof_bands_pvalues_abs[pt][band].append(p_value_abs)

    return fooof_bands_pvalues_rel, fooof_bands_pvalues_abs, fooof_bands

















def plot_aperiodic_difference_all_subjects(fooof_aperiodic, raw_info, pat_state, save_path,
                                           show_params=True, title=None, show_all_cbars=False,
                                           cmap=None, value_units_heading="Relative Difference",
                                           row_height_pad=0.45):
    """
    Channel-level relative-difference topomaps for the aperiodic exponent,
    one row (Exponent), subjects as columns.

    row_height_pad compensates the once-per-figure costs (subject titles +
    tight_layout top reservation) so single-row heads match the 5-row
    periodic figure's head size at equal printed width. Use the SAME value
    as in plot_band_power_difference_all_subjects_significance_aperiodic.
    """
    from matplotlib.colors import LinearSegmentedColormap
    if cmap is None:
        # soft_div: blue = lower in condition 2, pink = higher in condition 2
        cmap = LinearSegmentedColormap.from_list(
            "soft_div", ["#6295BF", "#F7F7F5", "#E28DA9"])

    subjects = list(fooof_aperiodic.keys())
    subjects.sort()
    paradigm_names = list(fooof_aperiodic[subjects[0]].keys())

    params = list(next(iter(fooof_aperiodic.values()))[paradigm_names[0]].keys())
    param_symbols = {
        'Offset': r'$\mathbf{b}$',
        'Exponent': r'$\boldsymbol{\chi}$',
    }

    # params.remove("Offset")

    param_display_names = {p: param_symbols.get(p, p.upper()) for p in params}

    # Compute vmin/vmax for each parameter separately
    param_vmin = {}
    param_vmax = {}
    for param in params:
        all_diffs_param = []
        for subject in subjects:
            print("Using Following Paradigm Names and in the following order: ", paradigm_names)

            passive_vals = np.nanmean(fooof_aperiodic[subject][paradigm_names[1]][param], axis=1)
            rest_vals    = np.nanmean(fooof_aperiodic[subject][paradigm_names[0]][param], axis=1)
            diff = (passive_vals - rest_vals) / rest_vals
            all_diffs_param.append(diff)

        all_diffs_param = np.concatenate(all_diffs_param)
        param_vmin[param] = np.nanmin(all_diffs_param)
        param_vmax[param] = np.nanmax(all_diffs_param)

    fig, axes = plt.subplots(
        len(params), len(subjects),
        figsize=(4 * len(subjects), 3 * len(params) + row_height_pad),
    )

    if len(subjects) == 1:
        axes = np.expand_dims(axes, axis=1)
    if len(params) == 1:
        axes = np.expand_dims(axes, axis=0)

    if show_params:
        for param_idx, param in enumerate(params):
            axes[param_idx, 0].set_ylabel(param_display_names[param], fontsize=24,
                                          rotation=0, ha="right", va="center",
                                          fontweight='bold')

    for subj_idx, subject in enumerate(subjects):
        sub_num = list(pat_state.keys()).index(subject) + 1 if pat_state and subject in pat_state else subj_idx + 1
        axes[0, subj_idx].set_title(f"Sub{sub_num} ({pat_state[subject]})",
                                    fontsize=20, pad=10, fontweight='bold')

    for subj_idx, subject in enumerate(subjects):
        for param_idx, param in enumerate(params):
            print("Paradigm 0: ", paradigm_names[0])
            print("Paradigm 1: ", paradigm_names[1])

            passive_vals = np.nanmean(fooof_aperiodic[subject][paradigm_names[1]][param], axis=1)
            rest_vals    = np.nanmean(fooof_aperiodic[subject][paradigm_names[0]][param], axis=1)
            diff = (passive_vals - rest_vals) / rest_vals

            ax = axes[param_idx, subj_idx]
            im, _ = mne.viz.plot_topomap(
                diff,
                raw_info,
                axes=ax,
                cmap=cmap,
                show=False
            )

            if show_all_cbars or subj_idx == len(subjects) - 1:
                cbar = plt.colorbar(im, ax=ax, shrink=0.7, orientation="vertical", pad=0.05)
                cbar.ax.tick_params(labelsize=18)

    plt.tight_layout(rect=[0, 0, 1, 0.88])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close(fig)
















def plot_band_power_difference_all_subjects_significance_aperiodic(
    data_dict, raw_info, pat_state, save_path, n_channels,
    show_params=False, title=None,
    cmap=ListedColormap(["#9EC49C", "#EDF4F6"]),
    power_units_heading="Power Diference",
    row_height_pad=0.45,   # extra height (in) compensating once-per-figure costs
):
    """
    Plot channel-level significance topomaps for the aperiodic exponent,
    one row (Exponent), subjects as columns.

    row_height_pad compensates for the subject-title row and tight_layout's
    top reservation, which are paid once per figure: without it, a 1-row
    figure renders smaller heads than the 5-row periodic figure at equal
    printed width. Tune empirically against the periodic figure.
    """
    subjects = list(data_dict.keys())
    subjects.sort()
    paradigms = list(data_dict[subjects[0]].keys())
    params = list(data_dict[subjects[0]][paradigms[0]].keys())
    print("Aperiodic Components: ", params)
    # params.remove("Offset")

    param_symbols = {
        'Offset': r'$\mathbf{b}$',
        'Exponent': r'$\boldsymbol{\chi}$',
    }
    param_display_names = {p: param_symbols.get(p, p.upper()) for p in params}

    fig, axes = plt.subplots(
        len(params), len(subjects),
        figsize=(4 * len(subjects), 3 * len(params) + row_height_pad),
    )

    if len(subjects) == 1:
        axes = np.expand_dims(axes, axis=1)
    if len(params) == 1:
        axes = np.expand_dims(axes, axis=0)

    if show_params:
        for param_idx, param in enumerate(params):
            axes[param_idx, 0].set_ylabel(param_display_names[param], fontsize=24,
                                          rotation=0, ha="right", va="center",
                                          fontweight='bold')

    for subj_idx, subject in enumerate(subjects):
        sub_num = list(pat_state.keys()).index(subject) + 1 if pat_state and subject in pat_state else subj_idx + 1
        axes[0, subj_idx].set_title(f"Sub{sub_num} ({pat_state[subject]})",
                                    fontsize=20, pad=10, fontweight='bold')

    p_vals_all_subs = {}

    for subj_idx, subject in enumerate(subjects):
        p_vals_all_subs[subject] = {}
        p_values_param = {param: [] for param in params}

        for param_indx, param in enumerate(params):
            print("Paradigm 0: ", paradigms[0])
            print("Paradigm 1: ", paradigms[1])
            passive_aperiodic = data_dict[subject][paradigms[1]][param]
            rest_aperiodic = data_dict[subject][paradigms[0]][param]

            for ch in range(passive_aperiodic.shape[0]):
                _, _, p_value, _ = compare_paired_arrays(passive_aperiodic[ch, :],
                                                         rest_aperiodic[ch, :], test_normality=False)
                p_values_param[param].append(p_value)

        p_vals_all_subs[subject] = p_values_param

    fdr_corrected_p_vals = get_fdr_p_values(p_vals_all_subs, p_vals_per_band=n_channels)

    for subj_idx, subject in enumerate(subjects):
        for param_indx, param in enumerate(params):
            ax = axes[param_indx, subj_idx]
            # local binary mask — do NOT mutate fdr_corrected_p_vals in place
            p_plot = np.where(
                np.asarray(fdr_corrected_p_vals[subject][param], float) < 0.05,
                0.0, 1.0,
            )

            im, _ = mne.viz.plot_topomap(
                p_plot,
                raw_info,
                axes=ax,
                sensors=True,
                cmap=cmap,
                show=False,
                vlim=(0, 1),
                contours=0,
                extrapolate='local',
            )

    plt.tight_layout(rect=[0, 0, 1, 0.88])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close(fig)



def get_band_power_difference_all_subjects_significance_aperiodic_fdr(data_dict, pat_state, n_channels):
    """
    Plot EEG band power difference (passive - rest) for all subjects in a single figure.

    Parameters:
    - band_powers_all (dict): Dictionary containing band power differences for all subjects.
    - eeg_data_all (dict): Dictionary containing EEG data for all subjects.
    - title (str): Title for the entire figure (default: None).
    - cmap (str): Colormap for visualization (default: "viridis").
    """
    subjects = list(data_dict.keys())  # Extract subject names
    subjects.sort()  # Sort subjects alphabetically
    paradigms = list(data_dict[subjects[0]].keys())
    params = list(data_dict[subjects[0]][paradigms[0]].keys())
    print("Aperiodic Components: ", params)


    p_vals_all_subs = {}

    for subj_idx, subject in enumerate(subjects):
        p_vals_all_subs[subject] = {}
        # Select the appropriate Raw object for electrode positions
        # raw_object_for_topomap = eeg_data_all[subject][paradigms[0]]  # Use the first chunk

        param_difference = {} 
        p_values_param = {param:[] for param in params}
        
        for param_indx, param in enumerate(params):
            print("Paradigm 0: ", paradigms[0])
            print("Paradigm 1: ", paradigms[1])
            passive_aperiodic = data_dict[subject][paradigms[1]][param]
            rest_aperiodic = data_dict[subject][paradigms[0]][param]

            for ch in range(passive_aperiodic.shape[0]):
                # print("aaaa")
                # print(rest_aperiodic[ch, :])
                _, _, p_value, _ = compare_paired_arrays(passive_aperiodic[ch, :], rest_aperiodic[ch, :], test_normality=False)
                # t_stat, p_value = ttest_rel(passive_aperiodic[ch, :], rest_aperiodic[ch, :])
                # if subj_idx==2:
                #     print(p_value)
                p_values_param[param].append(p_value)

        p_vals_all_subs[subject] = p_values_param
    
    fdr_corrected_p_vals = get_fdr_p_values(p_vals_all_subs, p_vals_per_band=n_channels)

    return fdr_corrected_p_vals












