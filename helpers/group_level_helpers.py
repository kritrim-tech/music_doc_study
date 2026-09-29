
import numpy as np
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests


def bootstrap_paired_ttest(vals1, vals2, n_boot=1000, rng=None):
    """
    Nonparametric bootstrap paired t-test with pooled resampling
    (Dwivedi, Mallawaarachchi & Alvarado, 2017, Statist. Med. 36:2187-2205).
    Validated by the authors down to n=3 paired observations.
    """
    vals1 = np.asarray(vals1)
    vals2 = np.asarray(vals2)
    if np.isnan(vals1).any() or np.isnan(vals2).any():
        return np.nan, np.nan  
    n = len(vals1)

    pooled = np.concatenate([vals1, vals2])
    t_obs = ttest_rel(vals1, vals2).statistic

    t_boot = np.full(n_boot, np.nan)
    for j in range(n_boot):
        g1 = rng.choice(pooled, size=n, replace=True)
        g2 = rng.choice(pooled, size=n, replace=True)
        if np.std(g1 - g2) == 0:
            continue
        t_boot[j] = ttest_rel(g1, g2).statistic

    # p_boot = np.nanmean(np.abs(t_boot) >= np.abs(t_obs))
    valid = ~np.isnan(t_boot)
    p_boot = np.mean(np.abs(t_boot[valid]) >= np.abs(t_obs)) if valid.any() else np.nan
    return t_obs, p_boot


def group_level_bootstrap_test(power_dict, subjects, condition1, condition2, bands,
                                abs_rel=0, n_channels=128, n_boot=1000, alpha=0.05,
                                avg_method='mean', seed=42):
    """
    Group-level test using epoch-level channel data, with subject as the unit
    of analysis and the Dwivedi et al. (2017) bootstrap paired t-test.

    power_dict[sub][cond][abs_rel][band] -> array of shape (n_channels, n_epochs),
    where n_epochs may differ across subjects/conditions.

    Step 1: collapse each subject's epochs into one summary value per
            channel/band/condition.
    Step 2: for each channel and band, run the bootstrap paired t-test
            ACROSS SUBJECTS (n = len(subjects)) on these summaries.
    Step 3: FDR-correct across all channel x band tests.

    Returns
    -------
    raw_pvals : dict[band] -> array (n_channels,) of uncorrected bootstrap p-values
    fdr_pvals : dict[band] -> array (n_channels,) of FDR-corrected p-values
    sig_mask  : dict[band] -> boolean array (n_channels,)
    """
    rng = np.random.default_rng(seed)
    raw_pvals = {band: np.full(n_channels, np.nan) for band in bands}

    for band in bands:
        subj_summary1, subj_summary2 = [], []

        for sub in subjects:
            arr1 = np.asarray(power_dict[sub][condition1][abs_rel][band])  # (channels, epochs)
            arr2 = np.asarray(power_dict[sub][condition2][abs_rel][band])

            if avg_method == 'mean':
                summ1 = np.nanmean(arr1, axis=1)
                summ2 = np.nanmean(arr2, axis=1)
            else:
                summ1 = np.nanmedian(arr1, axis=1)
                summ2 = np.nanmedian(arr2, axis=1)

            subj_summary1.append(summ1)
            subj_summary2.append(summ2)

        subj_summary1 = np.stack(subj_summary1, axis=0)  # (n_subjects, n_channels)
        subj_summary2 = np.stack(subj_summary2, axis=0)

        for ch in range(n_channels):
            _, p_val = bootstrap_paired_ttest(
                subj_summary1[:, ch], subj_summary2[:, ch], n_boot=n_boot, rng=rng
            )
            raw_pvals[band][ch] = p_val

    # FDR correction across all channel x band tests in this group-level analysis
    all_pvals_flat = np.concatenate([raw_pvals[band] for band in bands])
    all_pvals_flat = np.nan_to_num(all_pvals_flat, nan=1.0)
    _, all_pvals_fdr, _, _ = multipletests(all_pvals_flat, alpha=alpha, method='fdr_bh')

    fdr_pvals, sig_mask = {}, {}
    idx = 0
    for band in bands:
        fdr_pvals[band] = all_pvals_fdr[idx: idx + n_channels]
        sig_mask[band] = fdr_pvals[band] < alpha
        idx += n_channels

    return raw_pvals, fdr_pvals, sig_mask




import numpy as np
from statsmodels.stats.multitest import multipletests


def group_level_bootstrap_test_network(power_dict, subjects, condition1, condition2, bands, networks,
                                        n_boot=1000, alpha=0.05, avg_method='mean', seed=42):
    """
    Group-level test for NETWORK-level power, using subject as the unit of
    analysis, following the same logic as the channel-level version.

    power_dict[sub][cond][band][network] -> array of shape (n_epochs,),
    where n_epochs may differ across subjects/conditions.

    Step 1: collapse each subject's epochs into one summary value per
            network/band/condition.
    Step 2: for each network and band, run the bootstrap paired t-test
            ACROSS SUBJECTS (n = len(subjects)) on these summaries.
    Step 3: FDR-correct across all network x band tests.

    Returns
    -------
    raw_pvals : dict[band][network] -> uncorrected bootstrap p-value
    fdr_pvals : dict[band][network] -> FDR-corrected p-value
    sig_mask  : dict[band][network] -> bool
    """
    rng = np.random.default_rng(seed)
    raw_pvals = {band: {} for band in bands}

    for band in bands:
        for network in networks:
            subj_summary1, subj_summary2 = [], []

            for sub in subjects:
                arr1 = np.asarray(power_dict[sub][condition1][band][network])  # (n_epochs,)
                arr2 = np.asarray(power_dict[sub][condition2][band][network])

                if avg_method == 'mean':
                    summ1 = np.nanmean(arr1)
                    summ2 = np.nanmean(arr2)
                else:
                    summ1 = np.nanmedian(arr1)
                    summ2 = np.nanmedian(arr2)

                subj_summary1.append(summ1)
                subj_summary2.append(summ2)

            subj_summary1 = np.array(subj_summary1)  # (n_subjects,)
            subj_summary2 = np.array(subj_summary2)

            _, p_val = bootstrap_paired_ttest(subj_summary1, subj_summary2, n_boot=n_boot, rng=rng)
            raw_pvals[band][network] = p_val

    # FDR correction across all network x band tests in this group-level analysis
    all_pvals_flat = np.array([raw_pvals[band][network] for band in bands for network in networks])
    all_pvals_flat = np.nan_to_num(all_pvals_flat, nan=1.0)
    _, all_pvals_fdr, _, _ = multipletests(all_pvals_flat, alpha=alpha, method='fdr_bh')

    fdr_pvals, sig_mask = {band: {} for band in bands}, {band: {} for band in bands}
    idx = 0
    for band in bands:
        for network in networks:
            fdr_pvals[band][network] = all_pvals_fdr[idx]
            sig_mask[band][network] = all_pvals_fdr[idx] < alpha
            idx += 1

    return raw_pvals, fdr_pvals, sig_mask




def _stack_comparison_label(label):
    """
    Split a comparison label into two stacked lines at the first
    occurrence of "vs.", e.g. "Rest vs. Continuous Music" ->
    "Rest vs." / "Continuous Music". If "vs." is not found, the label
    is returned unchanged (single line).
    """
    idx = label.lower().find(" vs.")
    if idx == -1:
        return label

    idx_end = idx + len(" vs.")
    line1 = label[:idx_end].strip()
    line2 = label[idx_end:].strip()
    line2 = line2.lstrip("\\").strip()  # drop a stray leading "\" if the
                                         # original label had "vs.\ " escaping

    return f"\\shortstack[l]{{{line1}\\\\{line2}}}"


def build_channel_style_arrays(fdr_corrected_p_vals_network, sig_mask_network, subjects, bands, networks):
    """
    Convert network-level nested dicts (keyed by network name) into the
    array-indexed format expected by latex_group_vs_individual_table,
    using `networks` as the fixed ordering.
    """
    fdr_corrected_p_vals_arr = {}
    for sub in subjects:
        fdr_corrected_p_vals_arr[sub] = {}
        for band in bands:
            fdr_corrected_p_vals_arr[sub][band] = np.array(
                [fdr_corrected_p_vals_network[sub][band][net] for net in networks]
            )

    sig_mask_arr = {}
    for band in bands:
        sig_mask_arr[band] = np.array(
            [sig_mask_network[band][net] for net in networks]
        )

    return fdr_corrected_p_vals_arr, sig_mask_arr

def latex_group_vs_individual_table(
    fdr_corrected_p_vals,
    subjects,
    sig_mask,
    bands,
    comparison_label,
    mode='append',
    n_channels=128,
    alpha=0.05,
    caption="Group-level versus individual-level significance across channels and frequency bands.",
    label="tab:group_vs_individual",
):
    """
    Build LaTeX table rows (or the full table) summarising, per band, how many
    channels reach significance at the group level versus at the
    individual-subject level.

    Parameters
    ----------
    fdr_corrected_p_vals : dict
        fdr_corrected_p_vals[sub][band] -> array (n_channels,) of subject-level
        FDR-corrected p-values.
    subjects : list
        Subject IDs included in THIS comparison.
    sig_mask : dict
        sig_mask[band] -> boolean array (n_channels,), the group-level
        significance mask.
    bands : list
        Band keys to print, in order. 'gamma high' is dropped; 'gamma low'
        is relabelled and printed as plain $\\gamma$.
    comparison_label : str
        Row label. Split into two stacked lines at "vs." if present
        (e.g. "Rest vs. Continuous Music"), otherwise printed as-is.
    mode : {'start', 'append'}
        'start'  -> full table (header + this comparison's rows + closing lines).
        'append' -> only this comparison's rows, to paste before the closing
                    lines of an already-open table.

    Notes
    -----
    Requires \\usepackage{multirow,tabularx,booktabs} in the preamble.
    (\\shortstack, used for the two-line comparison label, is core LaTeX.)
    """
    greek_symbols = {
        'delta': r'$\delta$',
        'theta': r'$\theta$',
        'alpha': r'$\alpha$',
        'beta': r'$\beta$',
        'gamma low': r'$\gamma$',
    }

    bands = [b for b in bands if b != 'gamma high']

    n_subs = len(subjects)
    majority_thresh = int(np.ceil(n_subs / 2))

    rows = []
    for band in bands:
        group_sig = int(sig_mask[band].sum())

        any_sig_mask = np.zeros(n_channels, dtype=bool)
        n_subs_sig = np.zeros(n_channels, dtype=int)
        for sub in subjects:
            sub_sig = fdr_corrected_p_vals[sub][band] < alpha
            any_sig_mask |= sub_sig
            n_subs_sig += sub_sig.astype(int)

        any_sig = int(any_sig_mask.sum())
        majority_sig = int((n_subs_sig >= majority_thresh).sum())
        all_sig = int((n_subs_sig == n_subs).sum())

        band_label = greek_symbols.get(band, band.capitalize())
        rows.append(
            f"& {band_label} & {group_sig}/{n_channels} & {any_sig}/{n_channels} & "
            f"{majority_sig}/{n_channels} ($\\geq${majority_thresh}/{n_subs}) & "
            f"{all_sig}/{n_channels} \\\\"
        )

    n_bands = len(bands)
    stacked_label = _stack_comparison_label(comparison_label)
    body = f"\\multirow{{{n_bands}}}{{*}}{{{stacked_label}}}\n"
    body += "\n".join(rows) + "\n"

    if mode == 'start':
        return (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n"
            "\n"
            "\\begin{tabularx}{\\textwidth}{@{}l l X X X X@{}}\n"
            "\\toprule\n"
            "\\textbf{Comparison} &\n"
            "\\textbf{Band} &\n"
            "\\textbf{Group-level sig.} &\n"
            "\\textbf{Sig. in $\\geq$1 subject} &\n"
            "\\textbf{Sig. in majority} &\n"
            "\\textbf{Sig. in all subjects} \\\\\n"
            "\\midrule\n"
            "\n"
            f"{body}\n"
            "\\bottomrule\n"
            "\\end{tabularx}\n"
            "\\end{table}\n"
        )
    elif mode == 'append':
        return f"\\midrule\n\n{body}\n"
    else:
        raise ValueError("mode must be 'start' or 'append'")