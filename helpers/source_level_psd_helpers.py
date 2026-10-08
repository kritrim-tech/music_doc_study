# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path
from collections import defaultdict

import os
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import matplotlib.patches as mpatches
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from statsmodels.stats.multitest import fdrcorrection

# near the top of the file, after imports:
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# poles: blue = lower in condition 2, pink = higher in condition 2
soft_div = LinearSegmentedColormap.from_list(
    "soft_div", ["#6295BF", "#F7F7F5", "#E28DA9"])   # = pencil(steelblue,.15) / centre / pencil(palevioletred,.2)
cmap_sig = ListedColormap(["#6FB46F", "#EDF4F6"])    # = pencil(forestgreen,.35) / pencil("#DAE8EC",.5)



def cluster_and_compare_fooof(
    data_dict,
    cond1_name: str,
    cond2_name: str,
    mapping_dict,
    is_anatomical: bool = True,
    avg_method: str = 'mean',
    alpha: float = 0.05,
    abs_rel: int = 1,
):
    """
    Cluster regions, compare conditions, and also return cluster-averaged power.
    """

    region_list = list(mapping_dict.keys())
    if len(region_list) != 90:
        raise ValueError(f"Mapping must have exactly 90 regions. Got {len(region_list)}")

    region_to_cluster = {reg: mapping_dict[reg] for reg in region_list}
    cluster_list = list(dict.fromkeys(region_to_cluster.values()))  # preserve order
    n_clusters = len(cluster_list)

    raw_pvals = defaultdict(lambda: defaultdict(dict))
    cluster_means = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for sub in data_dict:
        if cond1_name not in data_dict[sub] or cond2_name not in data_dict[sub]:
            print(f"Skipping {sub}: missing condition")
            continue

        if abs_rel == 0 or abs_rel == 1:
            for band in data_dict[sub][cond1_name][abs_rel]:
                try:
                    array1 = np.asarray(data_dict[sub][cond1_name][abs_rel][band])  # (90, epochs)
                    array2 = np.asarray(data_dict[sub][cond2_name][abs_rel][band])

                    if array1.shape[0] != 90 or array2.shape[0] != 90:
                        print(f"Shape error for {sub} - {band}")
                        continue

                    # === Cluster averaging ===
                    clustered1 = np.zeros((n_clusters, array1.shape[1]))
                    clustered2 = np.zeros((n_clusters, array2.shape[1]))

                    for i, cluster in enumerate(cluster_list):
                        idx = [j for j, reg in enumerate(region_list) 
                            if region_to_cluster[reg] == cluster]

                        if avg_method == 'mean':
                            clustered1[i] = np.nanmean(array1[idx], axis=0)
                            clustered2[i] = np.nanmean(array2[idx], axis=0)
                        else:
                            clustered1[i] = np.nanmedian(array1[idx], axis=0)
                            clustered2[i] = np.nanmedian(array2[idx], axis=0)

                        # Save cluster means
                        cluster_means[sub][cond1_name][band][cluster] = clustered1[i].copy()
                        cluster_means[sub][cond2_name][band][cluster] = clustered2[i].copy()


                except Exception as e:
                    print(f"Error processing {sub} {band}: {e}")

        else:
            for band in data_dict[sub][cond1_name]:
                try:
                    array1 = np.asarray(data_dict[sub][cond1_name][band])  # (90, epochs)
                    array2 = np.asarray(data_dict[sub][cond2_name][band])

                    if array1.shape[0] != 90 or array2.shape[0] != 90:
                        print(f"Shape error for {sub} - {band}")
                        continue

                    # === Cluster averaging ===
                    clustered1 = np.zeros((n_clusters, array1.shape[1]))
                    clustered2 = np.zeros((n_clusters, array2.shape[1]))

                    for i, cluster in enumerate(cluster_list):
                        idx = [j for j, reg in enumerate(region_list) 
                            if region_to_cluster[reg] == cluster]

                        if avg_method == 'mean':
                            clustered1[i] = np.nanmean(array1[idx], axis=0)
                            clustered2[i] = np.nanmean(array2[idx], axis=0)
                        else:
                            clustered1[i] = np.nanmedian(array1[idx], axis=0)
                            clustered2[i] = np.nanmedian(array2[idx], axis=0)

                        # Save cluster means
                        cluster_means[sub][cond1_name][band][cluster] = clustered1[i].copy()
                        cluster_means[sub][cond2_name][band][cluster] = clustered2[i].copy()


                except Exception as e:
                    print(f"Error processing {sub} {band}: {e}")


    print(f"✅ Done! Used {'Anatomical' if is_anatomical else 'Functional Network'} clustering")
    print(f"Number of clusters: {n_clusters}")

    return raw_pvals, cluster_list, cluster_means















# ─────────────────────────────────────────────────────────────────────────────
# 1.  BOOTSTRAP SIGNIFICANCE
# ─────────────────────────────────────────────────────────────────────────────

def compute_bootstrap_pvals(
    cluster_means,
    condition1,
    condition2,
    bands,
    n_bootstrap=1000,
    alpha_level=0.05,
    rng_seed=42,
):
    """
    For every (subject, band, network) triple compute a permutation p-value
    for the difference  mean(condition2) - mean(condition1).

    Null distribution: pool all epochs from both conditions, randomly split
    into two groups of the original sizes n_bootstrap times. This centres
    the null at 0 by construction.

    p-value: two-tailed, with continuity correction:
        p = (count + 1) / (n_bootstrap + 1)
    where `count` is the number of permuted differences whose absolute
    value meets or exceeds the absolute observed difference.

    FDR correction (Benjamini-Hochberg) applied SEPARATELY for each
    (subject, network) pair, across the frequency bands within that pair
    only. No correction is applied across subjects (subject-specific
    hypotheses) or across networks (the three networks were selected a
    priori as planned comparisons).

    Returns
    -------
    pval_dict : dict [sub][band][network] -> FDR-corrected p-value
    sig_dict  : dict [sub][band][network] -> bool
    ci_dict   : dict [sub][band][network] -> (ci_low, ci_high) of null dist
    obs_dict  : dict [sub][band][network] -> observed mean difference
    """
    rng      = np.random.default_rng(rng_seed)
    subjects = sorted(cluster_means.keys())
    networks = list(cluster_means[subjects[0]][condition1][bands[0]].keys())

    pval_dict, sig_dict, ci_dict, obs_dict = {}, {}, {}, {}

    for sub in subjects:
        for net in networks:

            # ── collect raw (uncorrected) p-values across bands, for this (sub, net) ──
            band_keys, raw_pvals, ci_low, ci_hi, obs = [], [], [], [], []

            for band in bands:
                try:
                    v1 = np.asarray(cluster_means[sub][condition1][band][net], dtype=float)
                    v2 = np.asarray(cluster_means[sub][condition2][band][net], dtype=float)
                except KeyError:
                    continue

                v1 = v1[~np.isnan(v1)]
                v2 = v2[~np.isnan(v2)]
                if v1.size < 2 or v2.size < 2:
                    print(f"Skipping {sub}/{band}/{net}: too few valid epochs")
                    continue

                observed_diff = np.mean(v2) - np.mean(v1)
                n1, n2        = len(v1), len(v2)
                pooled        = np.concatenate([v1, v2])

                null_diffs = np.empty(n_bootstrap)
                for i in range(n_bootstrap):
                    perm = rng.permutation(pooled)
                    null_diffs[i] = np.mean(perm[n1:]) - np.mean(perm[:n1])

                # two-tailed p-value, continuity-corrected
                count = np.sum(np.abs(null_diffs) >= np.abs(observed_diff))
                p = (count + 1) / (n_bootstrap + 1)

                ci_l, ci_h = np.percentile(null_diffs, [2.5, 97.5])

                band_keys.append(band)
                raw_pvals.append(p)
                ci_low.append(ci_l)
                ci_hi.append(ci_h)
                obs.append(observed_diff)

            if not raw_pvals:
                continue

            # ── FDR correction across bands ONLY, within this (subject, network) pair ──
            _, pvals_fdr = fdrcorrection(raw_pvals, alpha=alpha_level, method="indep")

            for band, p_fdr, cl, ch, od in zip(band_keys, pvals_fdr, ci_low, ci_hi, obs):
                pval_dict.setdefault(sub, {}).setdefault(band, {})[net] = p_fdr
                sig_dict.setdefault(sub, {}).setdefault(band, {})[net]  = p_fdr < alpha_level
                ci_dict.setdefault(sub, {}).setdefault(band, {})[net]   = (cl, ch)
                obs_dict.setdefault(sub, {}).setdefault(band, {})[net]  = od

    return pval_dict, sig_dict, ci_dict, obs_dict


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SPLIT VIOLIN HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _split_violin(ax, data_left, data_right, pos, width=0.7,
                  color_left="#4C72B0", color_right="#DD8452", alpha=0.6):

    from scipy.stats import gaussian_kde

    def _kde(data, y_grid):
        if len(data) < 2 or np.std(data) == 0:
            return np.zeros_like(y_grid)
        return gaussian_kde(data, bw_method="scott")(y_grid)

    all_data = np.concatenate([data_left, data_right])
    y_grid   = np.linspace(np.nanmin(all_data), np.nanmax(all_data), 300)

    kde_l = _kde(data_left,  y_grid)
    kde_r = _kde(data_right, y_grid)

    norm   = max(kde_l.max(), kde_r.max(), 1e-10)
    half_w = width / 2
    kl     = kde_l / norm * half_w
    kr     = kde_r / norm * half_w

    verts_l = ([(pos - k, y) for k, y in zip(kl,       y_grid)] +
               [(pos,     y) for y    in             y_grid[::-1]])
    ax.add_patch(PathPatch(Path(verts_l + [verts_l[0]]),
                           facecolor=color_left, alpha=alpha,
                           edgecolor=color_left, lw=0.7))

    verts_r = ([(pos,     y) for y    in             y_grid] +
               [(pos + k, y) for k, y in zip(kr[::-1], y_grid[::-1])])
    ax.add_patch(PathPatch(Path(verts_r + [verts_r[0]]),
                           facecolor=color_right, alpha=alpha,
                           edgecolor=color_right, lw=0.7))

    ax.hlines(np.median(data_left),  pos - half_w * 0.6, pos,
              color="white", lw=1.8, zorder=5)
    ax.hlines(np.median(data_right), pos, pos + half_w * 0.6,
              color="white", lw=1.8, zorder=5)

    rng = np.random.default_rng(seed=42)
    for data, sign, color in [(data_left, -1, color_left),
                               (data_right, +1, color_right)]:
        jitter = rng.uniform(0.01, half_w * 0.5, size=len(data)) * sign
        ax.scatter(np.full(len(data), pos) + jitter, data,
                   s=5, color=color, alpha=0.45, zorder=4)


def plot_band_power_comparisons(
    cluster_means,
    condition1,
    condition2,
    bands=("delta", "theta", "alpha", "beta", "gamma"),
    alpha_level=0.05,
    n_bootstrap=1000,
    figsize=(28, 5),
    save_dir=".",
    save_tag="",
    legend_label1="",
    legend_label2="",
    power_unit=r"Periodic Power ($\log_{10}$ a.u.)",
    save=True,
    subject_order=None,   # ← pass a list to control left-to-right order
    pat_state=None,
    vertical=False,
    show_legend=True,
    FS_PVAL = 16, 
    FS_BAND = 26, 
    FS_YLAB = 20, 
    FS_YTICK = 20, 
    FS_LEG = 20, 
    FS_TITLE = 26,     show_ylabel=True,   # False -> omit the y-axis label text (state the unit in the caption)
):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    C1 = "#6295BF"   # condition 1 — pencil steelblue (soft_div's negative pole)
    C2 = "#E28DA9"   # condition 2 — pencil palevioletred (soft_div's positive pole)
    SIG_COL = "#2ca02c"
    NS_COL  = "#9a9a9a"

    # Greek-letter display mapping (keyed by real band names, including the
    # "gamma low" -> gamma consolidation)
    greek_symbols = {
        'delta': r'$\mathbf{\delta}$',
        'theta': r'$\mathbf{\theta}$',
        'alpha': r'$\mathbf{\alpha}$',
        'beta':  r'$\mathbf{\beta}$',
        'gamma': r'$\mathbf{\gamma}$',
        'gamma low': r'$\mathbf{\gamma}$',
        'Offset':   r'$\mathbf{b}$',
        'Exponent': r'$\boldsymbol{\chi}$',
    }
    band_display_labels = [greek_symbols.get(b, b.capitalize()) for b in bands]

    # Network display abbreviations (spell out in the figure caption:
    # SMN = Sensorimotor Network, DMN = Default Mode Network, LN = Limbic Network)
    net_abbrev = {
        'Sensorimotor Network': 'SMN',
        'Default Mode Network': 'DMN',
        'Limbic Network': 'LN',
    }

    # Mode-dependent fontsizes (tune from the MERGED pdf, not individual panels)

    # FS_PVAL, FS_BAND, FS_YLAB, FS_YTICK, FS_LEG, FS_TITLE = 16, 26, 20, 20, 20, 26
    # if vertical:
    #     FS_PVAL, FS_BAND, FS_YLAB, FS_YTICK, FS_LEG, FS_TITLE = 18, 26, 20, 20, 20, 26
    # else:
    #     FS_PVAL, FS_BAND, FS_YLAB, FS_YTICK, FS_LEG, FS_TITLE = 14, 16, 16, 16, 15, 16

    print("Running bootstrap tests …")
    pval_dict, sig_dict, ci_dict, obs_dict = compute_bootstrap_pvals(
        cluster_means, condition1, condition2,
        bands=bands, n_bootstrap=n_bootstrap, alpha_level=alpha_level,
    )
    print("Done.")

    # ── subject order: explicit list takes priority, else alphabetical ────────
    if subject_order is not None:
        missing = [s for s in subject_order if s not in cluster_means]
        if missing:
            print(f"Warning: subjects in subject_order not found in data: {missing}")
        subjects = [s for s in subject_order if s in cluster_means]
    else:
        subjects = sorted(cluster_means.keys())

    first_sub = subjects[0]
    last_sub  = subjects[-1]
    mid_sub   = subjects[len(subjects) // 2]

    try:
        start_offset = list(pat_state.keys()).index(subjects[0]) if pat_state else 0
    except ValueError:
        print(f"Warning: '{subjects[0]}' not found in pat_state keys — using start_offset=0.")
        start_offset = 0

    BAND_POS = {b: i + 1.0 for i, b in enumerate(bands)}
    VIOLIN_W = 0.72

    # ── global y limits ───────────────────────────────────────────────────────
    all_vals = []
    for sub in subjects:
        for band in bands:
            for net in cluster_means[sub][condition1][band]:
                try:
                    all_vals.extend(cluster_means[sub][condition1][band][net])
                    all_vals.extend(cluster_means[sub][condition2][band][net])
                except KeyError:
                    print(f"Warning: missing values for {sub}/{band}/{net}")

    g_min, g_max = np.nanmin(all_vals), np.nanmax(all_vals)
    g_rng  = g_max - g_min
    y_bot  = g_min - g_rng * 0.04
    y_sig  = g_max + g_rng * 0.06
    y_top  = g_max + g_rng * 0.22

    # ── per-subject figures ───────────────────────────────────────────────────
    for subj_idx, sub in enumerate(subjects):
        is_first = sub == first_sub
        is_last  = sub == last_sub

        networks = list(cluster_means[sub][condition1][bands[0]].keys())
        n_nets   = len(networks)
        first_net = networks[0]
        last_net  = networks[-1]

        row_height = figsize[1]
        if vertical:
            fig, axes = plt.subplots(1, n_nets,
                                     figsize=(figsize[0] * n_nets, row_height),
                                     squeeze=True)
        else:
            fig, axes = plt.subplots(n_nets, 1,
                                     figsize=(figsize[0], row_height * n_nets),
                                     squeeze=True)
        if n_nets == 1:
            axes = [axes]

        # ── subject label (horizontal mode only; vertical rows are named in
        #    the figure caption instead — merged row order is sub-number
        #    ascending, i.e. SUB3..SUB7) ──────────────────────────────────────
        # state_str = f" ({pat_state[sub]})" if pat_state and sub in pat_state else ""
        # if not vertical:
        #     axes[0].text(0.995, 1.04, f"SUB{start_offset + subj_idx + 1}{state_str}",
        #                  transform=axes[0].transAxes, ha="right", va="top",
        #                  fontsize=18, fontweight="bold", zorder=10)

        for ax, network in zip(axes, networks):

            for band in bands:
                pos = BAND_POS[band]
                hw  = VIOLIN_W / 2

                try:
                    v1 = np.asarray(cluster_means[sub][condition1][band][network], float)
                    v2 = np.asarray(cluster_means[sub][condition2][band][network], float)
                except KeyError:
                    print(f"Missing data for {sub} - {band} - {network}, skipping")
                    continue

                v1 = v1[~np.isnan(v1)]
                v2 = v2[~np.isnan(v2)]
                if v1.size < 2 or v2.size < 2:
                    print(f"Too few valid epochs for {sub} - {band} - {network}, skipping")
                    continue

                _split_violin(ax, v1, v2, pos=pos, width=VIOLIN_W,
                              color_left=C1, color_right=C2)

                sig      = sig_dict.get(sub, {}).get(band, {}).get(network, False)
                pval     = pval_dict.get(sub, {}).get(band, {}).get(network, np.nan)
                line_col = SIG_COL if sig else NS_COL
                line_len = hw * 0.9

                # ax.hlines(y_sig, pos - line_len, pos + line_len,
                #           color=line_col, lw=2.5, zorder=6)

                pval_str = ("<.001" if pval < 0.001
                            else f"{pval:.3f}" if not np.isnan(pval)
                            else "n/a")
                ax.text(pos, y_sig + g_rng * 0.018, pval_str,
                        ha="center", va="bottom",
                        fontsize=FS_PVAL, color=line_col, fontweight="normal", alpha=0.75)

            # ── axes cosmetics ────────────────────────────────────────────────
            ax.set_xlim(0.35, len(bands) + 0.65)
            ax.set_ylim(y_bot, y_top)
            ax.set_xticks(list(BAND_POS.values()))
            ax.spines[["top", "right"]].set_visible(False)

            for p in BAND_POS.values():
                ax.axvline(p, color="#e0e0e0", lw=0.4, zorder=0)

            # x-axis labels: bottom edge of the grid only
            if (is_last if vertical else network == last_net):
                ax.set_xticklabels(band_display_labels, fontsize=FS_BAND, fontweight="bold")
            else:
                ax.set_xticklabels([])
                ax.tick_params(bottom=False)

            # ── network titles + y-axis ───────────────────────────────────────
            if is_first or vertical:
                if is_first:
                    ax.set_title(net_abbrev.get(network, network), fontsize=FS_TITLE,
                                 fontweight="bold", pad=6, loc="left")
                else:
                    ax.set_title("")

                show_y = (network == first_net) if vertical else is_first
                if show_y:
                    if show_ylabel:
                        ax.set_ylabel(power_unit, fontsize=FS_YLAB)
                    ax.tick_params(axis="y", labelsize=FS_YTICK)
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
                else:
                    ax.set_yticklabels([])
                    ax.spines["left"].set_visible(False)
                    ax.tick_params(left=False)
            else:
                ax.set_title("")
                ax.set_yticklabels([])
                ax.spines["left"].set_visible(False)
                ax.tick_params(left=False)

            # ── condition legend: first network panel of the last subject ────
            if is_last and network == first_net:
                cond_handles = [
                    mpatches.Patch(color=C1, alpha=0.7, label=legend_label1),
                    mpatches.Patch(color=C2, alpha=0.7, label=legend_label2),
                ]

                if show_legend:
                    ax.legend(
                        handles=cond_handles,
                        loc="upper left",
                        bbox_to_anchor=(0.0, 0.85),
                        fontsize=FS_LEG,
                        frameon=False,
                    )

        plt.tight_layout()

        if save and save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fname = os.path.join(
                save_dir,
                f"{sub}_{condition1}_vs_{condition2}_{save_tag}.pdf",
            )
            plt.savefig(fname, bbox_inches="tight")

        plt.close(fig)

    # ── legend as a separate figure ───────────────────────────────────────────
    fig_leg, ax_leg = plt.subplots(figsize=(6, 0.7))
    ax_leg.axis("off")
    handles = [
        mpatches.Patch(color=C1,      alpha=0.7, label=condition1),
        mpatches.Patch(color=C2,      alpha=0.7, label=condition2),
        mpatches.Patch(color=SIG_COL, alpha=0.8, label=f"sig. (FDR p < {alpha_level})"),
        mpatches.Patch(color=NS_COL,  alpha=0.8, label="n.s."),
    ]
    ax_leg.legend(
        handles=handles, loc="center", ncol=4,
        fontsize=16, frameon=False,
    )
    plt.tight_layout()

    if save and save_dir is not None:
        os.makedirs(save_dir + "merged_all/", exist_ok=True)
        leg_path = os.path.join(save_dir + "merged_all/", f"legend_{condition1}_vs_{condition2}_{save_tag}.pdf")
        plt.savefig(leg_path, bbox_inches="tight")
        print(f"Legend saved → {leg_path}")

    plt.close(fig_leg)

    return pval_dict









def stack_subject_figures_pdf(
    folder_path,
    output_path,
    fig_width=16,       # inches — target panel width (vertical mode)
    row_height=5,        # inches — target panel height (horizontal mode)
    stack_mode="horizontal",  # "vertical" or "horizontal"
    gap=0.05,             # inches — small gap between stacked panels
):

    
    import fitz  # PyMuPDF — pip install pymupdf
    import os
    import re

    """
    Stack per-subject PDF figures into a single PDF, preserving full vector
    quality. Each source page is embedded via show_pdf_page, which copies
    the underlying vector content directly rather than rasterizing it.
    """
    PT_PER_INCH = 72

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    def _sub_num(filename):
        match = re.search(r"sub(\d+)", filename)
        return int(match.group(1)) if match else 999

    pdf_files = sorted(pdf_files, key=_sub_num)

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {folder_path}")

    src_docs = [fitz.open(os.path.join(folder_path, f)) for f in pdf_files]
    src_pages = [doc[0] for doc in src_docs]  # assumes each PDF is single-page

    gap_pt = gap * PT_PER_INCH
    out_doc = fitz.open()

    if stack_mode == "vertical":
        target_w_pt = fig_width * PT_PER_INCH
        # scale each page to target_w_pt, preserving its own aspect ratio
        scaled_heights = [target_w_pt * (p.rect.height / p.rect.width) for p in src_pages]
        total_h_pt = sum(scaled_heights) + gap_pt * (len(src_pages) - 1)

        out_page = out_doc.new_page(width=target_w_pt, height=total_h_pt)

        y_cursor = 0
        for src_doc, h_pt in zip(src_docs, scaled_heights):
            rect = fitz.Rect(0, y_cursor, target_w_pt, y_cursor + h_pt)
            out_page.show_pdf_page(rect, src_doc, 0)
            y_cursor += h_pt + gap_pt

    elif stack_mode == "horizontal":
        target_h_pt = row_height * PT_PER_INCH
        scaled_widths = [target_h_pt * (p.rect.width / p.rect.height) for p in src_pages]
        total_w_pt = sum(scaled_widths) + gap_pt * (len(src_pages) - 1)

        out_page = out_doc.new_page(width=total_w_pt, height=target_h_pt)

        x_cursor = 0
        for src_doc, w_pt in zip(src_docs, scaled_widths):
            rect = fitz.Rect(x_cursor, 0, x_cursor + w_pt, target_h_pt)
            out_page.show_pdf_page(rect, src_doc, 0)
            x_cursor += w_pt + gap_pt

    else:
        raise ValueError(f"stack_mode must be 'vertical' or 'horizontal', got '{stack_mode}'")

    out_doc.save(output_path)
    out_doc.close()
    for doc in src_docs:
        doc.close()

    print(f"Saved → {output_path}")










def build_cluster_summary(fdr_corrected_p_vals, power_dict, condition1, condition2, band, sub_id,
                           label_region_mapping, region_to_anatomical_lobe, alpha=0.05):
    """
    For one subject/band, aggregate region-level significance + directionality
    into anatomical-cluster-level summaries.
    """
    ordered_region_names = [label_region_mapping[k] for k in list(label_region_mapping.keys())[:90]]

    pvals = np.asarray(fdr_corrected_p_vals[sub_id][band])
    v1 = np.asarray(power_dict[sub_id][condition1][band])
    v2 = np.asarray(power_dict[sub_id][condition2][band])

    # collapse epochs FIRST (ratio of means), matching the region-level map
    # panels; per-epoch ratios explode when single epochs pass near zero
    if v1.ndim == 2:
        v1 = np.nanmean(v1, axis=1)
    if v2.ndim == 2:
        v2 = np.nanmean(v2, axis=1)
    v1 = v1.ravel()
    v2 = v2.ravel()

    with np.errstate(divide='ignore', invalid='ignore'):
        frac_diff = np.where(v1 != 0, (v2 - v1) / v1, np.nan)



    sig_mask = pvals < alpha

    clusters = {}
    for i, region in enumerate(ordered_region_names):
        cluster = region_to_anatomical_lobe.get(region)
        if cluster is None:
            continue
        c = clusters.setdefault(cluster, {'n_total': 0, 'n_sig': 0, 'sig_diffs': []})
        c['n_total'] += 1
        if sig_mask[i]:
            c['n_sig'] += 1
            c['sig_diffs'].append(frac_diff[i])

    summary = {}
    for cluster, d in clusters.items():
        prop_sig = d['n_sig'] / d['n_total'] if d['n_total'] > 0 else np.nan
        avg_diff = np.nanmean(d['sig_diffs']) if d['sig_diffs'] else np.nan
        summary[cluster] = {'prop_sig': prop_sig, 'avg_diff': avg_diff,
                             'n_sig': d['n_sig'], 'n_total': d['n_total']}
    return summary


def build_cluster_summary_all(fdr_corrected_p_vals, power_dict, condition1, condition2, bands,
                               label_region_mapping, region_to_anatomical_lobe, alpha=0.05,
                               subjects=None):
    subjects = subjects or sorted(fdr_corrected_p_vals.keys())
    result = {}
    for sub_id in subjects:
        result[sub_id] = {}
        for band in bands:
            result[sub_id][band] = build_cluster_summary(
                fdr_corrected_p_vals, power_dict, condition1, condition2, band, sub_id,
                label_region_mapping, region_to_anatomical_lobe, alpha=alpha,
            )
    return result



def plot_cluster_bubble_heatmap(cluster_summary, bands, cluster_order=None,
                                 pat_state=None, sub_labels=None,
                                 condition1_label="Condition 1", condition2_label="Condition 2",
                                 cmap=soft_div, max_bubble_size=600,
                                 figsize_per_panel=(4, 6), save_path=None,
                                 row_spacing=1.6,
                                 vertical=False,          # True -> one panel per subject, stacked as ROWS
                                 show_sub_labels=True,
                                 abbrev_clusters=False,   # True -> first letter only (expand in caption)
                                 cbar_location="right",   # "right" | "bottom"
                                 cbar_label=r"Mean $\Delta P$, significant regions",
                                 per_band_cnorm=False):   # True -> own colour scale + colorbar per band/component
    """
    Small-multiple bubble heatmap: cluster (y-axis) x band (x-axis) per panel,
    bubble size = proportion of significant regions, colour = mean fractional
    difference across significant regions (diverging, centred at 0).

    vertical=False (default): one panel per subject, side by side (original).
    vertical=True: one panel per subject, stacked vertically.
    abbrev_clusters=True: y-tick labels are the first letter of each cluster
    name only (F, P, T, O, I, C, L, S) -- define them in the figure caption.
    cbar_location="bottom": horizontal colorbar centred below the panels
    (frees the right margin so panels can be wider).
    per_band_cnorm=True: each band/component gets its own symmetric colour
    scale and its own colorbar (labelled with its symbol) -- use when value
    ranges differ wildly between columns (e.g. aperiodic offset vs exponent),
    where a shared scale washes out the smaller-ranged component.
    """
    FS_SUBTITLE = 22
    FS_CBAR_LABEL = 20
    FS_CBAR_TICK = 18
    FS_ROW = 20

    greek_symbols = {
        'delta': r'$\mathbf{\delta}$', 'theta': r'$\mathbf{\theta}$',
        'alpha': r'$\mathbf{\alpha}$', 'beta': r'$\mathbf{\beta}$',
        'gamma': r'$\mathbf{\gamma}$', 'gamma low': r'$\mathbf{\gamma}$',
        'Offset': r'$\mathbf{b}$', 'Exponent': r'$\boldsymbol{\chi}$',
    }

    band_labels = [greek_symbols.get(b, b.capitalize()) for b in bands]

    subjects = list(cluster_summary.keys())
    if cluster_order is None:
        cluster_order = sorted({c for sub in cluster_summary.values()
                                   for band_d in sub.values() for c in band_d.keys()})

    cluster_tick_labels = [c[0].upper() for c in cluster_order] if abbrev_clusters else cluster_order
    if abbrev_clusters and len(set(cluster_tick_labels)) < len(cluster_tick_labels):
        print("WARNING: first-letter cluster abbreviations are not unique:",
              cluster_tick_labels)

    n_subs = len(subjects)
    if vertical:
        fig, axes = plt.subplots(n_subs, 1,
                                 figsize=(figsize_per_panel[0], figsize_per_panel[1]*n_subs),
                                 squeeze=False)
        axes = axes[:, 0]
    else:
        fig, axes = plt.subplots(1, n_subs,
                                 figsize=(figsize_per_panel[0]*n_subs, figsize_per_panel[1]),
                                 squeeze=False)
        axes = axes[0]

    def _sym_range(diffs):
        m = np.nanmax(np.abs(diffs)) if len(diffs) else 1.0
        return (-m, m) if m > 0 else (-1.0, 1.0)

    if per_band_cnorm:
        band_range = {}
        for band in bands:
            diffs = [d['avg_diff'] for sub in cluster_summary.values()
                     for d in sub.get(band, {}).values()
                     if not np.isnan(d['avg_diff'])]
            band_range[band] = _sym_range(diffs)
    else:
        all_diffs = [d['avg_diff'] for sub in cluster_summary.values()
                     for band_d in sub.values() for d in band_d.values()
                     if not np.isnan(d['avg_diff'])]
        vmin, vmax = _sym_range(all_diffs)

    row_positions = [i * row_spacing for i in range(len(cluster_order))]

    for ax_idx, sub_id in enumerate(subjects):
        ax = axes[ax_idx]
        is_last = ax_idx == n_subs - 1

        for row_idx, cluster in enumerate(cluster_order):
            row = row_positions[row_idx]
            for col, band in enumerate(bands):
                d = cluster_summary[sub_id].get(band, {}).get(cluster)
                if d is None:
                    continue
                size = d['prop_sig'] * max_bubble_size
                color_val = d['avg_diff'] if not np.isnan(d['avg_diff']) else 0
                b_vmin, b_vmax = band_range[band] if per_band_cnorm else (vmin, vmax)
                ax.scatter(col, row, s=max(size, 8), c=[color_val], cmap=cmap,
                           vmin=b_vmin, vmax=b_vmax, zorder=3)

        ax.set_xlim(-0.5, len(bands) - 0.5)
        ax.set_ylim(-row_spacing*0.5, row_positions[-1] + row_spacing*0.5)
        ax.set_xticks(range(len(bands)))
        ax.invert_yaxis()

        # band labels: every panel when horizontal; bottom panel only when vertical
        if (not vertical) or is_last:
            ax.set_xticklabels(band_labels, fontsize=18, fontweight='bold')
        else:
            ax.set_xticklabels([])
            ax.tick_params(bottom=False)

        # cluster labels: first panel only when horizontal; every panel when vertical
        if vertical or ax_idx == 0:
            ax.set_yticks(row_positions)
            ax.set_yticklabels(cluster_tick_labels, fontsize=FS_ROW, fontweight='bold')
        
        else:
            ax.set_yticks([])

        if show_sub_labels:
            # same precedence as the grid builders: explicit sub_labels dict wins,
            # else number by the subject's own position in pat_state (robust to
            # missing/non-contiguous subjects), else the raw id
            if sub_labels and sub_id in sub_labels:
                label = sub_labels[sub_id]
            elif pat_state and sub_id in pat_state:
                label = f"SUB{list(pat_state.keys()).index(sub_id) + 1}"
            else:
                label = sub_id
            state = f" ({pat_state[sub_id]})" if pat_state and sub_id in pat_state else ""
            loc = 'right' if vertical else 'center'
            ax.set_title(f"{label}{state}", fontsize=FS_SUBTITLE, fontweight='bold', loc=loc)

        for row in row_positions:
            ax.axhline(row, color='#eeeeee', lw=0.5, zorder=0)

        for spine in ax.spines.values():
            spine.set_visible(False)
        # divider between panels: right edge when horizontal, bottom edge when vertical
        if ax_idx < n_subs - 1:
            side = 'bottom' if vertical else 'right'
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color('#bbbbbb')
            ax.spines[side].set_linewidth(0.8)

    # layout + colorbar(s)
    def _style_cbar(cbar, lo, hi, label, n_ticks=5):
        ticks = np.linspace(lo, hi, n_ticks)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.2f}" for t in ticks])
        cbar.set_label(label, fontsize=FS_CBAR_LABEL, labelpad=10)
        cbar.ax.tick_params(labelsize=FS_CBAR_TICK, length=4, width=1)
        cbar.outline.set_visible(False)

    if per_band_cnorm:
        # one bar per band/component: side by side (bottom) or stacked (right)
        n_bars = len(bands)
        if cbar_location == "bottom":
            plt.tight_layout(rect=[0, 0.14, 1, 1])
            bar_w, gap = 0.18, 0.06
            total = n_bars * bar_w + (n_bars - 1) * gap
            x0 = (1 - total) / 2
            for i, band in enumerate(bands):
                lo, hi = band_range[band]
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi))
                sm.set_array([])
                cax = fig.add_axes([x0 + i * (bar_w + gap), 0.05, bar_w, 0.03])
                cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
                _style_cbar(cbar, lo, hi, band_labels[i], n_ticks=3)
        else:
            plt.tight_layout(rect=[0, 0, 0.90, 1])
            bar_h, gap = 0.18, 0.06
            total = n_bars * bar_h + (n_bars - 1) * gap
            y0 = (1 - total) / 2
            for i, band in enumerate(bands):
                lo, hi = band_range[band]
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi))
                sm.set_array([])
                cax = fig.add_axes([0.93, y0 + i * (bar_h + gap), 0.012, bar_h])
                cbar = fig.colorbar(sm, cax=cax)
                _style_cbar(cbar, lo, hi, band_labels[i], n_ticks=3)
    else:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])

        if cbar_location == "bottom":
            # reserve a band at the bottom instead of the right margin
            plt.tight_layout(rect=[0, 0.12, 1, 1])
            cbar_ax = fig.add_axes([0.35, 0.05, 0.30, 0.03])
            cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        else:
            plt.tight_layout(rect=[0, 0, 0.92, 1])
            if vertical:
                cbar_ax = fig.add_axes([0.94, 0.42, 0.015, 0.16])
            else:
                cbar_ax = fig.add_axes([0.94, 0.35, 0.008, 0.3])
            cbar = fig.colorbar(sm, cax=cbar_ax)

        _style_cbar(cbar, vmin, vmax, cbar_label, n_ticks=5)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)





















def build_vector_grid_pdf_dual_folder(
    image_dir_a: str,
    image_dir_b: str,
    bands: list = None,
    pat_state: list = None,
    save_dir: str = None,
    sub_labels: dict = None,
    cell_width_in: float = 4.0,
    cell_height_in: float = 1.8,
    row_label_width_in: float = 0.7,   # transpose + labels: use ~1.6 for "SUB1 (UWS)"
    col_label_height_in: float = 0.5,
    col_label_fontsize: int = 18,
    gap_x_in: float = 0.05,
    gap_y_in: float = 0.06,          # gap BETWEEN blocks (bands, or subjects if transposed)
    within_gap_y_in: float = 0.01,   # gap between the two stacked panels of the SAME pair
    row_label_fontsize: int = 24,
    row_label_vshift_in: float = 0.08,
    out_filename: str = "all_subs_all_bands_dual.pdf",
    divider_color: tuple = (0.5, 0.5, 0.5),  # medium-dark gray, RGB 0-1
    divider_width: float = 0.75,
    transpose: bool = False,        # True → subjects as rows, bands as columns
    show_sub_labels: bool = True,   # False → omit subject labels entirely
    show_band_labels: bool = True,  # False → omit band/component labels entirely
    raster_dpi: float = 200,        # rasterize panels at this DPI; None → full vector embed
):
    """
    Like build_vector_grid_pdf, but pulls one panel per (sub, band) from EACH
    of two folders and stacks them vertically (folder_a on top, folder_b
    directly below) inside every cell block.

    Default layout: bands as block-rows (Greek row labels centered across the
    stacked pair), subjects as columns. With transpose=True: subjects as
    block-rows (labels optional via show_sub_labels), bands as columns with
    Greek headers; each block-row is one subject, and divider lines separate
    subjects instead of bands. The a-over-b stacking within each block is the
    same in both orientations.

    Filenames in both folders are matched generically as:
        {sub}_<anything>_{band}.pdf

    By default (raster_dpi=200) each panel is rasterized to an image at that
    DPI and placed in its cell, while the grid labels remain real vector
    text; this keeps the merged file small and fast to render. Set
    raster_dpi=None for the old fully-vector embed of every panel.
    """
    import re
    import os
    from pathlib import Path
    import matplotlib
    import fitz  # PyMuPDF

    def _vcenter_rect(width_pt, height_pt, y0, text, fontsize, fontfile, fontname):
        tmp_doc = fitz.open()
        tmp_page = tmp_doc.new_page(width=width_pt, height=height_pt)
        rc = tmp_page.insert_textbox(
            fitz.Rect(0, 0, width_pt, height_pt),
            text, fontsize=fontsize, fontfile=fontfile, fontname=fontname,
            align=fitz.TEXT_ALIGN_CENTER,
        )
        tmp_doc.close()
        shift = max(rc, 0) / 2
        return fitz.Rect(0, y0 + shift, width_pt, y0 + height_pt + shift)

    def _build_file_index(folder):
        pattern = re.compile(r"^(?P<sub>[^_]+)_.*_(?P<band>[^_]+)\.pdf$")
        idx = {}
        for fpath in sorted(Path(folder).glob("*.pdf")):
            m = pattern.match(fpath.name)
            if m is None:
                continue
            idx.setdefault(m.group("band"), {})[m.group("sub")] = fpath
        return idx

    PT = 72

    greek_symbols = {
        'delta': '\u03b4', 'theta': '\u03b8', 'alpha': '\u03b1',
        'beta': '\u03b2', 'gamma': '\u03b3', 'gamma low': '\u03b3',
        'Offset': 'b', 'Exponent': '\u03c7',
    }

    file_index_a = _build_file_index(image_dir_a)
    file_index_b = _build_file_index(image_dir_b)

    if not file_index_a and not file_index_b:
        print(f"No matching files found in {image_dir_a} or {image_dir_b}")
        return

    all_bands_found = set(file_index_a.keys()) | set(file_index_b.keys())
    detected_bands = [b for b in bands if b in all_bands_found] if bands else sorted(all_bands_found)

    all_subs = sorted(
        {s for d in file_index_a.values() for s in d.keys()} |
        {s for d in file_index_b.values() for s in d.keys()}
    )

    # ── label text helpers ────────────────────────────────────────────────
    # ── label text helpers ────────────────────────────────────────────────
    def _band_text(band):
        return greek_symbols.get(band, band.upper())

    def _sub_text(sub):
        # explicit sub_labels wins; else number by the subject's own position
        # in pat_state (robust to missing/non-contiguous subjects); else positional
        if sub_labels and sub in sub_labels:
            sub_label = sub_labels[sub]
        elif pat_state and sub in pat_state:
            sub_label = f"SUB{list(pat_state.keys()).index(sub) + 1}"
        else:
            sub_label = f"SUB{all_subs.index(sub) + 1}"
        state = f" ({pat_state[sub]})" if pat_state and sub in pat_state else ""
        return f"{sub_label}{state}"

    # ── orientation mapping ───────────────────────────────────────────────
    if transpose:
        row_items, col_items = all_subs, detected_bands     # subs=rows, bands=cols
        row_text, col_text = _sub_text, _band_text
        show_row_labels = show_sub_labels
        show_col_labels = show_band_labels
    else:
        row_items, col_items = detected_bands, all_subs
        row_text, col_text = _band_text, _sub_text
        show_row_labels = show_band_labels
        show_col_labels = show_sub_labels


    n_rows, n_cols = len(row_items), len(col_items)

    cell_w, cell_h = cell_width_in * PT, cell_height_in * PT
    row_lbl_w = (row_label_width_in * PT) if show_row_labels else 0.0
    col_lbl_h = (col_label_height_in * PT) if show_col_labels else 0.0
    gap_x, gap_y, within_gap = gap_x_in * PT, gap_y_in * PT, within_gap_y_in * PT

    block_h = 2 * cell_h + within_gap   # height of ONE block-row's stacked pair

    total_w = row_lbl_w + n_cols * (cell_w + gap_x) - gap_x
    total_h = col_lbl_h + n_rows * block_h + (n_rows - 1) * gap_y

    out_doc = fitz.open()
    out_page = out_doc.new_page(width=total_w, height=total_h)

    font_path_bold = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf")

    # ── column headers ────────────────────────────────────────────────────
    if show_col_labels:
        col_label_vshift_in = 0.2
        for col, item in enumerate(col_items):
            x = row_lbl_w + col * (cell_w + gap_x)
            y0 = col_label_vshift_in * PT
            rc = out_page.insert_textbox(
                fitz.Rect(x, y0, x + cell_w, col_lbl_h + y0),
                col_text(item), fontsize=col_label_fontsize,
                fontfile=font_path_bold, fontname="F-bold",
                align=fitz.TEXT_ALIGN_CENTER,
            )
            if rc < 0:
                print(f"WARNING: column header {col_text(item)!r} did not fit "
                      f"(short by {-rc:.1f}pt) — increase col_label_height_in")

    # ── row labels, centered across BOTH stacked panels of the block ──────
    if show_row_labels:
        for row, item in enumerate(row_items):
            y_block = col_lbl_h + row * (block_h + gap_y)
            label = row_text(item)

            row_rect = _vcenter_rect(
                row_lbl_w, block_h, y_block + row_label_vshift_in * PT,
                label, row_label_fontsize, font_path_bold, "F-bold",
            )
            rc = out_page.insert_textbox(
                row_rect, label, fontsize=row_label_fontsize,
                fontfile=font_path_bold, fontname="F-bold",
                align=fitz.TEXT_ALIGN_CENTER,
            )
            if rc < 0:
                print(f"WARNING: row label {label!r} did not fit "
                      f"(short by {-rc:.1f}pt) — increase row_label_width_in or lower fontsize")

    # ── panels: folder_a on top, folder_b directly below, per block/column ─
    src_docs = []
    for row, row_item in enumerate(row_items):
        y_block = col_lbl_h + row * (block_h + gap_y)
        y_a = y_block
        y_b = y_a + cell_h + within_gap

        for col, col_item in enumerate(col_items):
            band, sub = (col_item, row_item) if transpose else (row_item, col_item)
            x = row_lbl_w + col * (cell_w + gap_x)

            for y_pos, index, vanchor in ((y_a, file_index_a, "bottom"), (y_b, file_index_b, "top")):
                fpath = index.get(band, {}).get(sub)
                cell_rect = fitz.Rect(x, y_pos, x + cell_w, y_pos + cell_h)

                if fpath is None:
                    out_page.insert_textbox(cell_rect, "N/A", fontsize=12,
                                             align=fitz.TEXT_ALIGN_CENTER, color=(0.5, 0.5, 0.5))
                    continue

                src_doc = fitz.open(str(fpath))
                src_page = src_doc[0]

                scale = min(cell_w / src_page.rect.width, cell_h / src_page.rect.height)
                fit_w, fit_h = src_page.rect.width * scale, src_page.rect.height * scale
                offset_x = x + (cell_w - fit_w) / 2
                if vanchor == "bottom":
                    offset_y = y_pos + (cell_h - fit_h)   # push to bottom of its cell
                else:
                    offset_y = y_pos                       # push to top of its cell
                fit_rect = fitz.Rect(offset_x, offset_y, offset_x + fit_w, offset_y + fit_h)

                if raster_dpi is None:
                    out_page.show_pdf_page(fit_rect, src_doc, 0)
                    src_docs.append(src_doc)   # must stay open until save
                else:
                    zoom = raster_dpi / 72.0
                    pix = src_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                    out_page.insert_image(fit_rect, pixmap=pix)
                    src_doc.close()

    # ── solid divider lines between blocks (bands, or subjects if transposed)
    body_left = 0
    body_right = total_w

    for row in range(1, n_rows):
        y_gap_mid = col_lbl_h + row * (block_h + gap_y) - gap_y / 2
        out_page.draw_line(
            fitz.Point(body_left, y_gap_mid),
            fitz.Point(body_right, y_gap_mid),
            color=divider_color, width=divider_width,
        )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, out_filename)
        out_doc.save(out_path, garbage=4, deflate=True)
        mode = "fully vector" if raster_dpi is None else f"raster panels @ {raster_dpi:g} dpi, vector labels"
        print(f"Saved ({mode}) → {out_path}")

    out_doc.close()
    for d in src_docs:
        d.close()





# def build_vector_grid_pdf_dual_folder(
#     image_dir_a: str,
#     image_dir_b: str,
#     bands: list = None,
#     pat_state: list = None,
#     save_dir: str = None,
#     sub_labels: dict = None,
#     cell_width_in: float = 4.0,
#     cell_height_in: float = 1.8,
#     row_label_width_in: float = 0.7,   # transpose + labels: use ~1.6 for "SUB1 (UWS)"
#     col_label_height_in: float = 0.5,
#     col_label_fontsize: int = 18,
#     gap_x_in: float = 0.05,
#     gap_y_in: float = 0.06,          # gap BETWEEN blocks (bands, or subjects if transposed)
#     within_gap_y_in: float = 0.01,   # gap between the two stacked panels of the SAME pair
#     row_label_fontsize: int = 24,
#     row_label_vshift_in: float = 0.08,
#     out_filename: str = "all_subs_all_bands_dual.pdf",
#     divider_color: tuple = (0.5, 0.5, 0.5),  # medium-dark gray, RGB 0-1
#     divider_width: float = 0.75,
#     transpose: bool = False,        # True → subjects as rows, bands as columns
#     show_sub_labels: bool = True,   # False → omit subject labels entirely
#     raster_dpi: float = 200,        # rasterize panels at this DPI; None → full vector embed
#     sub_label_position: str = "side",   # "side" | "bottom" (transpose only): labels in a strip
#                                         # under each subject's panels, above the divider — no
#                                         # left margin, full width stays with the figures
#     sub_label_height_in: float = 0.55,  # height of the bottom label strip (must exceed
#                                         # ~1.2x row_label_fontsize in points / 72)
# ):
#     """
#     Like build_vector_grid_pdf, but pulls one panel per (sub, band) from EACH
#     of two folders and stacks them vertically (folder_a on top, folder_b
#     directly below) inside every cell block.

#     Default layout: bands as block-rows (Greek row labels centered across the
#     stacked pair), subjects as columns. With transpose=True: subjects as
#     block-rows (labels optional via show_sub_labels), bands as columns with
#     Greek headers; each block-row is one subject, and divider lines separate
#     subjects instead of bands. The a-over-b stacking within each block is the
#     same in both orientations.

#     Filenames in both folders are matched generically as:
#         {sub}_<anything>_{band}.pdf

#     By default (raster_dpi=200) each panel is rasterized to an image at that
#     DPI and placed in its cell, while the grid labels remain real vector
#     text; this keeps the merged file small and fast to render. Set
#     raster_dpi=None for the old fully-vector embed of every panel.
#     """
#     import re
#     import os
#     from pathlib import Path
#     import matplotlib
#     import fitz  # PyMuPDF

#     def _vcenter_rect(width_pt, height_pt, y0, text, fontsize, fontfile, fontname):
#         tmp_doc = fitz.open()
#         tmp_page = tmp_doc.new_page(width=width_pt, height=height_pt)
#         rc = tmp_page.insert_textbox(
#             fitz.Rect(0, 0, width_pt, height_pt),
#             text, fontsize=fontsize, fontfile=fontfile, fontname=fontname,
#             align=fitz.TEXT_ALIGN_CENTER,
#         )
#         tmp_doc.close()
#         shift = max(rc, 0) / 2
#         return fitz.Rect(0, y0 + shift, width_pt, y0 + height_pt + shift)

#     def _build_file_index(folder):
#         pattern = re.compile(r"^(?P<sub>[^_]+)_.*_(?P<band>[^_]+)\.pdf$")
#         idx = {}
#         for fpath in sorted(Path(folder).glob("*.pdf")):
#             m = pattern.match(fpath.name)
#             if m is None:
#                 continue
#             idx.setdefault(m.group("band"), {})[m.group("sub")] = fpath
#         return idx

#     PT = 72

#     greek_symbols = {
#         'delta': '\u03b4', 'theta': '\u03b8', 'alpha': '\u03b1',
#         'beta': '\u03b2', 'gamma': '\u03b3', 'gamma low': '\u03b3',
#         'Offset': 'b', 'Exponent': '\u03c7',
#     }

#     file_index_a = _build_file_index(image_dir_a)
#     file_index_b = _build_file_index(image_dir_b)

#     if not file_index_a and not file_index_b:
#         print(f"No matching files found in {image_dir_a} or {image_dir_b}")
#         return

#     all_bands_found = set(file_index_a.keys()) | set(file_index_b.keys())
#     detected_bands = [b for b in bands if b in all_bands_found] if bands else sorted(all_bands_found)

#     all_subs = sorted(
#         {s for d in file_index_a.values() for s in d.keys()} |
#         {s for d in file_index_b.values() for s in d.keys()}
#     )

#     # ── label text helpers ────────────────────────────────────────────────
#     start_offset = list(pat_state.keys()).index(all_subs[0]) if pat_state else 0

#     def _band_text(band):
#         return greek_symbols.get(band, band.upper())

#     def _sub_text(sub):
#         idx = all_subs.index(sub)
#         sub_label = sub_labels[sub] if sub_labels and sub in sub_labels else f"SUB{start_offset + idx + 1}"
#         state = f" ({pat_state[sub]})" if pat_state and sub in pat_state else ""
#         return f"{sub_label}{state}"

#     # ── orientation mapping ───────────────────────────────────────────────
#     bottom_sub_labels = (sub_label_position == "bottom" and transpose and show_sub_labels)

#     if transpose:
#         row_items, col_items = all_subs, detected_bands     # subs=rows, bands=cols
#         row_text, col_text = _sub_text, _band_text
#         # with bottom labels, the side column is suppressed entirely
#         show_row_labels = show_sub_labels and not bottom_sub_labels
#         show_col_labels = True
#     else:
#         row_items, col_items = detected_bands, all_subs
#         row_text, col_text = _band_text, _sub_text
#         show_row_labels = True
#         show_col_labels = show_sub_labels
#     n_rows, n_cols = len(row_items), len(col_items)

#     cell_w, cell_h = cell_width_in * PT, cell_height_in * PT
#     row_lbl_w = (row_label_width_in * PT) if show_row_labels else 0.0
#     col_lbl_h = (col_label_height_in * PT) if show_col_labels else 0.0
#     gap_x, gap_y, within_gap = gap_x_in * PT, gap_y_in * PT, within_gap_y_in * PT

#     sub_lbl_h = (sub_label_height_in * PT) if bottom_sub_labels else 0.0
#     block_h = 2 * cell_h + within_gap + sub_lbl_h   # stacked pair (+ label strip if bottom labels)

#     total_w = row_lbl_w + n_cols * (cell_w + gap_x) - gap_x
#     total_h = col_lbl_h + n_rows * block_h + (n_rows - 1) * gap_y

#     out_doc = fitz.open()
#     out_page = out_doc.new_page(width=total_w, height=total_h)

#     font_path_bold = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf")

#     # ── column headers ────────────────────────────────────────────────────
#     if show_col_labels:
#         col_label_vshift_in = 0.2
#         for col, item in enumerate(col_items):
#             x = row_lbl_w + col * (cell_w + gap_x)
#             y0 = col_label_vshift_in * PT
#             rc = out_page.insert_textbox(
#                 fitz.Rect(x, y0, x + cell_w, col_lbl_h + y0),
#                 col_text(item), fontsize=col_label_fontsize,
#                 fontfile=font_path_bold, fontname="F-bold",
#                 align=fitz.TEXT_ALIGN_CENTER,
#             )
#             if rc < 0:
#                 print(f"WARNING: column header {col_text(item)!r} did not fit "
#                       f"(short by {-rc:.1f}pt) — increase col_label_height_in")

#     # ── row labels, centered across BOTH stacked panels of the block ──────
#     if show_row_labels:
#         for row, item in enumerate(row_items):
#             y_block = col_lbl_h + row * (block_h + gap_y)
#             label = row_text(item)

#             row_rect = _vcenter_rect(
#                 row_lbl_w, block_h, y_block + row_label_vshift_in * PT,
#                 label, row_label_fontsize, font_path_bold, "F-bold",
#             )
#             rc = out_page.insert_textbox(
#                 row_rect, label, fontsize=row_label_fontsize,
#                 fontfile=font_path_bold, fontname="F-bold",
#                 align=fitz.TEXT_ALIGN_CENTER,
#             )
#             if rc < 0:
#                 print(f"WARNING: row label {label!r} did not fit "
#                       f"(short by {-rc:.1f}pt) — increase row_label_width_in or lower fontsize")

#     # ── panels: folder_a on top, folder_b directly below, per block/column ─
#     src_docs = []
#     for row, row_item in enumerate(row_items):
#         y_block = col_lbl_h + row * (block_h + gap_y)
#         y_a = y_block
#         y_b = y_a + cell_h + within_gap

#         for col, col_item in enumerate(col_items):
#             band, sub = (col_item, row_item) if transpose else (row_item, col_item)
#             x = row_lbl_w + col * (cell_w + gap_x)

#             for y_pos, index, vanchor in ((y_a, file_index_a, "bottom"), (y_b, file_index_b, "top")):
#                 fpath = index.get(band, {}).get(sub)
#                 cell_rect = fitz.Rect(x, y_pos, x + cell_w, y_pos + cell_h)

#                 if fpath is None:
#                     out_page.insert_textbox(cell_rect, "N/A", fontsize=12,
#                                              align=fitz.TEXT_ALIGN_CENTER, color=(0.5, 0.5, 0.5))
#                     continue

#                 src_doc = fitz.open(str(fpath))
#                 src_page = src_doc[0]

#                 scale = min(cell_w / src_page.rect.width, cell_h / src_page.rect.height)
#                 fit_w, fit_h = src_page.rect.width * scale, src_page.rect.height * scale
#                 offset_x = x + (cell_w - fit_w) / 2
#                 if vanchor == "bottom":
#                     offset_y = y_pos + (cell_h - fit_h)   # push to bottom of its cell
#                 else:
#                     offset_y = y_pos                       # push to top of its cell
#                 fit_rect = fitz.Rect(offset_x, offset_y, offset_x + fit_w, offset_y + fit_h)

#                 if raster_dpi is None:
#                     out_page.show_pdf_page(fit_rect, src_doc, 0)
#                     src_docs.append(src_doc)   # must stay open until save
#                 else:
#                     zoom = raster_dpi / 72.0
#                     pix = src_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
#                     out_page.insert_image(fit_rect, pixmap=pix)
#                     src_doc.close()

#     # ── bottom subject labels: strip under each block, above the divider ──
#     if bottom_sub_labels:
#         for row, row_item in enumerate(row_items):
#             y_block = col_lbl_h + row * (block_h + gap_y)
#             y_strip = y_block + 2 * cell_h + within_gap
#             label = _sub_text(row_item)
#             rc = out_page.insert_textbox(
#                 fitz.Rect(6, y_strip, total_w, y_strip + sub_lbl_h),
#                 label, fontsize=row_label_fontsize,
#                 fontfile=font_path_bold, fontname="F-bold",
#                 align=fitz.TEXT_ALIGN_LEFT,
#             )
#             if rc < 0:
#                 print(f"WARNING: bottom subject label {label!r} did not fit "
#                       f"(short by {-rc:.1f}pt) — increase sub_label_height_in or lower fontsize")

#     # ── solid divider lines between blocks (bands, or subjects if transposed)
#     body_left = 0
#     body_right = total_w

#     for row in range(1, n_rows):
#         y_gap_mid = col_lbl_h + row * (block_h + gap_y) - gap_y / 2
#         out_page.draw_line(
#             fitz.Point(body_left, y_gap_mid),
#             fitz.Point(body_right, y_gap_mid),
#             color=divider_color, width=divider_width,
#         )

#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)
#         out_path = os.path.join(save_dir, out_filename)
#         out_doc.save(out_path, garbage=4, deflate=True)
#         mode = "fully vector" if raster_dpi is None else f"raster panels @ {raster_dpi:g} dpi, vector labels"
#         print(f"Saved ({mode}) → {out_path}")

#     out_doc.close()
#     for d in src_docs:
#         d.close()











        
region_to_anatomical_cluster = {
    # ==================== 1. PRIMARY SENSORY & MOTOR ====================
    'Precentral_L': 'Primary_Motor',
    'Precentral_R': 'Primary_Motor',
    'Supp_Motor_Area_L': 'Primary_Motor',
    'Supp_Motor_Area_R': 'Primary_Motor',
    'Rolandic_Oper_L': 'Primary_Motor',
    'Rolandic_Oper_R': 'Primary_Motor',

    'Postcentral_L': 'Primary_Somatosensory',
    'Postcentral_R': 'Primary_Somatosensory',
    'Paracentral_Lobule_L': 'Primary_Somatosensory',
    'Paracentral_Lobule_R': 'Primary_Somatosensory',

    'Heschl_L': 'Primary_Auditory',
    'Heschl_R': 'Primary_Auditory',
    'Temporal_Sup_L': 'Primary_Auditory',
    'Temporal_Sup_R': 'Primary_Auditory',

    'Calcarine_L': 'Primary_Visual',
    'Calcarine_R': 'Primary_Visual',
    'Cuneus_L': 'Primary_Visual',
    'Cuneus_R': 'Primary_Visual',
    'Lingual_L': 'Primary_Visual',
    'Lingual_R': 'Primary_Visual',
    'Occipital_Sup_L': 'Primary_Visual',
    'Occipital_Sup_R': 'Primary_Visual',
    'Occipital_Mid_L': 'Primary_Visual',
    'Occipital_Mid_R': 'Primary_Visual',
    'Occipital_Inf_L': 'Primary_Visual',
    'Occipital_Inf_R': 'Primary_Visual',
    'Fusiform_L': 'Primary_Visual',
    'Fusiform_R': 'Primary_Visual',

    # ==================== 2. ASSOCIATION & HIGHER-ORDER ====================
    # Prefrontal / Executive
    'Frontal_Sup_L': 'Prefrontal_Executive',
    'Frontal_Sup_R': 'Prefrontal_Executive',
    'Frontal_Mid_L': 'Prefrontal_Executive',
    'Frontal_Mid_R': 'Prefrontal_Executive',
    'Frontal_Inf_Oper_L': 'Prefrontal_Executive',
    'Frontal_Inf_Oper_R': 'Prefrontal_Executive',
    'Frontal_Inf_Tri_L': 'Prefrontal_Executive',
    'Frontal_Inf_Tri_R': 'Prefrontal_Executive',
    'Frontal_Sup_Medial_L': 'Prefrontal_Executive',
    'Frontal_Sup_Medial_R': 'Prefrontal_Executive',

    # Orbitofrontal / Ventromedial
    'Frontal_Sup_Orb_L': 'Orbitofrontal',
    'Frontal_Sup_Orb_R': 'Orbitofrontal',
    'Frontal_Mid_Orb_L': 'Orbitofrontal',
    'Frontal_Mid_Orb_R': 'Orbitofrontal',
    'Frontal_Inf_Orb_L': 'Orbitofrontal',
    'Frontal_Inf_Orb_R': 'Orbitofrontal',
    'Frontal_Med_Orb_L': 'Orbitofrontal',
    'Frontal_Med_Orb_R': 'Orbitofrontal',
    'Rectus_L': 'Orbitofrontal',
    'Rectus_R': 'Orbitofrontal',
    'Olfactory_L': 'Orbitofrontal',
    'Olfactory_R': 'Orbitofrontal',

    # Parietal Association
    'Parietal_Sup_L': 'Parietal_Association',
    'Parietal_Sup_R': 'Parietal_Association',
    'Parietal_Inf_L': 'Parietal_Association',
    'Parietal_Inf_R': 'Parietal_Association',
    'SupraMarginal_L': 'Parietal_Association',
    'SupraMarginal_R': 'Parietal_Association',
    'Angular_L': 'Parietal_Association',
    'Angular_R': 'Parietal_Association',
    'Precuneus_L': 'Parietal_Association',
    'Precuneus_R': 'Parietal_Association',

    # Temporal Association (beyond core auditory)
    'Temporal_Mid_L': 'Temporal_Association',
    'Temporal_Mid_R': 'Temporal_Association',
    'Temporal_Inf_L': 'Temporal_Association',
    'Temporal_Inf_R': 'Temporal_Association',
    'Temporal_Pole_Sup_L': 'Temporal_Association',
    'Temporal_Pole_Sup_R': 'Temporal_Association',
    'Temporal_Pole_Mid_L': 'Temporal_Association',
    'Temporal_Pole_Mid_R': 'Temporal_Association',

    # ==================== 3. LIMBIC & MEDIAL ====================
    'Cingulum_Ant_L': 'Cingulate',
    'Cingulum_Ant_R': 'Cingulate',
    'Cingulum_Mid_L': 'Cingulate',
    'Cingulum_Mid_R': 'Cingulate',
    'Cingulum_Post_L': 'Cingulate',
    'Cingulum_Post_R': 'Cingulate',

    'Insula_L': 'Insula',
    'Insula_R': 'Insula',

    'Hippocampus_L': 'Medial_Temporal_Limbic',
    'Hippocampus_R': 'Medial_Temporal_Limbic',
    'ParaHippocampal_L': 'Medial_Temporal_Limbic',
    'ParaHippocampal_R': 'Medial_Temporal_Limbic',
    'Amygdala_L': 'Medial_Temporal_Limbic',
    'Amygdala_R': 'Medial_Temporal_Limbic',

    # ==================== 4. SUBCORTICAL ====================
    'Caudate_L': 'Basal_Ganglia',
    'Caudate_R': 'Basal_Ganglia',
    'Putamen_L': 'Basal_Ganglia',
    'Putamen_R': 'Basal_Ganglia',
    'Pallidum_L': 'Basal_Ganglia',
    'Pallidum_R': 'Basal_Ganglia',

    'Thalamus_L': 'Thalamus',
    'Thalamus_R': 'Thalamus',
}


region_to_network = { # Taken from https://www.medrxiv.org/content/10.1101/2025.03.05.25323471v3.supplementary-material
    'Precentral_L': 'Sensorimotor Network',
    'Precentral_R': 'Sensorimotor Network',
    'Frontal_Sup_L': 'Dorsal Attention Network',
    'Frontal_Sup_R': 'Dorsal Attention Network',
    'Frontal_Sup_Orb_L': 'Limbic Network',
    'Frontal_Sup_Orb_R': 'Limbic Network',
    'Frontal_Mid_L': 'Frontoparietal Network',
    'Frontal_Mid_R': 'Frontoparietal Network',
    'Frontal_Mid_Orb_L': 'Frontoparietal Network',
    'Frontal_Mid_Orb_R': 'Frontoparietal Network',
    'Frontal_Inf_Oper_L': 'Frontoparietal Network',
    'Frontal_Inf_Oper_R': 'Frontoparietal Network',
    'Frontal_Inf_Tri_L': 'Frontoparietal Network',
    'Frontal_Inf_Tri_R': 'Frontoparietal Network',
    'Frontal_Inf_Orb_L': 'Default Mode Network',
    'Frontal_Inf_Orb_R': 'Default Mode Network',
    'Rolandic_Oper_L': 'Sensorimotor Network',
    'Rolandic_Oper_R': 'Sensorimotor Network',
    'Supp_Motor_Area_L': 'Sensorimotor Network',
    'Supp_Motor_Area_R': 'Sensorimotor Network',
    'Olfactory_L': 'Limbic Network',
    'Olfactory_R': 'Limbic Network',
    'Frontal_Sup_Medial_L': 'Default Mode Network',
    'Frontal_Sup_Medial_R': 'Default Mode Network',
    'Frontal_Med_Orb_L': 'Default Mode Network',
    'Frontal_Med_Orb_R': 'Default Mode Network',
    'Rectus_L': 'Limbic Network',
    'Rectus_R': 'Limbic Network',
    'Insula_L': 'Ventral Attention Network',
    'Insula_R': 'Ventral Attention Network',
    'Cingulum_Ant_L': 'Default Mode Network',
    'Cingulum_Ant_R': 'Default Mode Network',
    'Cingulum_Mid_L': 'Ventral Attention Network',
    'Cingulum_Mid_R': 'Ventral Attention Network',
    'Cingulum_Post_L': 'Default Mode Network',
    'Cingulum_Post_R': 'Default Mode Network',
    'Hippocampus_L': 'Limbic Network',
    'Hippocampus_R': 'Limbic Network',
    'ParaHippocampal_L': 'Default Mode Network',
    'ParaHippocampal_R': 'Default Mode Network',
    'Amygdala_L': 'Limbic Network',
    'Amygdala_R': 'Limbic Network',
    'Calcarine_L': 'Visual Network',
    'Calcarine_R': 'Visual Network',
    'Cuneus_L': 'Visual Network',
    'Cuneus_R': 'Visual Network',
    'Lingual_L': 'Visual Network',
    'Lingual_R': 'Visual Network',
    'Occipital_Sup_L': 'Visual Network',
    'Occipital_Sup_R': 'Visual Network',
    'Occipital_Mid_L': 'Visual Network',
    'Occipital_Mid_R': 'Visual Network',
    'Occipital_Inf_L': 'Visual Network',
    'Occipital_Inf_R': 'Visual Network',
    'Fusiform_L': 'Visual Network',
    'Fusiform_R': 'Visual Network',
    'Postcentral_L': 'Sensorimotor Network',
    'Postcentral_R': 'Sensorimotor Network',
    'Parietal_Sup_L': 'Dorsal Attention Network',
    'Parietal_Sup_R': 'Dorsal Attention Network',
    'Parietal_Inf_L': 'Frontoparietal Network',
    'Parietal_Inf_R': 'Frontoparietal Network',
    'SupraMarginal_L': 'Ventral Attention Network',
    'SupraMarginal_R': 'Ventral Attention Network',
    'Angular_L': 'Default Mode Network',
    'Angular_R': 'Default Mode Network',
    'Precuneus_L': 'Default Mode Network',
    'Precuneus_R': 'Default Mode Network',
    'Paracentral_Lobule_L': 'Sensorimotor Network',
    'Paracentral_Lobule_R': 'Sensorimotor Network',
    'Caudate_L': 'Basal Ganglia Network',
    'Caudate_R': 'Basal Ganglia Network',
    'Putamen_L': 'Basal Ganglia Network',
    'Putamen_R': 'Basal Ganglia Network',
    'Pallidum_L': 'Basal Ganglia Network',
    'Pallidum_R': 'Basal Ganglia Network',
    'Thalamus_L': 'Basal Ganglia Network',
    'Thalamus_R': 'Basal Ganglia Network',
    'Heschl_L': 'Sensorimotor Network',
    'Heschl_R': 'Sensorimotor Network',
    'Temporal_Sup_L': 'Sensorimotor Network',
    'Temporal_Sup_R': 'Sensorimotor Network',
    'Temporal_Pole_Sup_L': 'Limbic Network',
    'Temporal_Pole_Sup_R': 'Limbic Network',
    'Temporal_Mid_L': 'Default Mode Network',
    'Temporal_Mid_R': 'Default Mode Network',
    'Temporal_Pole_Mid_L': 'Limbic Network',
    'Temporal_Pole_Mid_R': 'Limbic Network',
    'Temporal_Inf_L': 'Limbic Network',
    'Temporal_Inf_R': 'Limbic Network',
}





region_to_anatomical_lobe = {
    # ==================== FRONTAL ====================
    'Precentral_L': 'Frontal', 'Precentral_R': 'Frontal',
    'Frontal_Sup_L': 'Frontal', 'Frontal_Sup_R': 'Frontal',
    'Frontal_Sup_Orb_L': 'Frontal', 'Frontal_Sup_Orb_R': 'Frontal',
    'Frontal_Mid_L': 'Frontal', 'Frontal_Mid_R': 'Frontal',
    'Frontal_Mid_Orb_L': 'Frontal', 'Frontal_Mid_Orb_R': 'Frontal',
    'Frontal_Inf_Oper_L': 'Frontal', 'Frontal_Inf_Oper_R': 'Frontal',
    'Frontal_Inf_Tri_L': 'Frontal', 'Frontal_Inf_Tri_R': 'Frontal',
    'Frontal_Inf_Orb_L': 'Frontal', 'Frontal_Inf_Orb_R': 'Frontal',
    'Rolandic_Oper_L': 'Frontal', 'Rolandic_Oper_R': 'Frontal',   # judgment call — see note
    'Supp_Motor_Area_L': 'Frontal', 'Supp_Motor_Area_R': 'Frontal',
    'Olfactory_L': 'Frontal', 'Olfactory_R': 'Frontal',
    'Frontal_Sup_Medial_L': 'Frontal', 'Frontal_Sup_Medial_R': 'Frontal',
    'Frontal_Med_Orb_L': 'Frontal', 'Frontal_Med_Orb_R': 'Frontal',
    'Rectus_L': 'Frontal', 'Rectus_R': 'Frontal',

    # ==================== PARIETAL ====================
    'Postcentral_L': 'Parietal', 'Postcentral_R': 'Parietal',
    'Parietal_Sup_L': 'Parietal', 'Parietal_Sup_R': 'Parietal',
    'Parietal_Inf_L': 'Parietal', 'Parietal_Inf_R': 'Parietal',
    'SupraMarginal_L': 'Parietal', 'SupraMarginal_R': 'Parietal',
    'Angular_L': 'Parietal', 'Angular_R': 'Parietal',
    'Precuneus_L': 'Parietal', 'Precuneus_R': 'Parietal',
    'Paracentral_Lobule_L': 'Parietal', 'Paracentral_Lobule_R': 'Parietal',   # judgment call — see note

    # ==================== TEMPORAL ====================
    'Heschl_L': 'Temporal', 'Heschl_R': 'Temporal',
    'Temporal_Sup_L': 'Temporal', 'Temporal_Sup_R': 'Temporal',
    'Temporal_Pole_Sup_L': 'Temporal', 'Temporal_Pole_Sup_R': 'Temporal',
    'Temporal_Mid_L': 'Temporal', 'Temporal_Mid_R': 'Temporal',
    'Temporal_Pole_Mid_L': 'Temporal', 'Temporal_Pole_Mid_R': 'Temporal',
    'Temporal_Inf_L': 'Temporal', 'Temporal_Inf_R': 'Temporal',

    # ==================== OCCIPITAL ====================
    'Calcarine_L': 'Occipital', 'Calcarine_R': 'Occipital',
    'Cuneus_L': 'Occipital', 'Cuneus_R': 'Occipital',
    'Lingual_L': 'Occipital', 'Lingual_R': 'Occipital',
    'Occipital_Sup_L': 'Occipital', 'Occipital_Sup_R': 'Occipital',
    'Occipital_Mid_L': 'Occipital', 'Occipital_Mid_R': 'Occipital',
    'Occipital_Inf_L': 'Occipital', 'Occipital_Inf_R': 'Occipital',
    'Fusiform_L': 'Occipital', 'Fusiform_R': 'Occipital',

    # ==================== INSULA ====================
    'Insula_L': 'Insula', 'Insula_R': 'Insula',

    # ==================== CINGULATE ====================
    'Cingulum_Ant_L': 'Cingulate', 'Cingulum_Ant_R': 'Cingulate',
    'Cingulum_Mid_L': 'Cingulate', 'Cingulum_Mid_R': 'Cingulate',
    'Cingulum_Post_L': 'Cingulate', 'Cingulum_Post_R': 'Cingulate',

    # ==================== LIMBIC ====================
    'Hippocampus_L': 'Limbic', 'Hippocampus_R': 'Limbic',
    'ParaHippocampal_L': 'Limbic', 'ParaHippocampal_R': 'Limbic',
    'Amygdala_L': 'Limbic', 'Amygdala_R': 'Limbic',

    # ==================== SCGM (subcortical gray matter) ====================
    'Caudate_L': 'SCGM', 'Caudate_R': 'SCGM',
    'Putamen_L': 'SCGM', 'Putamen_R': 'SCGM',
    'Pallidum_L': 'SCGM', 'Pallidum_R': 'SCGM',
    'Thalamus_L': 'SCGM', 'Thalamus_R': 'SCGM',
}