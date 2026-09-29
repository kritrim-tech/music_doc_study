import numpy as np
from scipy.signal import resample
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import mne
from helpers.extract_data import crop_to_fixed_length
from helpers.frequency_analysis import compute_psd_mne
import copy
import fooof
import matplotlib.patches as mpatches


import matplotlib.image as mpimg
from matplotlib.colors import ListedColormap
from nilearn.image import new_img_like
from nilearn.plotting import plot_img_on_surf
import os
import re
from pathlib import Path
import pickle as pkl
from fooof import FOOOFGroup






def get_fooof_bands(eeg_fooof, freqs, psds, bands):    
    fooof_bands_absolute = {}
    fooof_bands_relative = {}
    # channels = 128

    for pt in eeg_fooof:
        fooof_bands_relative[pt] = {}
        fooof_bands_absolute[pt] = {}

        for parad_indx in eeg_fooof[pt]:
            # fooof_bands[pt][parad_indx] = {}
            fooof_periodic_parad = eeg_fooof[pt][parad_indx][2]
            # freqs = all_psds[pt][parad_indx][1]
            # psds = all_psds[pt][parad_indx][0]
            fooof_band_powers_absolute = {band: np.zeros(psds.shape[0]) for band in bands}

            for band, (fmin, fmax) in bands.items():
                band_idx = np.where((freqs >= fmin) & (freqs < fmax))[0]
                fooof_periodic_parad_ = np.nanmean(fooof_periodic_parad, axis=2) #mean over segments to then compute the integral over the freqs
                fooof_band_powers_absolute[band][:] = np.trapezoid(fooof_periodic_parad_[:, band_idx], freqs[band_idx], axis=1)

            fooof_total_power_bands = np.sum(np.array(list(fooof_band_powers_absolute.values())), axis=0)  # np.sum is applied on an array of size (5 bands, 128 channels, 159 segments)
            fooof_band_powers_relative = {band: fooof_band_powers_absolute[band] / fooof_total_power_bands for band in fooof_band_powers_absolute} 
            fooof_bands_relative[pt][parad_indx] = fooof_band_powers_relative
            fooof_bands_absolute[pt][parad_indx] = fooof_band_powers_absolute

    return fooof_bands_relative, fooof_bands_absolute




def get_fooof_aperiodic(eeg_fooof):
    fooof_aperiodic = {}
    for pt in eeg_fooof:
        fooof_aperiodic[pt] = {}
        for parad in eeg_fooof[pt]:
            n_seg = eeg_fooof[pt][parad][2].shape[2]      # matched epoch count
            ap = eeg_fooof[pt][parad][1][:, :n_seg, :]    # (n_ch, n_seg, 2); NaN where fit failed
            fooof_aperiodic[pt][parad] = {"Offset": ap[..., 0], "Exponent": ap[..., 1]}
    return fooof_aperiodic


# def get_fooof_aperiodic(eeg_fooof):    
#     fooof_aperiodic = {}

#     for pt in eeg_fooof:
#         fooof_aperiodic[pt] = {}

#         for parad in eeg_fooof[pt]:
#             segments = eeg_fooof[pt][parad][2].shape[2]
#             channels = eeg_fooof[pt][parad][2].shape[0]
#             aperiodic_params = {"Offset": np.zeros([channels, segments]), "Exponent": np.zeros([channels, segments])}
            
#             for ch in range(channels):
#                 for seg in range(segments):
#                     try:
#                         offset_b_aperiodic = eeg_fooof[pt][parad][1][ch, seg].get_params("aperiodic_params")[0]
#                         exponent_k_aperiodic = eeg_fooof[pt][parad][1][ch, seg].get_params("aperiodic_params")[1]
#                         aperiodic_params["Offset"][ch, seg] = offset_b_aperiodic
#                         aperiodic_params["Exponent"][ch, seg] = exponent_k_aperiodic
#                     except:
#                         print("No model fit results are available to extract, can not proceed.")
#                         aperiodic_params["Offset"][ch, seg] = 0
#                         aperiodic_params["Exponent"][ch, seg] = 0

#             fooof_aperiodic[pt][parad] = aperiodic_params

#     return fooof_aperiodic















def match_epoch_length(array1: np.ndarray, array2: np.ndarray, verbose: bool = True):
    """
    Matches the length of two 3D arrays along the 3rd dimension (epochs).
    The longer array is truncated from the END (last epochs are dropped).
    
    Parameters
    ----------
    array1, array2 : np.ndarray
        Both must be 3D (shape = (n_channels, n_times, n_epochs))
    verbose : bool
        Whether to print a message when truncation happens
    
    Returns
    -------
    array1_matched, array2_matched : tuple of np.ndarray
        Both now have the same shape in the 3rd dimension
    """
    if array1.ndim != 3 or array2.ndim != 3:
        raise ValueError(f"Both arrays must be 3D. Got shapes {array1.shape} and {array2.shape}")

    # # Optional safety check (first two dimensions should usually match)
    # if array1.shape[:2] != array2.shape[:2]:
    #     raise ValueError(f"First two dimensions don't match: {array1.shape[:2]} vs {array2.shape[:2]}")

    len1 = array1.shape[2]
    len2 = array2.shape[2]

    if len1 == len2:
        if verbose:
            print(f"✓ Lengths already match: {len1} epochs")
        return array1, array2

    min_len = min(len1, len2)

    if len1 > len2:
        matched1 = array1[:, :, :min_len]
        matched2 = array2
        if verbose:
            print(f"Truncated array1 from {len1} → {min_len} epochs (dropped last {len1-min_len})")
    else:
        matched1 = array1
        matched2 = array2[:, :, :min_len]
        if verbose:
            print(f"Truncated array2 from {len2} → {min_len} epochs (dropped last {len2-min_len})")

    return matched1, matched2





def get_matched_legths(eeg_fooof):
    eeg_fooof_matched = {sub: {} for sub in eeg_fooof}

    for sub in eeg_fooof:
        paradigms = list(eeg_fooof[sub].keys())
        print(f"Subject: {sub}, Paradigms: {paradigms}")

        arr_parad1 = eeg_fooof[sub][paradigms[0]][2]
        arr_parad2 = eeg_fooof[sub][paradigms[1]][2]

        arr_parad1_matched, arr_parad2_matched = match_epoch_length(arr_parad1, arr_parad2)

        eeg_fooof_matched[sub][paradigms[0]] = eeg_fooof[sub][paradigms[0]]
        eeg_fooof_matched[sub][paradigms[1]] = eeg_fooof[sub][paradigms[1]]

        eeg_fooof_matched[sub][paradigms[0]][2] = arr_parad1_matched
        eeg_fooof_matched[sub][paradigms[1]][2] = arr_parad2_matched

        print(eeg_fooof_matched[sub][paradigms[0]][2].shape, eeg_fooof_matched[sub][paradigms[1]][2].shape)

    return eeg_fooof_matched




def get_fooof(all_psds, freq_range=(0.5, 45), n_jobs=-1, keep_models=False, save_path=None):
    """FOOOF per channel/region per epoch, fitted over freq_range only.
    Failed fits -> NaN (logged). Returns per condition:
    [psd, aperiodic_params (n_ch, n_seg, 2) or FOOOFGroup, fooof_periodic, freqs_fit]
    """

    eeg_fooof = {}

    for pt in all_psds:
        eeg_fooof[pt] = {}
        for parad in all_psds[pt]:
            psd   = all_psds[pt][parad][0]          # (n_ch, n_freq, n_seg)
            freqs = all_psds[pt][parad][1]
            n_ch, n_freq, n_seg = psd.shape
            print(f"{pt} | {parad} | {n_ch} channels x {n_seg} segments")

            # all spectra of this condition as one (n_ch*n_seg, n_freq) batch
            spectra = psd.transpose(0, 2, 1).reshape(n_ch * n_seg, n_freq)

            fg = FOOOFGroup(peak_width_limits=[1, 8], max_n_peaks=8, verbose=False)
            fg.fit(freqs, spectra, freq_range=list(freq_range), n_jobs=n_jobs)

            freqs_fit = fg.freqs                     # grid actually fitted (trimmed)
            fmask = (freqs >= freqs_fit[0]) & (freqs <= freqs_fit[-1])

            ap_params = fg.get_params('aperiodic_params')   # (n_spec, 2); NaN if failed
            failed = np.isnan(ap_params).any(axis=1)
            if failed.any():
                bad = np.argwhere(failed.reshape(n_ch, n_seg))
                print(f"  !! {failed.sum()} failed fits at (ch, seg): {bad.tolist()[:20]}")

            # vectorized periodic spectrum: log10(psd) - (offset - exponent*log10(f))
            periodic = np.full((n_ch * n_seg, freqs_fit.size), np.nan)
            ok = ~failed
            log_psd = np.log10(spectra[:, fmask])
            offs, exps = ap_params[ok, 0], ap_params[ok, 1]
            periodic[ok] = log_psd[ok] - (offs[:, None] - exps[:, None] * np.log10(freqs_fit)[None, :])

            fooof_periodic = periodic.reshape(n_ch, n_seg, -1).transpose(0, 2, 1)
            ap_params = ap_params.reshape(n_ch, n_seg, 2)

            out = [psd, (fg if keep_models else ap_params), fooof_periodic, freqs_fit]
            eeg_fooof[pt][parad] = out

            if save_path:
                with open(f"{save_path}eeg_fooof_{pt}{parad}.pkl", "wb") as f:
                    pkl.dump(out, f)

    return eeg_fooof





def _pdf_page_to_array(pdf_path, target_width_in, target_height_in, oversample=2, page_num=0):
    """
    Render a PDF page to a numpy array, sized so its native pixel resolution
    comfortably covers the target display size (in inches) at typical
    print/screen density, avoiding any post-hoc upscaling blur.
    """
    import fitz
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    page_w_in = page.rect.width / 72
    page_h_in = page.rect.height / 72

    # Zoom needed so the rendered array covers the target cell size at
    # ~oversample x the typical print resolution (e.g., 2x150 = 300 "effective" dpi)
    base_dpi = 150
    zoom_x = (target_width_in * base_dpi * oversample) / (page_w_in * 72)
    zoom_y = (target_height_in * base_dpi * oversample) / (page_h_in * 72)
    zoom = max(zoom_x, zoom_y)  # preserve aspect ratio, cover the larger requirement

    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    doc.close()
    return img


def plot_significance_single_sub_band(
    fdr_corrected_p_vals,
    aal_img,
    label_region_mapping,
    sub_id: str,
    band: str,
    parad1: str,
    parad2: str,
    pvals: bool = True,
    view: str = 'lateral',
    save_dir: str = None,
    cmap_color=ListedColormap(["#EDF4F6", "#7DB17B"]),
):
    """
    Plot brain data for ONE subject and ONE band.

    Parameters
    ----------
    pvals : bool
        If True  → binary significance map (p < 0.05), no colorbar, black/white.
        If False → raw values from dictionary plotted with colorbar.
    """
    if sub_id not in fdr_corrected_p_vals:
        raise KeyError(f"Subject {sub_id} not found")
    
    if band not in fdr_corrected_p_vals[sub_id]:
        raise KeyError(f"Band {band} not found for subject {sub_id}")

    values = np.asarray(fdr_corrected_p_vals[sub_id][band]).ravel()

    # ── Build brain volume ────────────────────────────────────────────────────
    aal_data   = np.squeeze(np.asanyarray(aal_img.dataobj)).astype(int)
    brain_data = np.zeros(aal_data.shape)
    region_ids = list(label_region_mapping.keys())[:90]

    if pvals:
        # Binary significance mask
        plot_values = (values < 0.05).astype(float)
        colorbar    = False
        vmin, vmax  = 0, 1
        threshold   = None
        cmap        = cmap_color  
    else:
        # Raw values as-is
        plot_values = values
        colorbar    = True
        vmin, vmax  = np.nanmin(values), np.nanmax(values)
        threshold   = None
        cmap        = cmap_color  

    for j, region_id in enumerate(region_ids):
        brain_data[aal_data == region_id] = plot_values[j]

    img = new_img_like(aal_img, brain_data)

    # ── Plot ──────────────────────────────────────────────────────────────────
    # plot_img_on_surf(
    #     img,
    #     views=[view],
    #     hemispheres=["left", "right"],
    #     colorbar=colorbar,
    #     cmap=cmap,
    #     title=f"{sub_id} — {band.upper()}",
    #     vmin=vmin,
    #     vmax=vmax,
    #     threshold=threshold,
    #     bg_on_data=True
    # )

# ── Plot ──────────────────────────────────────────────────────────────────
    plot_img_on_surf(
        img,
        views=[view],
        hemispheres=["left", "right"],
        colorbar=False,          # always disable nilearn's colorbar
        cmap=cmap,
        title="",
        vmin=vmin,
        vmax=vmax,
        threshold=threshold,
        bg_on_data=True
    )


    # Modify title fontsize after the fact
    # fig = plt.gcf()
    # fig.suptitle(f"{band.upper()}", fontsize=20, fontweight='bold')  # adjust size as needed
    fig = plt.gcf()
    for ax in fig.axes:
        for coll in ax.collections:
            coll.set_rasterized(True)

    if not pvals:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=fig.axes, fraction=0.02, pad=0.02, shrink=0.5)
        cbar.solids.set_rasterized(True)
        ticks = np.linspace(vmin, vmax, 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.3f}" for t in ticks])
        cbar.ax.tick_params(labelsize=18)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname     = f"{sub_id}_{parad1}_{parad2}_{band}.pdf"
        save_path = os.path.join(save_dir, fname)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    plt.close()







def _safe_savefig(fig, output_path, retries=3, delay=1.0, **savefig_kwargs):
    import time
    import os
    base, ext = os.path.splitext(output_path)
    tmp_path = f"{base}_tmp{ext}"   # e.g. "...Rest_Reward_tmp.pdf" — keeps the real extension

    fig.savefig(tmp_path, **savefig_kwargs)

    for attempt in range(retries):
        try:
            os.replace(tmp_path, output_path)
            return
        except PermissionError:
            if attempt < retries - 1:
                print(f"'{output_path}' is locked, retrying in {delay}s ...")
                time.sleep(delay)
            else:
                raise PermissionError(
                    f"Could not overwrite '{output_path}' — close any program "
                    f"(PDF viewer, OneDrive sync) that may have it open, then retry."
                )
            













def build_vector_grid_pdf(
    image_dir: str,
    parad1: str,
    parad2: str,
    bands: list = None,
    pat_state: list = None,
    save_dir: str = None,
    sub_labels: dict = None,
    cell_width_in: float = 4.0,
    cell_height_in: float = 1.8,        # ← try shrinking this to reduce empty space
    row_label_width_in: float = 0.7,
    col_label_height_in: float = 0.5,
    col_label_fontsize: int = 18,
    gap_x_in: float = 0.05,
    gap_y_in: float = 0.015,
    row_label_fontsize: int = 22,
    row_label_vshift_in: float = 0.08,
    out_filename: str = None,
):
    """
    True-vector grid: subject/band PDF panels + text labels, no rasterization.
    """
    import re
    import os
    from pathlib import Path
    import matplotlib
    import fitz  # PyMuPDF

    def _vcenter_rect(width_pt, height_pt, y0, text, fontsize, fontfile, fontname):
        """Return a rect that vertically centers `text` within a height_pt-tall
        band starting at y0, using PyMuPDF's documented measure-then-shift technique."""
        import fitz  # PyMuPDF

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

    PT = 72
    image_dir = Path(image_dir)

    pattern = re.compile(
        rf"^(?P<sub>[^_]+(?:_[^_]+)*)_{re.escape(parad1)}_{re.escape(parad2)}_(?P<band>[^.]+)\.pdf$"
    )

    greek_symbols = {
        'delta': '\u03b4', 'theta': '\u03b8', 'alpha': '\u03b1',
        'beta': '\u03b2', 'gamma': '\u03b3', 'gamma low': '\u03b3',
        'Offset': 'b', 'Exponent': '\u03c7',
    }

    file_index = {}
    for fpath in sorted(image_dir.glob("*.pdf")):
        m = pattern.match(fpath.name)
        if m is None:
            continue
        file_index.setdefault(m.group("band"), {})[m.group("sub")] = fpath

    if not file_index:
        print(f"No matching files found in {image_dir}")
        return

    detected_bands = [b for b in bands if b in file_index] if bands else sorted(file_index.keys())
    all_subs = sorted({s for d in file_index.values() for s in d.keys()})
    n_bands, n_subs = len(detected_bands), len(all_subs)

    cell_w, cell_h = cell_width_in * PT, cell_height_in * PT
    row_lbl_w, col_lbl_h = row_label_width_in * PT, col_label_height_in * PT
    gap_x, gap_y = gap_x_in * PT, gap_y_in * PT

    total_w = row_lbl_w + n_subs * (cell_w + gap_x) - gap_x
    total_h = col_lbl_h + n_bands * (cell_h + gap_y) - gap_y

    out_doc = fitz.open()
    out_page = out_doc.new_page(width=total_w, height=total_h)

    font_path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
    font_path_bold = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf")

    # ── column headers (subjects) ─────────────────────────────────────────

    col_label_vshift_in = 0.2
    for col, sub in enumerate(all_subs):
        x = row_lbl_w + col * (cell_w + gap_x)
        sub_label = sub_labels[sub] if sub_labels and sub in sub_labels else f"SUB{col + 1}"
        state = f" ({pat_state[sub]})" if pat_state and sub in pat_state else ""
        text = f"{sub_label}{state}"
        y0 = col_label_vshift_in * PT
        out_page.insert_textbox(
            fitz.Rect(x, y0, x + cell_w, col_lbl_h + y0),
            text, fontsize=col_label_fontsize, fontfile=font_path_bold, fontname="F-bold",
            align=fitz.TEXT_ALIGN_CENTER,
        )
    # ── row labels (bands / params, Greek), vertically centered per row ────
    for row, band in enumerate(detected_bands):
        y = col_lbl_h + row * (cell_h + gap_y)
        label = greek_symbols.get(band, band.upper())

        row_rect = _vcenter_rect(
            row_lbl_w, cell_h, y + row_label_vshift_in * PT,
            label, row_label_fontsize, font_path_bold, "F-bold",
        )
        out_page.insert_textbox(
            row_rect, label, fontsize=row_label_fontsize,
            fontfile=font_path_bold, fontname="F-bold",
            align=fitz.TEXT_ALIGN_CENTER,
        )

    # ── panels: full vector embed via show_pdf_page ───────────────────────
    src_docs = []
    for row, band in enumerate(detected_bands):
        for col, sub in enumerate(all_subs):
            fpath = file_index.get(band, {}).get(sub)
            x = row_lbl_w + col * (cell_w + gap_x)
            y = col_lbl_h + row * (cell_h + gap_y)
            cell_rect = fitz.Rect(x, y, x + cell_w, y + cell_h)

            if fpath is None:
                out_page.insert_textbox(cell_rect, "N/A", fontsize=12,
                                         align=fitz.TEXT_ALIGN_CENTER, color=(0.5, 0.5, 0.5))
                continue

            src_doc = fitz.open(str(fpath))
            src_docs.append(src_doc)
            src_page = src_doc[0]

            scale = min(cell_w / src_page.rect.width, cell_h / src_page.rect.height)
            fit_w, fit_h = src_page.rect.width * scale, src_page.rect.height * scale
            offset_x = x + (cell_w - fit_w) / 2
            offset_y = y + (cell_h - fit_h) / 2
            fit_rect = fitz.Rect(offset_x, offset_y, offset_x + fit_w, offset_y + fit_h)

            out_page.show_pdf_page(fit_rect, src_doc, 0)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        if out_filename is None:
            out_filename = f"all_subs_all_bands_{parad1}_{parad2}.pdf"
        out_path = os.path.join(save_dir, out_filename)
        out_doc.save(out_path, garbage=4, deflate=True)

    out_doc.close()
    for d in src_docs:
        d.close()




def plot_relative_difference_single_sub_band(
    power_dict,
    aal_img,
    label_region_mapping,
    cmap_color,
    sub_id: str,
    band: str,
    condition1: str,
    condition2: str,
    view: str = 'lateral',
    save_dir: str = None,
    symmetric_cmap: bool = True,
):
    """
    Plot (condition2 - condition1) / condition1 relative power difference
    for ONE subject and ONE band, in the same visual style as
    plot_significance_single_sub_band's raw-value (pvals=False) mode.

    Parameters
    ----------
    power_dict : dict
        Nested as power_dict[sub_id][condition][band] -> array of shape (n_regions,)
    symmetric_cmap : bool
        If True, colorbar limits are set symmetrically around zero, so 0%
        change sits at the colormap's center — recommended for a diverging
        colormap (e.g. 'coolwarm', 'RdBu_r').
    """
    if sub_id not in power_dict:
        raise KeyError(f"Subject {sub_id} not found")
    if condition1 not in power_dict[sub_id] or condition2 not in power_dict[sub_id]:
        raise KeyError(f"Condition(s) not found for subject {sub_id}")
    if band not in power_dict[sub_id][condition1] or band not in power_dict[sub_id][condition2]:
        raise KeyError(f"Band {band} not found for subject {sub_id}")

    values1 = np.asarray(power_dict[sub_id][condition1][band])
    values2 = np.asarray(power_dict[sub_id][condition2][band])

    # Average across epochs if present (shape (90, n_epochs) -> (90,))
    if values1.ndim == 2:
        values1 = np.nanmean(values1, axis=1)
    if values2.ndim == 2:
        values2 = np.nanmean(values2, axis=1)

    values1 = values1.ravel()
    values2 = values2.ravel()

    # Safe relative difference: avoid divide-by-zero blowups
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = np.where(values1 != 0, (values2 - values1) / values1, np.nan)

    # ── Build brain volume ────────────────────────────────────────────────
    aal_data   = np.squeeze(np.asanyarray(aal_img.dataobj)).astype(int)
    brain_data = np.zeros(aal_data.shape)
    region_ids = list(label_region_mapping.keys())[:90]

    for j, region_id in enumerate(region_ids):
        brain_data[aal_data == region_id] = rel_diff[j]

    img = new_img_like(aal_img, brain_data)

    if symmetric_cmap:
        max_abs = np.nanmax(np.abs(rel_diff))
        vmin, vmax = -max_abs, max_abs
    else:
        vmin, vmax = np.nanmin(rel_diff), np.nanmax(rel_diff)

    # ── Plot ─────────────────────────────────────────────────────────────
    plot_img_on_surf(
        img,
        views=[view],
        hemispheres=["left", "right"],
        colorbar=False,
        cmap=cmap_color,
        title="",
        vmin=vmin,
        vmax=vmax,
        threshold=None,
        bg_on_data=True
    )

    fig = plt.gcf()
    fig = plt.gcf()

    # rasterize the surface meshes (and colorbar gradient); text stays vector
    for ax in fig.axes:
        for coll in ax.collections:
            coll.set_rasterized(True)

    sm = plt.cm.ScalarMappable(cmap=cmap_color, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, fraction=0.02, pad=0.02, shrink=0.5)
    cbar.solids.set_rasterized(True)
    ticks = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks])
    cbar.ax.tick_params(labelsize=18)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"{sub_id}_{condition1}_{condition2}_{band}.pdf"
        save_path = os.path.join(save_dir, fname)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")

    plt.show()
    plt.close()
