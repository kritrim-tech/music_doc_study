# Sensitivity check: group-level network results without FDR correction

**Date:** 2026-10-22
**Notebook:** `source_level_psd_analysis_cluster_based.ipynb`
**Purpose:** Verify that the group-level null results at the network level are not
an artifact of FDR strictness, by inspecting the raw (uncorrected) bootstrap
p-values from `group_level_bootstrap_test_network` for all three comparisons.

## Method note

- `raw_pvals` is the first return value of `group_level_bootstrap_test_network`
  (helpers/group_level_helpers.py) and is uncorrected by construction; no code
  changes were needed to obtain it.
- During the same session, FDR was additionally bypassed for exploration by
  monkeypatching `multipletests` / `fdrcorrection` in the helper modules
  (helpers.topo_eeg, helpers.source_level_psd_helpers,
  helpers.group_level_helpers). Any tables or figures generated in that session
  are therefore UNCORRECTED and were discarded. The kernel was restarted
  afterwards, restoring FDR for all manuscript outputs.
- Group test: nonparametric bootstrap paired t-test with pooled resampling
  (Dwivedi et al., 2017), 1000 resamples, seed 42, subject as unit of analysis.
  3 networks (SMN, DMN, LN) x 5 bands = 15 tests per comparison, 45 in total.

## Results

### 1. Rest vs. continuous music (MCS- subgroup, n = 5)

```
delta      Sensorimotor Network      p = 0.2920
delta      Default Mode Network      p = 0.7680
delta      Limbic Network            p = 0.9400
theta      Sensorimotor Network      p = 0.6590
theta      Default Mode Network      p = 0.9110
theta      Limbic Network            p = 0.9630
alpha      Sensorimotor Network      p = 0.7190
alpha      Default Mode Network      p = 0.7640
alpha      Limbic Network            p = 0.8550
beta       Sensorimotor Network      p = 0.9880
beta       Default Mode Network      p = 0.9970
beta       Limbic Network            p = 0.8490
gamma low  Sensorimotor Network      p = 0.6820
gamma low  Default Mode Network      p = 0.8360
gamma low  Limbic Network            p = 0.7030
```

All 15 raw p >= 0.29. The network-band combinations that were individually
significant in every one of the five subjects sit at raw p ~ 0.68-0.86 at the
group level. No correction scheme could render any of these significant: the
group-level null reflects cancellation of opposite-signed within-subject
effects (Sub5 mirror-image profile), not correction strictness or marginal
power. **Basis for the Results sentence:** "In the rest versus continuous music
comparison, no network-band combination approached significance at the group
level even before correction for multiple comparisons (all uncorrected
p >= 0.29), including the combinations significant in every individual subject."

### 2. Segmented music vs. segmented pink noise (n = 4)

```
delta      Sensorimotor Network      p = 0.1420
delta      Default Mode Network      p = 0.1110
delta      Limbic Network            p = 0.4520
theta      Sensorimotor Network      p = 0.4540
theta      Default Mode Network      p = 0.8999
theta      Limbic Network            p = 0.5130
alpha      Sensorimotor Network      p = 0.4320
alpha      Default Mode Network      p = 0.1050
alpha      Limbic Network            p = 0.4780
beta       Sensorimotor Network      p = 0.6530
beta       Default Mode Network      p = 0.0080  <-- p < 0.05 uncorrected
beta       Limbic Network            p = 0.3073
gamma low  Sensorimotor Network      p = 0.1730
gamma low  Default Mode Network      p = 0.3520
gamma low  Limbic Network            p = 0.1280
```

One uncorrected hit (beta, DMN, p = 0.008): significant in no individual
subject, nonsignificant after BH-FDR across the 15 tests (corrected p ~ 0.12),
consistent with the manuscript's reported 0/3 group-level results for this
comparison. Same character as the isolated beta-band channel reported in the
manuscript for this comparison (group-level only, no individual significance).
Treated as a chance-level finding; not reported in the manuscript.

### 3. Continuous music vs. continuous pink noise (n = 3)

```
delta      Sensorimotor Network      p = 0.0995
delta      Default Mode Network      p = 0.8036
delta      Limbic Network            p = 0.9799
theta      Sensorimotor Network      p = 0.3330
theta      Default Mode Network      p = 0.3360
theta      Limbic Network            p = 0.3283
alpha      Sensorimotor Network      p = 0.7769
alpha      Default Mode Network      p = 0.3367
alpha      Limbic Network            p = 0.3357
beta       Sensorimotor Network      p = 0.8475
beta       Default Mode Network      p = 0.6908
beta       Limbic Network            p = 0.5000
gamma low  Sensorimotor Network      p = 0.8693
gamma low  Default Mode Network      p = 0.4884
gamma low  Limbic Network            p = 0.3293
```

No uncorrected significance; minimum raw p = 0.0995 (delta, SMN). Note the
limited resolution at n = 3.

## Overall conclusion

Across 45 uncorrected network-level group tests, 1 fell below 0.05 versus
~2.25 expected by chance at alpha = 0.05. The group-level network null results
are robust to the removal of multiple-comparison correction. Manuscript
footprint: one sentence in the Results (group-level section), scoped to the
rest vs. continuous music comparison. Everything else in this file is held for
potential reviewer responses.