# Person- and task-contrast-specific structure in HCP test/retest fMRI effect estimates

This repository holds the analysis code for the manuscript

> Winthrop, J. *Person- and task-contrast-specific structure in cross-session fMRI effect estimates persists after resting-state and morphometric controls.* (Manuscript; not peer reviewed.)

The analyses ask whether there is reproducible person- and task-contrast-specific structure across HCP test/retest fMRI effect estimates. Specifically: after removing each person's average profile and each contrast's average profile, can a person's residual pattern still be matched across sessions, and how much of that structure remains after specified resting-state and morphometric controls?

**No HCP data are included.** You must obtain the data yourself under the HCP Open Access Data Use Terms (see [Data access](#data-access)).

**Version.** This is v1.0.1, a bug-fix release of v1.0.0. It corrects how the anatomy block of the morphometric control is built (sulcal-depth vertex alignment and five unparsed global volumes); the primary and resting-state analyses are unchanged. v1.0.0 remains available as a historical release. See [`docs/CHANGELOG.md`](docs/CHANGELOG.md).

---

## 1. What the code computes

**Model.** For person *i*, task contrast *t* and parcel *p*, the task map is described additively as

`X(i,t) = P(i) + T(t) + I(i,t) + ε`

- *P(i)* is a generic person profile.
- *T(t)* is a generic contrast profile.
- *I(i,t)* is the person-by-contrast interaction.

Within each session separately, the code removes the person and contrast means (feature-level two-way double-centering):

`I[i,t,p,s] = X[i,t,p,s] − mean_t X − mean_i X + mean_(i,t) X`

This removes any additive person-plus-contrast profile exactly on the analysed feature scale. For each contrast, it then compares the test-session residuals with the retest-session residuals using direct cosine similarity.

**Statistics** (`hcpee/ix.py`):

- **Rank-1:** the fraction of the 168 person×contrast queries whose own-person prototype is most similar. Chance is 1/42.
- **Idiff:** 100 × (mean same-person similarity − mean different-person/same-contrast similarity), following Amico & Goñi (2018).
- **AUC:** pooled same-person similarities versus the off-diagonal similarities.
- **Permutation p:** one retest-person permutation, applied identically to all four contrasts. The run uses 10,000 draws with seed 20260823, and p = (1 + #null ≥ observed)/(1 + draws), so the minimum is 1/10001.

**Controls:**

- **Resting-state control** (`scripts/11_rest_control.py`). The REST1 functional-connectivity block (Fisher-z, 71,631 edges) has 10 principal components. They are fitted on the test session, and each session's task features are residualised against them. The script reports:
  - working-memory contrast retrieval;
  - pooled specificity (SS/SD/DS);
  - secondary rungs (k = 5, 20; REST2; a combined block; the reverse fit direction; a motion subset);
  - subject-scrambled nuisance draws.

  The pooled **interaction** after rest control (R-REST) is computed in `scripts/12_anatomy_control.py`, as the "before anatomy" state of R-REST.
- **Measured-morphometry control** (`scripts/12_anatomy_control.py`). The anatomy block has 1,366 FreeSurfer-derived features, of which 1,309 are retained after a zero-variance screen:
  - Desikan-Killiany thickness, area and curvature (2 × 34 × 3);
  - 19 subcortical volumes and six global volumes from `aseg.stats` (estimated intracranial, total gray, cortex, subcortical gray, supratentorial and brain-segmentation volume);
  - parcel means (379 labels) of corrected thickness, myelin-related contrast (MyelinMap_BC) and sulcal depth. Each surface map is placed on the atlas by its CIFTI vertex indices (`hcpee/surface_map.py`).

  The screen removes the 57 columns of the 19 subcortical labels in the three surface measures, which carry no surface data.

  The anatomy is mapped to the interaction residual by leave-one-subject-out prediction (fold-standardised PCA with k = 20, then ridge with λ = 1), separately for each session and contrast. The prediction is subtracted from the residual, and survival is re-tested. Attribution compares the Idiff decrease with 1,000 anatomy-row permutations.

**Tasks and contrasts** (HCP level-2 COPE directory index in brackets):

| Task | Contrast | COPE index |
|---|---|---|
| Working memory | 2BK − 0BK | 11 |
| Language | STORY − MATH | 4 |
| Motor | AVG − CUE | 21 |
| Relational | REL − MATCH | 4 |

The constituent-condition COPEs are also fetched and used only for secondary blocks: WM 2BK=9 / 0BK=10; LANGUAGE STORY=2 / MATH=1; MOTOR AVG=7 / CUE=1; RELATIONAL REL=2 / MATCH=1.

**Sample.** 42 HCP Young Adult test/retest participants (S1200-generation packaging, HCP_1200 = test and HCP_Retest = retest), with a published median retest interval of about five months. The list is `inputs/included_subjects.txt`. Of the 46 published retest participants, 4 were excluded in the original availability check; the reasons are in `inputs/exclusions.csv`. Family relationships were neither obtained nor inferred, and permutations are not family-blocked.

## 2. Effect estimates versus z-statistics — read this

HCP level-2 task analyses release several map types for each contrast. **This code uses only the COPE effect-estimate maps**:

```
{HCP_1200|HCP_Retest}/{subject}/MNINonLinear/Results/tfMRI_{TASK}/
  tfMRI_{TASK}_hp200_s2_level2_MSMAll.feat/GrayordinatesStats/cope{d}.feat/cope1.dtseries.nii
```

These hold the contrast of parameter estimates in the fitted model's BOLD signal scale. They are not z-statistics and not variances.

**Do not substitute** any of the following:

- `zstat1` / `tstat1` / `varcope1` in the same `cope{d}.feat` directory;
- the merged per-task file `{subject}_tfMRI_{TASK}_level2_hp200_s2_MSMAll.dscalar.nii`. Its maps are generated from `zstat1` outputs, so they are dimensionless z-statistics.

The author's initial analyses used that merged z-statistic file while describing it as effect estimates. The manuscript documents this correction, and all primary results here come from COPE maps. The earlier z-statistic values appear only as a labelled historical comparison (`expected/historical_zstat_reference.json`, used for Figure 1), and no z-statistic pipeline is shipped.

**Automated guard** (`hcpee/map_type_guard.py`). It works in two layers:

1. The fetch script refuses any object key that does not match the COPE path pattern above, or that names a z/t/variance statistic or the merged dscalar. It then writes `MAP_TYPE_MANIFEST.json` declaring `map_type = COPE_effect_estimate`. Every analysis script refuses a task cache without that manifest.
2. A secondary value-scale check rejects a cache whose contrast values look like z-statistics. This catches a relabelled z-statistic cache: on the real data, a z-statistic cache with a forged manifest is rejected.

`python tests/test_map_type_guard.py` exercises both layers on synthetic data.

## 3. Data access

You need:

1. An HCP ConnectomeDB account, with the **WU-Minn HCP Open Access Data Use Terms** accepted.
2. AWS credentials for the `hcp-openaccess` S3 bucket, issued through ConnectomeDB. Put them in `~/.hcp_aws.env` (mode 600), or in the file named by `$HCP_CRED_FILE`:
   ```
   HCP_AWS_ACCESS_KEY_ID=...
   HCP_AWS_SECRET_ACCESS_KEY=...
   ```
   The client (`hcpee/hcp_s3.py`, standard library only) signs byte-range GETs. It never prints or stores credentials. It streams only what is needed, parcellates in memory, and writes only parcel-level derived arrays.
3. No HCP Restricted Data are required, and none are used.

The HCP Open Access terms govern any redistribution of derived data. Keep your `work/` directory out of public version control; the provided `.gitignore` excludes it.

**Inputs fetched, and where they go** (the default root is `./work`, set by `HCPEE_WORK`; each location can be overridden, see `hcpee/paths.py`):

| Step | HCP objects read | Written to |
|---|---|---|
| `00_build_atlas.py --fetch` | One member of `HCP_Resources/Workbench/HCP_S1200_GroupAvg_v1.zip`, the HCP-MMP1.0 group dlabel (SHA-256 checked), plus the header of one COPE file for the 91,282-grayordinate layout | `work/atlas/labels379.npy`, `parcel_names.json` |
| `01_fetch_effect_estimates.py` | 1,008 COPE files: 42 × 2 sessions × 4 tasks × (contrast + 2 conditions) | `work/effect_estimate_cache/{subject}_{session}.npz` + `MAP_TYPE_MANIFEST.json` (≈29 MB) |
| `02_fetch_rest.py` | `rfMRI_REST{1,2}_{LR,RL}_Atlas_MSMAll_hp2000_clean.dtseries.nii` + `Movement_RelativeRMS_mean.txt`, both sessions (336 runs; ≈147 GB streamed, ≈15 min with 4 threads) | `work/rest_cache/{subject}_{session}_{run}.npz` (parcellated timeseries, ≈440 MB) |
| `03_fetch_anatomy.py` (needs the dlabel written by `00_build_atlas.py`) | `T1w/{s}/stats/{lh,rh}.aparc.stats`, `aseg.stats`; `MNINonLinear/fsaverage_LR32k/{s}.{corrThickness,MyelinMap_BC,sulc}.32k_fs_LR.dscalar.nii` | `work/anatomy/anatomy.npz`, `anatomy_retest.npz`, `anatomy_provenance{,_retest}.json` (input sizes and SHA-256) |

## 4. Setup and execution

Tested with Python 3.12.3 and numpy 2.2.6, which is the only third-party dependency.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export HCPEE_WORK=$PWD/work          # optional

python tests/test_map_type_guard.py
python tests/test_anatomy_inputs.py
python scripts/00_build_atlas.py --fetch
python tests/test_anatomy_real_data.py   # needs HCP credentials; skips without them
python scripts/01_fetch_effect_estimates.py
python scripts/02_fetch_rest.py
python scripts/03_fetch_anatomy.py
python scripts/10_interaction.py      # ~1 min
python scripts/11_rest_control.py     # ~4 min
python scripts/12_anatomy_control.py  # ~6 min
python scripts/20_verify_expected.py
python scripts/30_make_figures.py
```

`run_all.sh` runs the same sequence. Times are single-process on a 3-vCPU machine.

**Numerical reproducibility.** With numpy 2.2.6 (the pinned version) the outputs are bit-identical across our runs. With numpy 2.3.3 we observed differences of at most about 1.4e-14 in individual statistics; `scripts/20_verify_expected.py` compares at an absolute tolerance of 1e-9.

**Outputs** (in `work/outputs/`):

| File | Contents |
|---|---|
| `01_interaction_effect_estimate.json` | R-ORIG interaction (primary), per-contrast tests, bootstrap, negative controls |
| `02_rest_control.json` | Rest control: WM contrast retrieval and pooled specificity at each rung, nulls, diagnostics |
| `01_anatomy_interaction.json` | Anatomy control for R-ORIG and R-REST: survival, attribution, k×λ ladder, secondary variants |
| `20_verification.json` | Pass/fail against `expected/expected_values.json` |
| `figures/fig{1,2,3}_*.svg` | Manuscript Figures 1–3 |

The run also writes machine-readable logs `00_fetch_log.json` and `01_fetch_rest_log.json`.

## 5. Headline values

| Quantity | Value |
|---|---|
| Interaction rank-1 (R-ORIG) | 0.6130952381 (chance 0.0238095) |
| Interaction Idiff | +42.15113695 |
| Interaction AUC | 0.9031485053 |
| Rank-1 / Idiff permutation p | 1/10001 each; all four contrasts p = q = 1/10001 |
| WM contrast retrieval, before → after REST1 k=10 | 0.4523809524 → 0.2142857143; AUC 0.7431557989; p = 1/2001 |
| Pooled specificity AUC(SS,DS) after rest | 0.8641935941 |
| Pooled interaction after rest (R-REST) | rank-1 0.3988095238, Idiff +31.58981047, AUC 0.8464686688, p = 1/10001 |
| Anatomy, R-ORIG before → after | rank-1 0.6131 → 0.4821428571; Idiff 42.1511 → 34.90888454; AUC 0.9031 → 0.8685567723 |
| Anatomy, R-REST before → after | rank-1 0.3988 → 0.2738095238; Idiff 31.5898 → 26.11739034; AUC 0.8465 → 0.8062105940 |
| Idiff attenuation | 17.1816 % (R-ORIG), 17.3234 % (R-REST); attribution p = 1/1001 each |

**These are descriptive metric changes, not percentages of variance or causal fractions explained.** `scripts/20_verify_expected.py` checks all of them (36 values plus 2 derived) at an absolute tolerance of 1e-9. The anatomy rows are v1.0.1 values; v1.0.0's are listed in `docs/CHANGELOG.md`.

## 6. Scope and limitations

These are the manuscript's own boundaries:

- The results show reproducible *measured* structure in an indirect BOLD measurement under the specified controls. They do not establish a neural mechanism.
- Individual functional topography is a leading unresolved explanation. So are unmeasured morphology, vascular and neurovascular differences, and residual EPI distortion, alignment and acquisition effects.
- The resting-state and morphometric controls attenuate the structure without eliminating it. Neither rest nor anatomy is ruled out, and the residual is not "anatomy-free".
- n = 42, two sessions, one site, 42-way retrieval.
- Family structure is unavailable, so permutations are not family-blocked, and no genetic claim is possible. No twin analysis was performed, and no HCP Restricted Data were accessed.
- This repository reproduces the manuscript's fMRI analyses only and contains no EEG analysis.
- Several saved negative-control and inference caveats are described in the manuscript's Supplement S1: wrong-contrast matches are not uniformly at chance; the saved bootstrap resamples candidates; pooled specificity tests ignore within-person clustering.

## 7. Provenance

- `docs/PROVENANCE.md` maps each reported number to its script, inputs and output key.
- `docs/PATCHES.md` lists every difference from the project scripts the code was taken from. These are path and import wiring, the map-type guard, removal of unused code, and (v1.0.1) the anatomy-input correction.
- `docs/CHANGELOG.md` records the v1.0.1 correction and its effect on the reported values.

The two analysis plans were frozen locally on 2026-09-26, before the effect-estimate analyses were run. They were not deposited in an external registry. Their SHA-256 hashes are recorded in `hcpee/paths.py` and in every output. The documents themselves are not part of this repository.

**Naming in the code.** Module docstrings and output keys keep the project's historical labels:
- "Phase 2.9 / 3.0 / 3.1 / 3.2" and "Run 11 / 12" name the original analysis stages.
- Keys such as `WM_2BK_minus_0BK` and `per_transition` use "transition". The manuscript calls these *task contrasts*: condition comparisons, not time-resolved changes of state.
- Docstrings also mention project files that are not included, such as self-tests, plans and the EEG pipeline.

These labels were kept so the vendored modules stay byte-identical to the code that produced the reported outputs.

## 8. Citation and acknowledgement

See `CITATION.cff`. If you use HCP data, include the HCP acknowledgement:

> Data were provided [in part] by the Human Connectome Project, WU-Minn Consortium (Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657) funded by the 16 NIH Institutes and Centers that support the NIH Blueprint for Neuroscience Research; and by the McDonnell Center for Systems Neuroscience at Washington University.

The HCP-MMP1.0 parcellation is from Glasser et al. (2016), *Nature* 536:171–178.

## 9. AI assistance

Large-language-model coding agents (principally Anthropic's Claude via Claude Code, with OpenAI Codex also used) helped write, debug, audit and package this code, including the v1.0.1 correction. The author directed the work and is responsible for it.

## 10. License

The code is released under the MIT License (see `LICENSE`). The license covers this code only; HCP data remain governed by the HCP Data Use Terms.
