# Changelog

## v1.0.1 — morphometric-input correction (bug-fix release)

v1.0.1 corrects two implementation defects in how v1.0.0 built the FreeSurfer-derived anatomy block for the measured-morphometry control (`scripts/03_fetch_anatomy.py`). The primary effect-estimate analysis and the resting-state control do not use that block and are unchanged. v1.0.0 remains available as a historical release (Zenodo DOI 10.5281/zenodo.23089967).

The defects were found by an independent pre-submission audit. That audit (2026-10-01) checked them against the HCP file metadata of all 84 subject-sessions before any code was changed.

### 1. Sulcal-depth features were not aligned to the parcel atlas

- **What the files contain.** HCP `{subject}.sulc.32k_fs_LR.dscalar.nii` has 64,984 columns: all 32,492 vertices of each hemisphere (`VertexIndices` 0–32491), medial wall included.
- **What the analysis frame expects.** The 91,282-grayordinate frame and the HCP-MMP1.0 atlas list 59,412 cortical vertices (left 29,696, right 29,716), with the medial wall excluded.
- **The defect.** v1.0.0 copied every surface map into that frame by position (`full[:gray.size] = gray`). For sulc, each "parcel mean" therefore averaged the wrong vertices, and values spilled into six subcortical labels.
- **Extent.** 366 of the 1,137 block-B columns were affected: 360 cortical sulc parcels plus 6 subcortical columns that should have been empty.
- **Thickness and myelin were not affected.** `corrThickness` and `MyelinMap_BC` list exactly the atlas's 59,412 vertices in atlas order (84/84 files), so the positional copy was correct for them, and their features are unchanged.
- **The fix.** Every surface map is now placed by vertex identity through both files' CIFTI `BrainModel` `VertexIndices` (`hcpee/surface_map.py`).
- **Independent check.** On four checked subject-sessions, corrected subject sulcal depth correlates r = 0.90–0.93 across the 360 cortical parcels with the HCP S1200 group-average sulc map; the v1.0.0 features gave r = −0.03 to 0.02.

### 2. Five of the six global volumes were fill values

- **How `aseg.stats` writes measures.** Lines read `# Measure BrainSeg, BrainSegVol, Brain Segmentation Volume, <value>, mm^3`.
- **The defect.** v1.0.0 stored each measure under its first field and then looked up `TotalGrayVol`, `CortexVol`, `SubCortGrayVol`, `SupraTentorialVol` and `BrainSegVol`, which are second-field names. Only `EstimatedTotalIntraCranialVol` (a first-field name) was found.
- **The fill.** The five misses became NaN, and the builder replaced every NaN with the participant's mean over the remaining block-A features. The five columns were therefore identical to each other within each participant.
- **The fix.** Each global quantity is now matched by its verified `(first, second)` field pair (`ASEG_GLOBAL_FIELDS`), must occur exactly once, and any missing or non-finite block-A value raises instead of being filled.

### Effect on features and results

- **Feature count.** The block still has 1,366 candidate features. The zero-variance screen now removes 57 columns (the 19 subcortical labels of each of the three surface measures, which carry no surface data), leaving **1,309** retained features (v1.0.0: 51 removed, 1,315 retained).
- **Morphometric control** (k = 20, λ = 1, leave-one-subject-out; 10,000 subject permutations; 1,000 anatomy shuffles):

| Quantity | v1.0.0 | v1.0.1 |
|---|---|---|
| R-ORIG rank-1 after anatomy | 0.4940476190 | 0.4821428571 |
| R-ORIG Idiff after anatomy | 35.40935298 | 34.90888454 |
| R-ORIG AUC after anatomy | 0.8653092334 | 0.8685567723 |
| R-ORIG Idiff attenuation | 15.9943% | 17.1816% |
| R-ORIG attribution p (Idiff; rank-1) | 1/1001; 2/1001 | 1/1001; 1/1001 |
| R-REST rank-1 after anatomy | 0.2559523810 | 0.2738095238 |
| R-REST Idiff after anatomy | 26.16110323 | 26.11739034 |
| R-REST AUC after anatomy | 0.8023555459 | 0.8062105940 |
| R-REST Idiff attenuation | 17.1850% | 17.3234% |
| R-REST attribution p (Idiff; rank-1) | 1/1001; 1/1001 | 1/1001; 1/1001 |

- **Survival.** Survival p after anatomy remains 1/10001 for rank-1 and Idiff in both spaces.
- **Unchanged outputs.** `01_interaction_effect_estimate.json` and `02_rest_control.json` are bit-identical to v1.0.0 (numpy 2.2.6). The "before anatomy" states in `01_anatomy_interaction.json` are also unchanged.

### Other changes

- **`scripts/00_build_atlas.py`.** It now checks that the task CIFTI's cortical `VertexIndices` equal the dlabel's. The label vector written is unchanged (SHA-256 `0e361613…`).
- **`scripts/03_fetch_anatomy.py`.** It requires the atlas dlabel at `$HCPEE_ATLAS_DLABEL`, and writes `anatomy_provenance{,_retest}.json` with the size and SHA-256 of every input file.
- **`scripts/30_make_figures.py`.** The Figure 3 footnote is computed from the outputs rather than fixed text; v1.0.0 outputs still give byte-identical v1.0.0 figures.
- **New tests.** `tests/test_anatomy_inputs.py` (no data needed) and `tests/test_anatomy_real_data.py` (HCP credentials; skips without them). Both fail if either defect is re-introduced.
- **`expected/expected_values.json`.** Updated morphometric values, plus five additional checks (after-anatomy survival p values and the R-REST rank-1 attribution p).
- **Documentation.** README, `docs/PROVENANCE.md`, `docs/PATCHES.md` and `docs/REPRODUCIBILITY.md` are updated; `CITATION.cff` carries version and repository fields.

The corrected morphometric analysis is an implementation correction of the intended analysis. It was run after the v1.0.0 outcomes were known; the model, its parameters and all inference settings are those of v1.0.0.

## v1.0.0 — initial public release (2026-10-01)

Initial release of the analysis code: interaction retrieval, resting-state and morphometric controls, the COPE map-type guard, and verification against the reported values. Zenodo DOI 10.5281/zenodo.23089967 (all versions: 10.5281/zenodo.23089966). Its morphometric block contains the two defects described above.
