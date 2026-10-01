# Provenance of the reported numbers

Each value reported in the manuscript (v2) comes from one output file of this repository. `expected/expected_values.json` lists the exact JSON path for each, and `scripts/20_verify_expected.py` checks them.

## Chain

```
HCP Open Access (hcp-openaccess S3)
 ├─ HCP_S1200_GroupAvg_v1.zip :: HCP-MMP1.0 group dlabel ──► 00_build_atlas.py ──► labels379.npy (379 parcels)
 ├─ level-2 COPE maps: cope{d}.feat/cope1.dtseries.nii ──────► 01_fetch_effect_estimates.py ──► effect_estimate_cache/ (+ MAP_TYPE_MANIFEST.json)
 ├─ rfMRI_REST{1,2}_{LR,RL}_Atlas_MSMAll_hp2000_clean ────────► 02_fetch_rest.py ──► rest_cache/
 └─ FreeSurfer stats + 32k corrThickness/MyelinMap_BC/sulc ────► 03_fetch_anatomy.py ──► anatomy/anatomy{,_retest}.npz
                                                                    │
 effect_estimate_cache ─────────────────────────────────────────────┼─► 10_interaction.py ──► 01_interaction_effect_estimate.json
 effect_estimate_cache + rest_cache ────────────────────────────────┼─► 11_rest_control.py ──► 02_rest_control.json
 effect_estimate_cache + rest_cache + anatomy ──────────────────────┴─► 12_anatomy_control.py ──► 01_anatomy_interaction.json
                                                                        (R-ORIG and R-REST, before and after anatomy)
 outputs ──► 20_verify_expected.py (checks against the manuscript) ──► 30_make_figures.py (Figures 1–3)
```

## Manuscript element → source

| Manuscript element | Script | Output key(s) |
|---|---|---|
| §3.1: effect-estimate interaction rank-1, Idiff, AUC, p; per-contrast rank-1, p, q | `10_interaction.py` | `R_ORIG.{rank1,idiff,auc,p_rank1_vs_null_person,p_idiff_vs_null_person,per_transition}` |
| Table 1, effect-estimate rows | R-ORIG: `10_interaction.py`; R-REST: `12_anatomy_control.py` | `R_ORIG.*`; `R_REST.before.*` |
| Table 1, historical z-statistic rows; Figure 1 z bars | *not computed here* | `expected/historical_zstat_reference.json` (labelled historical) |
| §3.2: WM contrast retrieval before/after rest; AUC; p = 1/2001 | `11_rest_control.py` | `baseline.transfer.*`; `primary_REST1.k10.transfer.*` |
| Table 2: pooled SS/SD/DS/AUC before and after rest | `11_rest_control.py` | `baseline.specificity.*`; `primary_REST1.k10.specificity.pooled.*` |
| §3.2: pooled interaction after rest (0.3988 …) | `12_anatomy_control.py` | `R_REST.before.*` |
| §3.2: subject-scrambled nuisance interval | `11_rest_control.py` | `NC2_permuted_nuisance.k10` |
| §3.3 / Table 3: anatomy before/after, attribution | `12_anatomy_control.py` | `R_ORIG.{before,after,attribution}`, `R_REST.{before,after,attribution}` |
| Table 4: per-contrast rank-1 before/after anatomy | `12_anatomy_control.py` | `R_*.{before,after}.per_transition` |
| §3.3: k × λ ladder; out-of-fold R² | `12_anatomy_control.py` | `NC6_ladder`, `R_*.cv_r2_*` |
| Supplement S1 | `10`, `11`, `12` | bootstrap `rank1_ci` / `idiff_ci`; NC controls; `secondary`; `diag_*` |
| Figures 1–3 | `30_make_figures.py` | reads the three outputs plus the historical z reference |

The repository contains no EEG analysis. The historical z-statistic analyses (Table 1 z rows, Supplement S2) are **not** reproduced by this repository; only three z-statistic scalars are shipped, as a labelled reference for Figure 1.

## Inclusion list

`inputs/included_subjects.txt` holds 42 IDs: the 46 published HCP retest participants minus the 4 in `inputs/exclusions.csv`. Those four were excluded in the project's original availability check because a required level-2 map was missing for at least one task and session. The effect-estimate analyses reuse the same 42 IDs in the same order, as their frozen plan requires. For all 42, `01_fetch_effect_estimates.py` retrieved every required COPE object with no exclusions.

## Seeds and settings

- Seed 20260823 throughout.
- 10,000 subject permutations for interaction survival, specificity and per-contrast tests.
- 2,000 permutations in the inherited rest-transfer function.
- 1,000 anatomy-row shuffles for primary attribution, and 200 for secondary cells.
- 10,000 bootstrap draws.
- Rest: PCA k = 10 on the test session. Anatomy: k = 20, λ = 1, leave-one-subject-out.
