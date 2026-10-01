# Reproducibility checks

## v1.0.1 (2026-10-01)

Three runs of the v1.0.1 code were compared. All used Python 3.12.3 and numpy 2.2.6 unless noted.

1. **Official corrected run.**
   - Inputs: the anatomy block was freshly fetched with the corrected `03_fetch_anatomy.py`. The COPE and resting-state caches were the 2026-09-29 retrievals, which are array-identical to the original inputs (see below).
   - Results: `20_verify_expected.py` reports ALL MATCH (38 checks).
   - `01_interaction_effect_estimate.json` (248 values) and `02_rest_control.json` (3,415 values) are identical to the outputs that produced the manuscript's values. Only run-time and bookkeeping fields differ.
   - `01_anatomy_interaction.json` differs intentionally, in the morphometric values (see `docs/CHANGELOG.md`). Its "before anatomy" states are unchanged.
2. **Clean full run.**
   - A fresh clone of the release candidate, a new virtual environment built only from `requirements.txt`, an empty work directory, and `run_all.sh` end to end.
   - Every input was retrieved again: atlas; 1,008 COPE objects with 0 exclusions; 336 rest runs requested, 335 retrieved, 1 unavailable as originally, 146.95 GB streamed; anatomy for 42 × 2 subject-sessions with none missing.
   - Inputs compared with run 1: 1,092/1,092 COPE arrays equal, 335/335 rest timeseries equal, anatomy blocks equal, and the SHA-256 of every anatomy input file equal.
   - Outputs: every value of all three output files is identical to run 1, and the three figures are byte-identical. Tests passed, and ALL MATCH (38 checks).
3. **Cross-host run.**
   - Run 1's inputs, a different machine (2-vCPU DigitalOcean AMD instead of 3-vCPU AMD EPYC-Rome), and numpy 2.2.6.
   - `01_interaction_effect_estimate.json` is identical. `02_rest_control.json` and `01_anatomy_interaction.json` differ only by floating-point rounding (maximum absolute difference 2.3e-14).
   - ALL MATCH (38 checks).

Earlier, with numpy 2.3.3 on the original machine, statistics differed from the numpy 2.2.6 outputs by at most about 1.4e-14. The verifier's absolute tolerance is 1e-9.

**Anatomy-input tests.** The tests passed in all runs; on real data, corrected sulcal depth correlates r = 0.932 with the HCP group-average sulc map, against −0.033 for the v1.0.0 construction.

## v1.0.0 (2026-09-29)


The released code was run end to end on freshly retrieved HCP inputs, in a clean virtual environment built only from `requirements.txt` (Python 3.12.3, numpy 2.2.6). The results were compared with the outputs that produced the manuscript's numbers.

## Fresh inputs

Each input was retrieved again from the HCP Open Access bucket with this repository's scripts and compared with the inputs originally used.

| Input | Fresh retrieval | Comparison with the original inputs |
|---|---|---|
| 379-label atlas (`00_build_atlas.py --fetch`) | dlabel SHA-256 `f3315dd2…`; labels written | `labels379.npy` byte-identical (SHA-256 `0e361613…`); parcel names identical |
| COPE effect-estimate cache (`01_fetch_effect_estimates.py`) | 1,008 COPE objects, 1.17 GB streamed, 0 exclusions, manifest written | 84/84 subject-session files; **1,092/1,092 arrays exactly equal** |
| Resting-state cache (`02_fetch_rest.py`) | 336 runs requested, 335 retrieved, 1 unavailable (as originally), 147 GB streamed | **335/335 timeseries exactly equal**, motion summaries equal |
| Anatomy (`03_fetch_anatomy.py`) | Both sessions, 0 missing | `subjects`, `blockA`, `blockB` exactly equal, test and retest |

## Fresh analysis outputs

| Output | Values compared | Result |
|---|---|---|
| `01_interaction_effect_estimate.json` | 243 | all bit-identical |
| `02_rest_control.json` | 3,414 | all bit-identical |
| `01_anatomy_interaction.json` | 2,539 | all bit-identical |
| `20_verify_expected.py` | 31 manuscript values + 2 derived attenuations | ALL MATCH |
| Figures 1–3 (`30_make_figures.py`) | SVG files | byte-identical to the manuscript figures |

The comparison excluded only run-time and bookkeeping fields: elapsed seconds, the new map-type-guard record, and the analysis-plan hashes. The plan hash equals the recorded hash when the plan documents are supplied, and is null otherwise.

## Map-type guard on real data

- The original z-statistic task cache (84 files) was presented to the analysis loader and **rejected** because it had no manifest.
- With a forged `COPE_effect_estimate` manifest it was **rejected again**, by the value-scale check. The per-contrast cohort median of median |diff| was 0.40–1.16, below the 3.0 floor. The fresh COPE cache gives 8.76–34.53.

## Running times

Measured on 3 vCPUs.

- Interaction: ≈1 min.
- Rest control: ≈10 min when run concurrently with the anatomy control.
- Anatomy control: ≈12 min.
- Peak resident memory of the interaction script: 79 MB. Memory was not measured for the other scripts.
