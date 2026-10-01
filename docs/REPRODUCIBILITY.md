# Reproducibility check performed before release (2026-09-29)

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
