"""Guards against substituting z-statistic maps where effect estimates are required.

Background. HCP level-2 task analyses (HCP Pipelines TaskfMRILevel2) release, for
each subject, task and contrast, both
  * a contrast-of-parameter-estimates (COPE) map - the effect estimate, in the fitted
    model's BOLD signal units:
        .../tfMRI_{TASK}_hp200_s2_level2_MSMAll.feat/GrayordinatesStats/cope{d}.feat/cope1.dtseries.nii
  * statistic maps for the same contrast (zstat1 / tstat1 / varcope1 in the same
    cope{d}.feat directory), and a merged per-task file
        .../{subject}_tfMRI_{TASK}_level2_hp200_s2_MSMAll.dscalar.nii
    whose maps are produced from zstat1 outputs, i.e. dimensionless z-statistics.

The primary analysis of the paper uses COPE effect estimates only. An earlier project
analysis used the merged z-statistic file while describing it as effect estimates; these
checks exist so that such a substitution fails loudly.

Two independent layers:
  1. Path semantics (authoritative): every remote object must match COPE_KEY_RE and must
     not name a z/t/variance statistic or the merged level-2 dscalar. The fetch script
     writes a manifest declaring map_type=COPE_effect_estimate; analysis loaders refuse
     a cache without that manifest.
  2. Value scale (heuristic, secondary): parcel-mean z-statistic contrasts in this
     cohort have per-map median |value| between 0.26 and 2.19; the matching COPE
     contrasts have 4.64-99.85 (observed in the 84 subject-session caches of each type,
     2026-09-29). A cache whose per-contrast cohort median of median |diff| falls below
     COPE_MIN_MEDIAN_ABS is rejected. This catches a relabelled z-stat cache; it is
     not a substitute for layer 1.
"""
import json
import os
import re

import numpy as np

MAP_TYPE = "COPE_effect_estimate"
MANIFEST_NAME = "MAP_TYPE_MANIFEST.json"
COPE_KEY_RE = re.compile(
    r"^HCP_(1200|Retest)/\d{6}/MNINonLinear/Results/tfMRI_(WM|LANGUAGE|MOTOR|RELATIONAL)/"
    r"tfMRI_(WM|LANGUAGE|MOTOR|RELATIONAL)_hp200_s2_level2_MSMAll\.feat/"
    r"GrayordinatesStats/cope\d+\.feat/cope1\.dtseries\.nii$")
FORBIDDEN_RE = re.compile(r"(zstat|tstat|varcope|zfstat|level2_hp200_s2_MSMAll\.dscalar)",
                          re.IGNORECASE)
COPE_MIN_MEDIAN_ABS = 3.0


class MapTypeError(RuntimeError):
    pass


def assert_cope_key(key):
    """Raise unless `key` is an HCP level-2 COPE effect-estimate object."""
    if FORBIDDEN_RE.search(key):
        raise MapTypeError(f"refusing non-effect-estimate object (statistic/merged map): {key}")
    if not COPE_KEY_RE.match(key):
        raise MapTypeError(f"object is not an HCP level-2 COPE effect-estimate path: {key}")
    return True


def write_manifest(cache_dir, cope_index_map, keys_fetched, producer):
    m = dict(map_type=MAP_TYPE,
             producer=producer,
             key_pattern=COPE_KEY_RE.pattern,
             cope_index_map=cope_index_map,
             n_objects=len(keys_fetched),
             objects=sorted(keys_fetched))
    with open(os.path.join(cache_dir, MANIFEST_NAME), "w") as fh:
        json.dump(m, fh, indent=1)
    return m


def assert_effect_estimate_cache(cache_dir, transitions, subjects=None):
    """Called by every analysis loader before reading the task cache."""
    mp = os.path.join(cache_dir, MANIFEST_NAME)
    if not os.path.exists(mp):
        raise MapTypeError(
            f"{cache_dir}: no {MANIFEST_NAME}. Only caches produced by "
            "scripts/01_fetch_effect_estimates.py are accepted as effect estimates.")
    m = json.load(open(mp))
    if m.get("map_type") != MAP_TYPE:
        raise MapTypeError(f"{mp}: map_type={m.get('map_type')!r}, expected {MAP_TYPE!r}")
    for k in m.get("objects", []):
        assert_cope_key(k)
    files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".npz"))
    if subjects is not None:
        want = {f"{s}_{ses}.npz" for s in subjects for ses in ("test", "retest")}
        missing = sorted(want - set(files))
        if missing:
            raise MapTypeError(f"{cache_dir}: missing {len(missing)} subject-session files, "
                               f"e.g. {missing[:3]}")
        files = sorted(want)
    med = {t: [] for t in transitions}
    for f in files:
        z = np.load(os.path.join(cache_dir, f))
        for t in transitions:
            med[t].append(float(np.median(np.abs(z[f"{t}__diff"]))))
    summary = {t: float(np.median(v)) for t, v in med.items()}
    low = {t: v for t, v in summary.items() if v < COPE_MIN_MEDIAN_ABS}
    if low:
        raise MapTypeError(
            f"{cache_dir}: value scale looks like z-statistics, not COPE effect estimates "
            f"(cohort median of per-map median |diff| < {COPE_MIN_MEDIAN_ABS}): {low}")
    return dict(ok=True, manifest=os.path.basename(mp), n_objects=int(m.get("n_objects", 0)),
                median_abs_diff_by_contrast=summary)
