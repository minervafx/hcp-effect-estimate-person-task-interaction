#!/usr/bin/env python
"""Fetch HCP effect-estimate (COPE/beta) maps for the falsification test.

This script downloads the individual COPE dtseries files for the 4 task contrasts
of interest (plus their constituent state contrasts for the static control),
parcellates them to 379 parcels, and saves a cache parallel to the z-stat cache.

Maps are sourced from HCP Open Access bucket under the same permissions as z-stats.

[release] Path/import wiring changed (hcpee/paths.py). Every object key is checked by
hcpee.map_type_guard.assert_cope_key before it is requested, and a MAP_TYPE_MANIFEST.json
declaring map_type=COPE_effect_estimate is written next to the cache; the analysis
scripts refuse caches without it. Selection, parcellation and cache layout unchanged.
See docs/PATCHES.md.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "hcpee"))
import paths                                                          # noqa: E402
import map_type_guard                                                 # noqa: E402
import cifti
import hcp_s3
import p29 as P

RES = paths.OUT
CACHE = paths.EE_CACHE
KEYS = set()
GRAYORDINATES = 91282
F32 = 4
MIN_FREE_BYTES = 300 * 1024 * 1024

PREFIX = {"test": "HCP_1200", "retest": "HCP_Retest"}
REGISTRATION = "MSMAll"

# Contrast-to-cope-index mapping from Contrasts.txt for each task
# (1-based index as listed in Contrasts.txt)
COPE_MAP = {
    "WM": {
        "2BK-0BK": 11,
        "2BK": 9,
        "0BK": 10,
    },
    "LANGUAGE": {
        "STORY-MATH": 4,
        "STORY": 2,
        "MATH": 1,
    },
    "MOTOR": {
        "AVG-CUE": 21,
        "AVG": 7,
        "CUE": 1,
    },
    "RELATIONAL": {
        "REL-MATCH": 4,
        "REL": 2,
        "MATCH": 1,
    },
}

# Which contrasts we need for each transition (diff, B, A)
TRANSITION_COPE_NEEDS = {
    "WM_2BK_minus_0BK": {
        "task": "WM",
        "diff": "2BK-0BK",
        "B": "2BK",
        "A": "0BK",
    },
    "LANG_STORY_minus_MATH": {
        "task": "LANGUAGE",
        "diff": "STORY-MATH",
        "B": "STORY",
        "A": "MATH",
    },
    "MOTOR_AVG_minus_CUE": {
        "task": "MOTOR",
        "diff": "AVG-CUE",
        "B": "AVG",
        "A": "CUE",
    },
    "REL_REL_minus_MATCH": {
        "task": "RELATIONAL",
        "diff": "REL-MATCH",
        "B": "REL",
        "A": "MATCH",
    },
}


def level2_feat_dir(prefix, subj, task, smoothing="s2", reg=REGISTRATION):
    suf = f"_{reg}" if reg else ""
    return (f"{prefix}/{subj}/MNINonLinear/Results/tfMRI_{task}/"
            f"tfMRI_{task}_hp200_{smoothing}_level2{suf}.feat")


def cope_key(feat_dir, cope_idx):
    """Key for cope{d}.feat/cope1.dtseries.nii"""
    return f"{feat_dir}/GrayordinatesStats/cope{cope_idx}.feat/cope1.dtseries.nii"


def open_remote_header(s3, key, first=800_000):
    buf = s3.get_range(key, 0, first)
    while True:
        try:
            return cifti.open_header(buf)
        except cifti.NeedMoreBytes as e:
            buf = s3.get_range(key, 0, e.needed)


def check_geometry(s3, key, d):
    if d.n_cols != GRAYORDINATES:
        raise RuntimeError(f"{key}: {d.n_cols} grayordinates, expected {GRAYORDINATES}")
    size = s3.head_size(key)
    calc = d.data_start + d.n_rows * d.n_cols * d.itemsize
    if size is not None and calc != size:
        raise RuntimeError(f"{key}: data offset check failed (computed {calc}, object is {size})")
    return size


def fetch_cope_map(s3, key, d):
    """Fetch and extract the single map from a cope dtseries file."""
    lo, hi = d.data_range()
    raw = s3.get_range(key, lo, hi)
    if len(raw) != d.data_nbytes:
        raise RuntimeError(f"{key}: short read ({len(raw)} of {d.data_nbytes})")
    # These are 1-map dscalars (stored as dtseries with 1 volume)
    return d.extract_map(raw, 0)


def free_bytes(path=paths.EE_CACHE):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


def fetch_subject_session(s3, subj, ses, atlas_labels, n_parcels=379, verbose=True):
    """Fetch all needed cope maps for one subject-session."""
    out_path = os.path.join(CACHE, f"{subj}_{ses}.npz")
    for tname, tinfo in TRANSITION_COPE_NEEDS.items():          # [release] guard all keys
        fd = level2_feat_dir(PREFIX[ses], subj, tinfo["task"])
        for contrast in (tinfo["diff"], tinfo["B"], tinfo["A"]):
            k = cope_key(fd, COPE_MAP[tinfo["task"]][contrast])
            map_type_guard.assert_cope_key(k)
            KEYS.add(k)
    if os.path.exists(out_path):
        if verbose:
            print(f"  [{subj} {ses}] cached, skipping")
        return

    if free_bytes() < MIN_FREE_BYTES:
        raise RuntimeError("free disk below the 300 MB floor - stopping")

    store = {}
    for tname, tinfo in TRANSITION_COPE_NEEDS.items():
        task = tinfo["task"]
        feat_dir = level2_feat_dir(PREFIX[ses], subj, task)
        
        # Fetch diff, B, A cope maps
        maps = {}
        for label, contrast in [("diff", tinfo["diff"]), ("B", tinfo["B"]), ("A", tinfo["A"])]:
            cope_idx = COPE_MAP[task][contrast]
            key = cope_key(feat_dir, cope_idx)
            d = open_remote_header(s3, key)
            check_geometry(s3, key, d)
            gray = fetch_cope_map(s3, key, d)
            if not np.isfinite(gray).all() or float(gray.std()) == 0.0:
                raise RuntimeError(f"{key} [{label}]: non-finite or constant map")
            maps[label] = gray

        # Parcellate
        for label, gray in maps.items():
            parcel_data = P.parcellate(gray[None, :], atlas_labels, n_parcels)[0].astype(np.float32)
            store[f"{tname}__{label}"] = parcel_data

        # Also cache the primary grayordinate map for potential secondary analysis
        if tname == P.PRIMARY_TRANSITION:
            store["primary_grayordinate"] = maps["diff"].astype(np.float32)

    np.savez_compressed(out_path, **store)
    if verbose:
        print(f"  [{subj} {ses}]: saved {len(store)} arrays to {out_path}")


def fetch_all(subjects, atlas_labels, n_parcels=379, verbose=True):
    s3 = hcp_s3.S3()
    os.makedirs(CACHE, exist_ok=True)
    exclusions = []
    for n, subj in enumerate(subjects, 1):
        for ses in P.SESSIONS:
            try:
                fetch_subject_session(s3, subj, ses, atlas_labels, n_parcels, verbose)
            except Exception as e:
                exclusions.append(dict(subject=subj, session=ses, reason=str(e)[:300]))
                if verbose:
                    print(f"  EXCLUDE {subj} {ses}: {str(e)[:120]}")
    return s3, exclusions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true", help="Verify cope files exist without downloading")
    a = ap.parse_args()

    os.makedirs(RES, exist_ok=True)

    # Load subject list (same 42 as Phase 2.9)
    included = [l.strip() for l in open(paths.SUBJECTS_FILE) if l.strip()]
    subs = included

    labels = np.load(paths.ATLAS_LABELS)

    if a.verify_only:
        print("Verifying cope file availability...")
        s3 = hcp_s3.S3()
        missing = []
        for subj in subs[:3]:  # Check first 3 subjects
            for ses in P.SESSIONS:
                for tname, tinfo in TRANSITION_COPE_NEEDS.items():
                    task = tinfo["task"]
                    feat_dir = level2_feat_dir(PREFIX[ses], subj, task)
                    for label, contrast in [("diff", tinfo["diff"]), ("B", tinfo["B"]), ("A", tinfo["A"])]:
                        cope_idx = COPE_MAP[task][contrast]
                        key = cope_key(feat_dir, cope_idx)
                        map_type_guard.assert_cope_key(key)
                        try:
                            size = s3.head_size(key)
                            print(f"  OK: {key} ({size} bytes)")
                        except Exception as e:
                            missing.append(f"{subj} {ses} {tname} {label}: {e}")
        if missing:
            print(f"\nMISSING ({len(missing)}):")
            for m in missing:
                print(f"  {m}")
            return 1
        else:
            print("\nAll cope files verified for sample subjects.")
            return 0

    try:
        s3, exclusions = fetch_all(subs, labels, 379)
        json.dump(dict(exclusions=exclusions,
                       bytes_streamed=int(s3.bytes_transferred),
                       requests=int(s3.requests)),
                  open(os.path.join(RES, "00_fetch_log.json"), "w"), indent=2)
        print(f"Done. Streamed {s3.bytes_transferred/1e6:.1f} MB, {s3.requests} requests.")
        print(f"Exclusions: {len(exclusions)}")
        if exclusions:
            print("Manifest NOT written: exclusions present.", file=sys.stderr)
            return 3
        m = map_type_guard.write_manifest(CACHE, COPE_MAP, KEYS,
                                          producer="scripts/01_fetch_effect_estimates.py")
        print(f"Wrote {map_type_guard.MANIFEST_NAME} ({m['n_objects']} COPE objects).")
        return 0
    except hcp_s3.AccessGateError as e:
        print("ACCESS GATE - cannot fetch effect-estimate maps.", file=sys.stderr)
        print(e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())