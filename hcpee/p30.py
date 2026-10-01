#!/usr/bin/env python
"""Phase 3.0 shared helpers - resting-state static-functional control.

Fresh implementation.  Phase 2.9 modules are imported READ-ONLY for the estimators the
pre-registration requires to be identical (`p29.transfer`, `p29.specificity`,
`p29.residualise`, `p29.df_matched_null`, `p29.pca_fit_test_only`, `p29.parcellate`) and
for S3/CIFTI access.  Nothing under src/phase2_9 is modified.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cifti                                                          # noqa: E402
import hcp_s3                                                         # noqa: E402
import p29 as P29                                                     # noqa: E402

import paths                                                         # noqa: E402

CACHE = paths.REST_CACHE          # parcellated resting-state timeseries

N_PARCELS = 379
GRAYORDINATES = 91282
MIN_TIMEPOINTS = 900                       # pre-registration X3
PREFIX = {"test": "HCP_1200", "retest": "HCP_Retest"}
RUN_GROUPS = {"REST1": ("rfMRI_REST1_LR", "rfMRI_REST1_RL"),
              "REST2": ("rfMRI_REST2_LR", "rfMRI_REST2_RL")}
ALL_RUNS = tuple(r for g in ("REST1", "REST2") for r in RUN_GROUPS[g])
SEED = P29.SEED


def run_key(session, subject, run):
    return (f"{PREFIX[session]}/{subject}/MNINonLinear/Results/{run}/"
            f"{run}_Atlas_MSMAll_hp2000_clean.dtseries.nii")


def motion_key(session, subject, run):
    return (f"{PREFIX[session]}/{subject}/MNINonLinear/Results/{run}/"
            f"Movement_RelativeRMS_mean.txt")


def subjects():
    return [l.strip() for l in open(paths.SUBJECTS_FILE) if l.strip()]


def atlas_labels():
    return np.load(paths.ATLAS_LABELS)


# ------------------------------------------------------------------ streaming
def open_remote_header(s3, key, first=800_000):
    """Parse the CIFTI-2 header of a remote dtseries with as few requests as possible."""
    buf = s3.get_range(key, 0, first)
    while True:
        try:
            return cifti.open_header(buf)
        except cifti.NeedMoreBytes as e:
            buf = s3.get_range(key, 0, e.needed)


def stream_parcellate(s3, key, labels, n_parcels=N_PARCELS, chunk_gray=8192):
    """Return (T, 379) parcel-mean timeseries, streaming the file in grayordinate blocks.

    The CIFTI-2 on-disk order is (n_grayordinates, n_timepoints) with TIME fastest, so a
    byte range is a contiguous block of grayordinates carrying all of their timepoints.
    That lets the 439 MB dense file be reduced to 379 parcel means without ever holding
    it whole, in memory or on disk.
    """
    d = open_remote_header(s3, key)
    T, G = int(d.n_rows), int(d.n_cols)
    if G != GRAYORDINATES:
        raise ValueError(f"{key}: {G} grayordinates, expected {GRAYORDINATES}")
    if labels.shape[0] != G:
        raise ValueError(f"atlas has {labels.shape[0]} labels, file has {G}")
    item = d.itemsize
    start = d.data_start
    total = np.zeros((n_parcels, T), np.float64)
    count = np.zeros(n_parcels, np.int64)
    for lo in range(0, G, chunk_gray):
        hi = min(lo + chunk_gray, G)
        raw = s3.get_range(key, start + lo * T * item, start + hi * T * item)
        want = (hi - lo) * T
        if len(raw) < want * item:
            raise RuntimeError(f"{key}: short read at grayordinate {lo}")
        blk = np.frombuffer(raw, dtype=d.dtype, count=want).reshape(hi - lo, T)
        lab = labels[lo:hi]
        for p in range(1, n_parcels + 1):
            m = lab == p
            if m.any():
                total[p - 1] += blk[m].sum(0, dtype=np.float64)
                count[p - 1] += int(m.sum())
    if (count == 0).any():
        raise RuntimeError(f"{key}: {(count == 0).sum()} empty parcels in atlas")
    return (total / count[:, None]).T.astype(np.float32), T      # (T, 379)


# ---------------------------------------------------------- FC representation
def fc_vector(ts):
    """(T, 379) parcel timeseries -> Fisher-z upper triangle (71631,).

    Pre-registration section 4.2 steps 3-6: per-parcel temporal z-score, Pearson
    correlation, Fisher r-to-z, upper triangle k=1.
    """
    X = np.asarray(ts, np.float64)
    sd = X.std(0)
    if not np.isfinite(X).all():
        raise ValueError("non-finite parcel timeseries")
    if (sd <= 0).any():
        raise ValueError(f"{int((sd <= 0).sum())} parcels with zero temporal variance")
    Z = (X - X.mean(0)) / sd
    C = (Z.T @ Z) / (X.shape[0] - 1)
    np.fill_diagonal(C, 0.0)
    C = np.clip(C, -0.999999, 0.999999)
    iu = np.triu_indices(C.shape[0], 1)
    return np.arctanh(C)[iu].astype(np.float32)


# [release] load_task() (reader of the historical z-statistic task cache) removed; the
# effect-estimate scripts load COPE caches through hcpee.map_type_guard-checked loaders.
