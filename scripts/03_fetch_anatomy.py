#!/usr/bin/env python
"""Fetch the FreeSurfer-derived anatomy blocks used by the morphometric control.

Block A (per subject-session): Desikan-Killiany thickness / area / mean curvature
from lh/rh.aparc.stats, 19 subcortical volumes and 6 global measures from aseg.stats.
Block B: parcel means (379-label atlas) of corrThickness, MyelinMap_BC and sulc
32k_fs_LR dscalars. Both come from HCP Open Access structural outputs.

Writes $HCPEE_ANAT_DIR/anatomy.npz (test = HCP_1200) and anatomy_retest.npz
(retest = HCP_Retest), each with arrays `subjects`, `blockA`, `blockB`.

[release] The functions below are copied verbatim from the project's historical
fetch script (src/phase2_9/02_fetch.py: open_remote_header, parse_aparc, parse_aseg,
ANAT_SUBCORT, ANAT_GLOBAL, fetch_anatomy); only this header, the path wiring and the
small main() are new. See docs/PATCHES.md.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "hcpee"))
import paths                                                          # noqa: E402
import cifti                                                          # noqa: E402
import hcp_s3                                                         # noqa: E402
import p29 as P                                                       # noqa: E402

GRAYORDINATES = 91282
PREFIX = {"test": "HCP_1200", "retest": "HCP_Retest"}
CACHE = paths.ANAT_DIR


def open_remote_header(s3, key, first=800_000):
    """Parse a remote CIFTI header, widening the range request if the XML needs it.

    HCP level-2 dscalars carry a ~650-720 KB CIFTI XML extension AND write
    vox_offset = 0, so the data offset has to be derived by walking the extension
    list. Both facts were discovered against the real objects; the synthetic
    fixtures in 99_selftest.py wrote a well-formed vox_offset and never exercised
    this path.
    """
    buf = s3.get_range(key, 0, first)
    while True:
        try:
            return cifti.open_header(buf)
        except cifti.NeedMoreBytes as e:
            buf = s3.get_range(key, 0, e.needed)


ANATOMY_KEYS = [
    ("{p}/{s}/T1w/{s}/stats/lh.aparc.stats", 6_000,
     "Desikan-Killiany: 34 left-hemisphere parcels x (thickness, area, volume, curv)"),
    ("{p}/{s}/T1w/{s}/stats/rh.aparc.stats", 6_000,
     "Desikan-Killiany: 34 right-hemisphere parcels, same measures"),
    ("{p}/{s}/T1w/{s}/stats/aseg.stats", 4_000,
     "subcortical volumes + intracranial volume"),
    ("{p}/{s}/MNINonLinear/fsaverage_LR32k/{s}.corrThickness.32k_fs_LR.dscalar.nii",
     265_000, "vertexwise cortical thickness on the common mesh"),
    ("{p}/{s}/MNINonLinear/fsaverage_LR32k/{s}.MyelinMap_BC.32k_fs_LR.dscalar.nii",
     265_000, "T1w/T2w myelin map on the common mesh"),
    ("{p}/{s}/MNINonLinear/fsaverage_LR32k/{s}.sulc.32k_fs_LR.dscalar.nii",
     265_000, "sulcal depth (cortical folding) on the common mesh"),
]


def parse_aparc(text):
    """Desikan-Killiany per-parcel stats from a FreeSurfer ?h.aparc.stats file."""
    rows = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        if len(f) >= 10:
            # StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv ...
            rows[f[0]] = dict(area=float(f[2]), volume=float(f[3]),
                              thickness=float(f[4]), curv=float(f[6]))
    return rows


def parse_aseg(text):
    """Subcortical volumes plus the global measures from aseg.stats."""
    vols, glob = {}, {}
    for line in text.splitlines():
        if line.startswith("# Measure"):
            f = [x.strip() for x in line[len("# Measure"):].split(",")]
            if len(f) >= 4:
                try:
                    glob[f[0]] = float(f[3])
                except ValueError:
                    pass
        elif not line.startswith("#") and line.strip():
            f = line.split()
            if len(f) >= 5:
                try:
                    vols[f[4]] = float(f[3])
                except ValueError:
                    pass
    return vols, glob


ANAT_SUBCORT = ["Left-Thalamus-Proper", "Right-Thalamus-Proper", "Left-Caudate",
                "Right-Caudate", "Left-Putamen", "Right-Putamen", "Left-Pallidum",
                "Right-Pallidum", "Left-Hippocampus", "Right-Hippocampus",
                "Left-Amygdala", "Right-Amygdala", "Left-Accumbens-area",
                "Right-Accumbens-area", "Left-VentralDC", "Right-VentralDC",
                "Brain-Stem", "Left-Cerebellum-Cortex", "Right-Cerebellum-Cortex"]
ANAT_GLOBAL = ["EstimatedTotalIntraCranialVol", "TotalGrayVol", "CortexVol",
               "SubCortGrayVol", "SupraTentorialVol", "BrainSegVol"]


def fetch_anatomy(subjects, atlas_labels, n_parcels=379, verbose=True, session="test"):
    """PART J blocks A and B. Compact by construction.

    The pre-registered nuisance model uses the TEST session only, and that is what the
    PCA is fitted on and what every residualisation uses - unchanged.  `session` is
    additionally called with "retest" so that pre-registration section 6 question 1,
    "how strongly does anatomy ALONE identify subjects", can be answered by genuine
    cross-session anatomical retrieval instead of a degenerate self-match.  That is an
    ADDITION to the declared fetch, not a substitution: no nuisance regression is ever
    fitted on retest anatomy.
    """
    s3 = hcp_s3.S3()
    suffix = "" if session == "test" else f"_{session}"
    out_path = os.path.join(CACHE, f"anatomy{suffix}.npz")  # CACHE = paths.ANAT_DIR
    if os.path.exists(out_path):
        return s3, out_path, []
    blockA, blockB, kept, missing = {}, {}, [], []
    for subj in subjects:
        p = PREFIX[session]
        try:
            lh = parse_aparc(s3.get(f"{p}/{subj}/T1w/{subj}/stats/lh.aparc.stats")
                             .decode("utf8", "replace"))
            rh = parse_aparc(s3.get(f"{p}/{subj}/T1w/{subj}/stats/rh.aparc.stats")
                             .decode("utf8", "replace"))
            vols, glob = parse_aseg(
                s3.get(f"{p}/{subj}/T1w/{subj}/stats/aseg.stats").decode("utf8", "replace"))
            names = sorted(set(lh) & set(rh))
            vec = []
            for h in (lh, rh):
                for nm in names:
                    vec += [h[nm]["thickness"], h[nm]["area"], h[nm]["curv"]]
            vec += [vols.get(k, np.nan) for k in ANAT_SUBCORT]
            vec += [glob.get(k, np.nan) for k in ANAT_GLOBAL]
            a = np.asarray(vec, np.float32)

            surf = []
            for meas in ("corrThickness", "MyelinMap_BC", "sulc"):
                key = (f"{p}/{subj}/MNINonLinear/fsaverage_LR32k/"
                       f"{subj}.{meas}.32k_fs_LR.dscalar.nii")
                d = open_remote_header(s3, key, first=600_000)
                lo, hi = d.data_range()
                gray = d.extract_map(s3.get_range(key, lo, hi), 0)
                # these are cortex-only (59,412); pad to the 91,282 grayordinate frame
                full = np.full(GRAYORDINATES, np.nan, np.float32)
                full[:gray.size] = gray
                surf.append(P.parcellate(full, atlas_labels, n_parcels)[0])
            b = np.concatenate(surf).astype(np.float32)
            if not np.isfinite(a).all():
                a = np.nan_to_num(a, nan=float(np.nanmean(a)))
            b = np.nan_to_num(b, nan=0.0)
            blockA[subj], blockB[subj] = a, b
            kept.append(subj)
            if verbose:
                print(f"  anat {subj}: A={a.size} B={b.size} "
                      f"({s3.bytes_transferred / 1e6:.1f} MB)")
        except Exception as e:
            missing.append(dict(subject=subj, reason=str(e)[:200]))
            if verbose:
                print(f"  anat EXCLUDE {subj}: {str(e)[:120]}")
    np.savez_compressed(out_path, subjects=np.array(kept),
                        blockA=np.stack([blockA[s] for s in kept]),
                        blockB=np.stack([blockB[s] for s in kept]))
    return s3, out_path, missing


def main():
    os.makedirs(CACHE, exist_ok=True)
    subs = [l.strip() for l in open(paths.SUBJECTS_FILE) if l.strip()]
    labels = np.load(paths.ATLAS_LABELS)
    for ses in ("test", "retest"):
        s3, path, missing = fetch_anatomy(subs, labels, 379, session=ses)
        print(f"{ses}: {path}  missing={len(missing)}  "
              f"streamed {s3.bytes_transferred / 1e6:.1f} MB")
        if missing:
            print(missing)
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
