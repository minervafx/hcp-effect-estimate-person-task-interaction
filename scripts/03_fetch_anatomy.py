#!/usr/bin/env python
"""Fetch the FreeSurfer-derived anatomy blocks used by the morphometric control.

Block A (per subject-session, 229 values): Desikan-Killiany thickness / area / mean
curvature from lh/rh.aparc.stats (2 x 34 x 3 = 204), 19 subcortical volumes and six
global measures from aseg.stats.
Block B (3 x 379 values): parcel means (379-label atlas) of the corrThickness,
MyelinMap_BC and sulc 32k_fs_LR dscalars. The 19 subcortical labels have no surface
data; those columns are 0 for every subject and are removed by the analysis's
zero-variance screen. All inputs are HCP Open Access structural outputs.

Writes $HCPEE_ANAT_DIR/anatomy.npz (test = HCP_1200) and anatomy_retest.npz
(retest = HCP_Retest), each with arrays `subjects`, `blockA`, `blockB`, plus
anatomy_provenance{,_retest}.json (object keys, sizes and SHA-256 of every input file).

Requires the atlas dlabel at $HCPEE_ATLAS_DLABEL (written by 00_build_atlas.py --fetch):
its cortical VertexIndices define where each surface vertex goes.

[release v1.0.1] Corrected. v1.0.0 shipped the historical fetch functions verbatim
(src/phase2_9/02_fetch.py), which had two input defects:
  1. Sulcal depth. HCP {s}.sulc.32k_fs_LR.dscalar.nii has 64,984 columns (both full
     32,492-vertex hemispheres, medial wall included), not the 59,412 cortical
     grayordinates. v1.0.0 wrote every surface map into the 91,282 frame by position
     (`full[:gray.size] = gray`), so the sulc parcel means averaged the wrong vertices
     and spilled into subcortical labels. corrThickness and MyelinMap_BC carry exactly
     the atlas's 59,412 vertices in atlas order, so the positional copy was correct for
     them. Every surface map is now placed by vertex identity (hcpee/surface_map.py).
  2. Global volumes. aseg.stats lines read "# Measure BrainSeg, BrainSegVol, ...". The
     old parser stored each measure under its first field, but five of the six requested
     names are second-field names, so only EstimatedTotalIntraCranialVol was found. The
     five misses became NaN and were silently replaced by the subject's mean over the
     other block-A features. Measures are now matched by their verified (first, second)
     field pair, and any missing or non-finite block-A value raises.
Everything else (feature definitions, order, parcellation) is unchanged. See
docs/CHANGELOG.md and docs/PATCHES.md.
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "hcpee"))
import paths                                                          # noqa: E402
import cifti                                                          # noqa: E402
import p29 as P                                                       # noqa: E402
import surface_map as SM                                              # noqa: E402

GRAYORDINATES = 91282
N_PARCELS = 379
N_CORTEX_PARCELS = 360
PREFIX = {"test": "HCP_1200", "retest": "HCP_Retest"}
CACHE = paths.ANAT_DIR
SURFACE_MEASURES = ("corrThickness", "MyelinMap_BC", "sulc")


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
    """Subcortical volumes and the '# Measure' rows of a FreeSurfer aseg.stats file.

    Returns (vols, measures): vols maps StructName -> Volume_mm3; measures is a list of
    (first_field, second_field, value) for every '# Measure' line.
    [release v1.0.1] Returns all fields instead of a dict keyed by the first field.
    """
    vols, measures = {}, []
    for line in text.splitlines():
        if line.startswith("# Measure"):
            f = [x.strip() for x in line[len("# Measure"):].split(",")]
            if len(f) >= 4:
                try:
                    measures.append((f[0], f[1], float(f[3])))
                except ValueError:
                    pass
        elif not line.startswith("#") and line.strip():
            f = line.split()
            if len(f) >= 5:
                try:
                    vols[f[4]] = float(f[3])
                except ValueError:
                    pass
    return vols, measures


ANAT_SUBCORT = ["Left-Thalamus-Proper", "Right-Thalamus-Proper", "Left-Caudate",
                "Right-Caudate", "Left-Putamen", "Right-Putamen", "Left-Pallidum",
                "Right-Pallidum", "Left-Hippocampus", "Right-Hippocampus",
                "Left-Amygdala", "Right-Amygdala", "Left-Accumbens-area",
                "Right-Accumbens-area", "Left-VentralDC", "Right-VentralDC",
                "Brain-Stem", "Left-Cerebellum-Cortex", "Right-Cerebellum-Cortex"]
ANAT_GLOBAL = ["EstimatedTotalIntraCranialVol", "TotalGrayVol", "CortexVol",
               "SubCortGrayVol", "SupraTentorialVol", "BrainSegVol"]
# [release v1.0.1] The (first, second) '# Measure' fields of each intended global
# quantity, as written by FreeSurfer's mri_segstats in every HCP aseg.stats used here
# (84/84 files checked, 2026-10-01). v1.0.0 looked the ANAT_GLOBAL names up among the
# FIRST fields only, which matches EstimatedTotalIntraCranialVol and nothing else.
ASEG_GLOBAL_FIELDS = {
    "EstimatedTotalIntraCranialVol": ("EstimatedTotalIntraCranialVol", "eTIV"),
    "TotalGrayVol": ("TotalGray", "TotalGrayVol"),
    "CortexVol": ("Cortex", "CortexVol"),
    "SubCortGrayVol": ("SubCortGray", "SubCortGrayVol"),
    "SupraTentorialVol": ("SupraTentorial", "SupraTentorialVol"),
    "BrainSegVol": ("BrainSeg", "BrainSegVol"),
}


def global_volumes(measures):
    """The six ANAT_GLOBAL values, in order. Raises unless each occurs exactly once."""
    out = []
    for name in ANAT_GLOBAL:
        pair = ASEG_GLOBAL_FIELDS[name]
        hits = [v for f0, f1, v in measures if (f0, f1) == pair]
        if len(hits) != 1:
            raise ValueError(f"aseg.stats: global measure {name} {pair} found {len(hits)} times")
        out.append(hits[0])
    return out


def build_block_a(lh_text, rh_text, aseg_text):
    """Block A (229 values) from the three FreeSurfer stats files. Raises on any gap."""
    lh, rh = parse_aparc(lh_text), parse_aparc(rh_text)
    vols, measures = parse_aseg(aseg_text)
    names = sorted(set(lh) & set(rh))
    vec = []
    for h in (lh, rh):
        for nm in names:
            vec += [h[nm]["thickness"], h[nm]["area"], h[nm]["curv"]]
    missing = [k for k in ANAT_SUBCORT if k not in vols]
    if missing:
        raise ValueError(f"aseg.stats: subcortical volumes missing: {missing}")
    vec += [vols[k] for k in ANAT_SUBCORT]
    vec += global_volumes(measures)
    a = np.asarray(vec, np.float32)
    if not np.isfinite(a).all():                    # v1.0.0 silently mean-filled here
        raise ValueError("block A has non-finite values")
    return a


def surface_parcel_means(values, src_models, ref_models, atlas_labels):
    """379 parcel means of one surface map, placed on the atlas frame by vertex identity.

    The 19 subcortical parcels have no surface vertices: they are NaN here and become 0
    in the block (as in v1.0.0). Raises if any cortical parcel is not finite.
    """
    full = SM.map_to_frame(values, src_models, ref_models, GRAYORDINATES)
    pm = P.parcellate(full, atlas_labels, N_PARCELS)[0]
    if not np.isfinite(pm[:N_CORTEX_PARCELS]).all():
        raise ValueError("non-finite cortical parcel mean")
    if np.isfinite(pm[N_CORTEX_PARCELS:]).any():
        raise ValueError("surface data reached subcortical parcels")
    return np.nan_to_num(pm, nan=0.0).astype(np.float32)


def read_surface_object(raw):
    """(values, surface models) of a whole single-map CIFTI-2 dscalar held in memory."""
    d = cifti.open_header(raw)
    if d.n_rows != 1:
        raise ValueError(f"expected 1 map, found {d.n_rows}")
    lo, hi = d.data_range()
    if hi != len(raw):
        raise ValueError(f"data range ends at {hi}, object has {len(raw)} bytes")
    return d.extract_map(raw[lo:hi], 0), SM.surface_models(raw)


def reference_frame():
    """Atlas labels and the atlas dlabel's cortical models (the reference frame)."""
    labels = np.load(paths.ATLAS_LABELS)
    ref = SM.surface_models(open(paths.ATLAS_DLABEL, "rb").read())
    n_cortex = sum(m["count"] for m in ref.values())
    if labels.shape != (GRAYORDINATES,) or n_cortex != 59412:
        raise SystemExit("atlas labels / dlabel cortex do not describe the 91,282 frame")
    if not (labels[:n_cortex] >= 1).all() or not (labels[:n_cortex] <= N_CORTEX_PARCELS).all():
        raise SystemExit("cortical grayordinates of the atlas must carry labels 1..360")
    return labels, ref


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def fetch_anatomy(subjects, atlas_labels, ref_models, verbose=True, session="test"):
    """PART J blocks A and B for one session. Returns (s3, out_path, missing)."""
    import hcp_s3
    s3 = hcp_s3.S3()
    suffix = "" if session == "test" else f"_{session}"
    out_path = os.path.join(CACHE, f"anatomy{suffix}.npz")
    if os.path.exists(out_path):
        return s3, out_path, []
    blockA, blockB, kept, missing, prov = {}, {}, [], [], {}
    for subj in subjects:
        p = PREFIX[session]
        try:
            rec, texts = {}, []
            for nm in ("lh.aparc.stats", "rh.aparc.stats", "aseg.stats"):
                key = f"{p}/{subj}/T1w/{subj}/stats/{nm}"
                raw = s3.get(key)
                rec[key] = dict(bytes=len(raw), sha256=_sha(raw))
                texts.append(raw.decode("utf8", "replace"))
            a = build_block_a(*texts)
            surf = []
            for meas in SURFACE_MEASURES:
                key = (f"{p}/{subj}/MNINonLinear/fsaverage_LR32k/"
                       f"{subj}.{meas}.32k_fs_LR.dscalar.nii")
                raw = s3.get(key)
                values, src = read_surface_object(raw)
                rec[key] = dict(bytes=len(raw), sha256=_sha(raw), n_columns=int(values.size),
                                vertex_lists_identical_to_atlas=bool(SM.same_cortex(src, ref_models)))
                surf.append(surface_parcel_means(values, src, ref_models, atlas_labels))
            b = np.concatenate(surf).astype(np.float32)
            blockA[subj], blockB[subj] = a, b
            prov[subj] = rec
            kept.append(subj)
            if verbose:
                print(f"  anat {subj}: A={a.size} B={b.size} "
                      f"({s3.bytes_transferred / 1e6:.1f} MB)")
        except Exception as e:
            missing.append(dict(subject=subj, reason=str(e)[:200]))
            if verbose:
                print(f"  anat EXCLUDE {subj}: {str(e)[:120]}")
    if missing:
        return s3, None, missing
    np.savez_compressed(out_path, subjects=np.array(kept),
                        blockA=np.stack([blockA[s] for s in kept]),
                        blockB=np.stack([blockB[s] for s in kept]))
    json.dump(dict(session=session, prefix=PREFIX[session], n_subjects=len(kept),
                   blockA_layout="DK 2x34x(thickness,area,curv) | 19 subcortical | 6 global",
                   blockB_layout="379 parcels x (corrThickness, MyelinMap_BC, sulc)",
                   aseg_global_fields=ASEG_GLOBAL_FIELDS, files=prov),
              open(os.path.join(CACHE, f"anatomy_provenance{suffix}.json"), "w"), indent=1)
    return s3, out_path, missing


def main():
    os.makedirs(CACHE, exist_ok=True)
    subs = [l.strip() for l in open(paths.SUBJECTS_FILE) if l.strip()]
    labels, ref = reference_frame()
    for ses in ("test", "retest"):
        s3, path, missing = fetch_anatomy(subs, labels, ref, session=ses)
        print(f"{ses}: {path}  missing={len(missing)}  "
              f"streamed {s3.bytes_transferred / 1e6:.1f} MB")
        if missing:
            print(missing)
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
