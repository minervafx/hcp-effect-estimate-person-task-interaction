#!/usr/bin/env python
"""Build the 379-parcel grayordinate label vector used by every analysis.

Labels 1-360: the HCP-MMP1.0 group cortical parcellation (Glasser et al., 2016), read
from the group dlabel shipped in the HCP S1200 group-average package:
    HCP_Resources/Workbench/HCP_S1200_GroupAvg_v1.zip ::
    HCP_S1200_GroupAvg_v1/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii
    (SHA-256 f3315dd22c10c64234a38dde923db05fafa2c137b3874bec9e49fee85edce9a6)
Labels 361-379: the 19 subcortical CIFTI voxel structures of the standard 91,282
grayordinate space, numbered in brain-model order as they appear in the task COPE files.

The atlas is not redistributed here. Either place the dlabel at $HCPEE_ATLAS_DLABEL, or
pass --fetch to extract that single member from the HCP bucket with your own HCP Open
Access credentials. The subcortical layout is read from the header of one COPE file of
the first included subject (--fetch), or from a local 91,282-grayordinate CIFTI (--cifti).

[release] New script. The historical project built the same vector interactively; this
script reproduces the saved vector exactly (verified 2026-09-29: array-equal to the file
used for all reported analyses, SHA-256 0e361613...). See docs/PROVENANCE.md.

[release v1.0.1] Adds a check that the cortical VertexIndices of the COPE (or --cifti)
header equal those of the dlabel, so labels 1-360 provably sit on the same surface
vertices as the task data. The label vector written is unchanged.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "hcpee"))
import paths                                                          # noqa: E402
import cifti                                                          # noqa: E402
import surface_map as SM                                              # noqa: E402

ZIP_KEY = "HCP_Resources/Workbench/HCP_S1200_GroupAvg_v1.zip"
MEMBER = ("HCP_S1200_GroupAvg_v1/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_"
          "Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii")
DLABEL_SHA256 = "f3315dd22c10c64234a38dde923db05fafa2c137b3874bec9e49fee85edce9a6"
N_CORTEX, GRAYORDINATES, N_PARCELS = 59412, 91282, 379


def header_from_bytes(get):
    buf = get(0, 800_000)
    while True:
        try:
            return cifti.open_header(buf), buf
        except cifti.NeedMoreBytes as e:
            buf = get(0, e.needed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true",
                    help="extract the dlabel and read a COPE header from the HCP bucket")
    ap.add_argument("--cifti", help="local 91,282-grayordinate CIFTI for the subcortical layout")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(paths.ATLAS_LABELS), exist_ok=True)
    s3 = None
    if a.fetch:
        import hcp_s3
        import s3zip
        s3 = hcp_s3.S3()
        if not os.path.exists(paths.ATLAS_DLABEL):
            b = s3zip.S3Zip(s3, ZIP_KEY).read(MEMBER)
            os.makedirs(os.path.dirname(paths.ATLAS_DLABEL), exist_ok=True)
            open(paths.ATLAS_DLABEL, "wb").write(b)
    raw = open(paths.ATLAS_DLABEL, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != DLABEL_SHA256:
        raise SystemExit(f"dlabel SHA-256 {sha} != expected {DLABEL_SHA256}")
    d = cifti.open_header(raw)
    lo, hi = d.data_range()
    cortex = d.extract_map(raw[lo:hi], 0)
    if cortex.size != N_CORTEX or not np.array_equal(np.unique(cortex), np.arange(1, 361)):
        raise SystemExit("dlabel does not carry labels 1..360 over 59,412 cortical grayordinates")

    if a.cifti:
        buf = open(a.cifti, "rb").read()
        h, hbuf = header_from_bytes(lambda lo_, hi_: buf[lo_:hi_])
    elif s3 is not None:
        subj = [l.strip() for l in open(paths.SUBJECTS_FILE) if l.strip()][0]
        key = (f"HCP_1200/{subj}/MNINonLinear/Results/tfMRI_WM/tfMRI_WM_hp200_s2_level2_"
               f"MSMAll.feat/GrayordinatesStats/cope11.feat/cope1.dtseries.nii")
        h, hbuf = header_from_bytes(lambda lo_, hi_: s3.get_range(key, lo_, hi_))
    else:
        raise SystemExit("need --fetch or --cifti for the subcortical brain-model layout")
    st = h.structures if not callable(h.structures) else h.structures()
    if h.n_cols != GRAYORDINATES:
        raise SystemExit(f"CIFTI has {h.n_cols} grayordinates, expected {GRAYORDINATES}")
    if not SM.same_cortex(SM.surface_models(hbuf), SM.surface_models(raw)):
        raise SystemExit("cortical VertexIndices of the task CIFTI differ from the dlabel's")

    labels = np.zeros(GRAYORDINATES, np.int32)
    labels[:N_CORTEX] = cortex.astype(np.int32)
    names = [f"MMP_{i}" for i in range(1, 361)]
    vox = [s for s in st if s["model_type"] == "CIFTI_MODEL_TYPE_VOXELS"]
    for j, s in enumerate(vox, start=361):
        o, n = int(s["index_offset"]), int(s["index_count"])
        labels[o:o + n] = j
        names.append(s["name"].replace("CIFTI_STRUCTURE_", ""))
    if len(names) != N_PARCELS or (labels == 0).any():
        raise SystemExit("unexpected brain-model layout")
    np.save(paths.ATLAS_LABELS, labels)
    json.dump(names, open(os.path.join(os.path.dirname(paths.ATLAS_LABELS),
                                       "parcel_names.json"), "w"))
    print("wrote", paths.ATLAS_LABELS, "sha256",
          hashlib.sha256(open(paths.ATLAS_LABELS, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
