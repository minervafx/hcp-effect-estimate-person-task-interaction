#!/usr/bin/env python
"""Integration test of the anatomy inputs on real HCP Open Access files.

Needs HCP Open Access credentials and the atlas from scripts/00_build_atlas.py; without
them it prints SKIPPED and exits 0. Reads one subject-session's stats and surface files
(about 1.5 MB) and the HCP S1200 group-average sulcal-depth map.

Checks:
  1. Sulcal depth placed by vertex identity agrees with the HCP group-average sulc map
     across the 360 cortical parcels (r > 0.8; observed 0.90-0.93 on four checked
     subject-sessions, 2026-10-01). The v1.0.0 positional copy gives |r| < 0.3
     (observed -0.03 to 0.02), so this test separates the two.
  2. corrThickness and MyelinMap_BC list the atlas's cortical vertices in atlas order,
     so vertex mapping and the v1.0.0 positional copy agree exactly for them.
  3. Block A builds with all six global volumes read from aseg.stats, each equal to the
     value on its '# Measure' line; the v1.0.0 first-field lookup finds only one of six.

Run:  python tests/test_anatomy_real_data.py
"""
import importlib.util
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "hcpee"))
import paths                                                          # noqa: E402
import cifti                                                          # noqa: E402
import surface_map as SM                                              # noqa: E402
import p29 as P                                                       # noqa: E402

spec = importlib.util.spec_from_file_location(
    "fetch_anatomy", os.path.join(REPO, "scripts", "03_fetch_anatomy.py"))
FA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FA)

GROUP_ZIP = "HCP_Resources/Workbench/HCP_S1200_GroupAvg_v1.zip"
GROUP_SULC = "HCP_S1200_GroupAvg_v1/S1200.sulc_MSMAll.32k_fs_LR.dscalar.nii"


def parcel_means(raw, ref, labels):
    values, src = FA.read_surface_object(raw)
    return FA.surface_parcel_means(values, src, ref, labels)[:360], values


def positional_means(values, labels):
    full = np.full(91282, np.nan, np.float32)
    full[:values.size] = values                                   # v1.0.0 behaviour
    return P.parcellate(full, labels, 379)[0][:360]


def main():
    import hcp_s3
    import s3zip
    try:
        hcp_s3.load_credentials()
    except Exception:
        print("anatomy real-data test: SKIPPED (no HCP Open Access credentials)")
        return 0
    if not (os.path.exists(paths.ATLAS_LABELS) and os.path.exists(paths.ATLAS_DLABEL)):
        print("anatomy real-data test: SKIPPED (run scripts/00_build_atlas.py first)")
        return 0
    labels, ref = FA.reference_frame()
    s3 = hcp_s3.S3()
    subj = [l.strip() for l in open(paths.SUBJECTS_FILE) if l.strip()][0]
    base = f"HCP_1200/{subj}"

    group, _ = parcel_means(s3zip.S3Zip(s3, GROUP_ZIP).read(GROUP_SULC), ref, labels)
    raw = s3.get(f"{base}/MNINonLinear/fsaverage_LR32k/{subj}.sulc.32k_fs_LR.dscalar.nii")
    sulc, values = parcel_means(raw, ref, labels)
    assert values.size == 64984, values.size                      # full surfaces, medial wall included
    r_new = float(np.corrcoef(sulc, group)[0, 1])
    r_old = float(np.corrcoef(positional_means(values, labels), group)[0, 1])
    assert r_new > 0.8, r_new
    assert abs(r_old) < 0.3, r_old

    for meas in ("corrThickness", "MyelinMap_BC"):
        raw = s3.get(f"{base}/MNINonLinear/fsaverage_LR32k/{subj}.{meas}.32k_fs_LR.dscalar.nii")
        new, values = parcel_means(raw, ref, labels)
        assert SM.same_cortex(FA.read_surface_object(raw)[1], ref)
        assert np.array_equal(new, positional_means(values, labels).astype(np.float32)), meas

    texts = [s3.get(f"{base}/T1w/{subj}/stats/{n}").decode("utf8", "replace")
             for n in ("lh.aparc.stats", "rh.aparc.stats", "aseg.stats")]
    a = FA.build_block_a(*texts)
    _, meas = FA.parse_aseg(texts[2])
    line_values = {f1: v for _, f1, v in meas}
    want = [line_values[FA.ASEG_GLOBAL_FIELDS[k][1]] for k in FA.ANAT_GLOBAL]
    assert a.size == 229 and np.allclose(a[-6:], want), (a[-6:], want)
    assert sum(k in {f0 for f0, _, _ in meas} for k in FA.ANAT_GLOBAL) == 1
    print(f"anatomy real-data test: passed (subject {subj}; sulc r with group map "
          f"{r_new:.3f}, v1.0.0 positional {r_old:+.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
