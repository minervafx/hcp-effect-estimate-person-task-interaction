#!/usr/bin/env python
"""Self-contained tests for hcpee.map_type_guard (no HCP data needed).

Run:  python tests/test_map_type_guard.py
"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "hcpee"))
import map_type_guard as G                                            # noqa: E402

T = ["WM_2BK_minus_0BK", "LANG_STORY_minus_MATH", "MOTOR_AVG_minus_CUE",
     "REL_REL_minus_MATCH"]
FEAT = "HCP_1200/100000/MNINonLinear/Results/tfMRI_WM/tfMRI_WM_hp200_s2_level2_MSMAll.feat"
GOOD = FEAT + "/GrayordinatesStats/cope11.feat/cope1.dtseries.nii"
BAD = [FEAT + "/GrayordinatesStats/cope11.feat/zstat1.dtseries.nii",
       FEAT + "/GrayordinatesStats/cope11.feat/tstat1.dtseries.nii",
       FEAT + "/GrayordinatesStats/cope11.feat/varcope1.dtseries.nii",
       FEAT + "/100000_tfMRI_WM_level2_hp200_s2_MSMAll.dscalar.nii",
       "HCP_1200/100000/MNINonLinear/Results/tfMRI_WM/tfMRI_WM_hp200_s2_level2.feat/"
       "GrayordinatesStats/cope11.feat/cope1.dtseries.nii"]           # MSMSulc, not MSMAll


def expect_fail(fn, *a):
    try:
        fn(*a)
    except G.MapTypeError:
        return True
    raise AssertionError(f"expected MapTypeError for {a[:1]}")


def make_cache(d, scale, manifest=True, map_type=G.MAP_TYPE):
    rng = np.random.default_rng(0)
    for s in ("100000", "100001"):
        for ses in ("test", "retest"):
            np.savez(os.path.join(d, f"{s}_{ses}.npz"),
                     **{f"{t}__{x}": (rng.standard_normal(379) * scale).astype(np.float32)
                        for t in T for x in ("diff", "A", "B")})
    if manifest:
        json.dump(dict(map_type=map_type, objects=[GOOD], n_objects=1),
                  open(os.path.join(d, G.MANIFEST_NAME), "w"))


def main():
    assert G.assert_cope_key(GOOD)
    for k in BAD:
        expect_fail(G.assert_cope_key, k)
    subs = ["100000", "100001"]
    with tempfile.TemporaryDirectory() as d:          # effect-estimate scale, manifest
        make_cache(d, 30.0)
        assert G.assert_effect_estimate_cache(d, T, subs)["ok"]
    with tempfile.TemporaryDirectory() as d:          # no manifest
        make_cache(d, 30.0, manifest=False)
        expect_fail(G.assert_effect_estimate_cache, d, T, subs)
    with tempfile.TemporaryDirectory() as d:          # wrong declared type
        make_cache(d, 30.0, map_type="zstat")
        expect_fail(G.assert_effect_estimate_cache, d, T, subs)
    with tempfile.TemporaryDirectory() as d:          # z-statistic scale relabelled as COPE
        make_cache(d, 1.0)
        expect_fail(G.assert_effect_estimate_cache, d, T, subs)
    print("map_type_guard: all tests passed")


if __name__ == "__main__":
    main()
