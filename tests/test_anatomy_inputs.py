#!/usr/bin/env python
"""Self-contained tests for the v1.0.1 anatomy-input corrections (no HCP data needed).

Covers the two v1.0.0 defects:
  * surface maps must be placed on the atlas frame by CIFTI vertex identity, not by
    position (the 64,984-column sulc file broke the positional copy);
  * every intended aseg.stats global volume must be parsed from its verified field pair,
    and a missing value must raise instead of becoming a fill value.

Run:  python tests/test_anatomy_inputs.py
"""
import importlib.util
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "hcpee"))
import surface_map as SM                                              # noqa: E402
import p29 as P                                                       # noqa: E402

spec = importlib.util.spec_from_file_location(
    "fetch_anatomy", os.path.join(REPO, "scripts", "03_fetch_anatomy.py"))
FA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FA)

L, R = SM.CORTEX


def expect_fail(fn, *a, exc=Exception):
    try:
        fn(*a)
    except exc:
        return True
    raise AssertionError(f"expected {exc.__name__} from {fn.__name__}")


# ----------------------------------------------------------------- synthetic frame
# Two 8-vertex hemispheres. The reference (atlas) cortex omits a "medial wall"
# (left 3, 6; right 1, 5) and is followed by 4 non-surface columns, like the
# 91,282 frame (cortex, then subcortical voxels).
NV = 8
REF_L = np.array([0, 1, 2, 4, 5, 7])
REF_R = np.array([0, 2, 3, 4, 6, 7])
N_FRAME = REF_L.size + REF_R.size + 4
REF = {L: dict(offset=0, count=6, n_vertices=NV, vertices=REF_L),
       R: dict(offset=6, count=6, n_vertices=NV, vertices=REF_R)}
FULL = {L: dict(offset=0, count=NV, n_vertices=NV, vertices=np.arange(NV)),     # sulc-like
        R: dict(offset=NV, count=NV, n_vertices=NV, vertices=np.arange(NV))}
FULL_VALUES = np.concatenate([100 + np.arange(NV), 200 + np.arange(NV)]).astype(np.float32)
# parcels: 1 = left cortex, 2 = right cortex, 3 = "subcortical" columns
LABELS = np.array([1] * 6 + [2] * 6 + [3] * 4)


def vertex_value(hemi, v):
    return (100 if hemi == L else 200) + v


def test_full_surface_mapped_by_vertex():
    out = SM.map_to_frame(FULL_VALUES, FULL, REF, N_FRAME)
    want = np.concatenate([[vertex_value(L, v) for v in REF_L], [vertex_value(R, v) for v in REF_R]])
    assert np.array_equal(out[:12], want), out
    assert np.isnan(out[12:]).all()                               # nothing outside cortex


def test_old_positional_copy_is_detected():
    old = np.full(N_FRAME, np.nan, np.float32)
    old[:min(N_FRAME, FULL_VALUES.size)] = FULL_VALUES[:N_FRAME]  # v1.0.0: full[:gray.size] = gray
    new = SM.map_to_frame(FULL_VALUES, FULL, REF, N_FRAME)
    assert not np.array_equal(old[:12], new[:12])                 # wrong vertices
    pm_old = P.parcellate(old, LABELS, 3)[0]
    assert np.isfinite(pm_old[2])                                 # leaked into "subcortical"
    pm_new = P.parcellate(new, LABELS, 3)[0]
    assert np.isnan(pm_new[2])


def test_atlas_ordered_source_equals_positional():
    # corrThickness / MyelinMap_BC case: same vertices, same order as the atlas
    vals = np.arange(12, dtype=np.float32) + 0.5
    out = SM.map_to_frame(vals, REF, REF, N_FRAME)
    assert np.array_equal(out[:12], vals) and SM.same_cortex(REF, REF)


def test_reordered_source():
    perm = {L: dict(offset=0, count=6, n_vertices=NV, vertices=REF_L[::-1].copy()),
            R: dict(offset=6, count=6, n_vertices=NV, vertices=REF_R[::-1].copy())}
    vals = np.concatenate([[vertex_value(L, v) for v in REF_L[::-1]],
                           [vertex_value(R, v) for v in REF_R[::-1]]]).astype(np.float32)
    out = SM.map_to_frame(vals, perm, REF, N_FRAME)
    want = np.concatenate([[vertex_value(L, v) for v in REF_L], [vertex_value(R, v) for v in REF_R]])
    assert np.array_equal(out[:12], want)
    assert not SM.same_cortex(perm, REF)


def test_mapping_refuses_bad_inputs():
    E = SM.SurfaceMapError
    short = {L: dict(offset=0, count=5, n_vertices=NV, vertices=REF_L[:5]),
             R: dict(offset=5, count=6, n_vertices=NV, vertices=REF_R)}
    expect_fail(SM.map_to_frame, np.zeros(11), short, REF, N_FRAME, exc=E)   # vertex missing
    expect_fail(SM.map_to_frame, np.zeros(15), FULL, REF, N_FRAME, exc=E)    # wrong length
    expect_fail(SM.map_to_frame, np.zeros(8), {L: FULL[L]}, REF, N_FRAME, exc=E)  # one hemisphere
    other = {k: dict(v, n_vertices=NV + 1) for k, v in FULL.items()}
    expect_fail(SM.map_to_frame, FULL_VALUES, other, REF, N_FRAME, exc=E)    # mesh mismatch


def test_xml_parser():
    def bm(s, off, verts, nv=NV):
        return (f'<BrainModel IndexOffset="{off}" IndexCount="{len(verts)}" '
                f'ModelType="CIFTI_MODEL_TYPE_SURFACE" BrainStructure="{s}" '
                f'SurfaceNumberOfVertices="{nv}"><VertexIndices>'
                + " ".join(map(str, verts)) + "</VertexIndices></BrainModel>")
    xml = ('<CIFTI Version="2"><Matrix><MatrixIndicesMap AppliesToMatrixDimension="1" '
           'IndicesMapToDataType="CIFTI_INDEX_TYPE_BRAIN_MODELS">'
           + bm(L, 0, REF_L) + bm(R, 6, REF_R)
           + '<BrainModel IndexOffset="12" IndexCount="4" ModelType="CIFTI_MODEL_TYPE_VOXELS" '
             'BrainStructure="CIFTI_STRUCTURE_ACCUMBENS_LEFT"></BrainModel>'
           '</MatrixIndicesMap></Matrix></CIFTI>').encode()
    m = SM.surface_models_from_xml(xml)
    assert set(m) == {L, R} and SM.same_cortex(m, REF)


# ------------------------------------------------------------------- aseg / block A
ASEG = """# Title Segmentation Statistics
# generating_program mri_segstats
# Measure BrainSeg, BrainSegVol, Brain Segmentation Volume, 1100000.000000, mm^3
# Measure BrainSegNotVent, BrainSegVolNotVent, Brain Segmentation Volume Without Ventricles, 1080000.000000, mm^3
# Measure Cortex, CortexVol, Total cortical gray matter volume, 480000.000000, mm^3
# Measure SubCortGray, SubCortGrayVol, Subcortical gray matter volume, 57000.000000, mm^3
# Measure TotalGray, TotalGrayVol, Total gray matter volume, 640000.000000, mm^3
# Measure SupraTentorial, SupraTentorialVol, Supratentorial volume, 970000.000000, mm^3
# Measure SupraTentorialNotVent, SupraTentorialVolNotVent, Supratentorial volume, 950000.000000, mm^3
# Measure BrainSegVol-to-eTIV, BrainSegVol-to-eTIV, Ratio of BrainSegVol to eTIV, 0.75, unitless
# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated Total Intracranial Volume, 1500000.000000, mm^3
# ColHeaders  Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange
"""
WANT_GLOBAL = [1500000.0, 640000.0, 480000.0, 57000.0, 970000.0, 1100000.0]
for i, s in enumerate(FA.ANAT_SUBCORT):
    ASEG += f"{i + 1:3d} {i + 10:4d} {1000 + i:7d} {1000.0 + i:9.1f} {s} 80.0 10.0 30.0 120.0 90.0\n"
APARC = "# Table of FreeSurfer cortical parcellation anatomical statistics\n" + "".join(
    f"{nm} 5000 {2000 + i} {6000 + i} {2.5 + i / 100:.3f} 0.6 {0.12 + i / 1000:.3f} 0.03 20 4.0\n"
    for i, nm in enumerate(["bankssts", "caudalanteriorcingulate", "cuneus"]))


def test_all_global_volumes_parsed():
    vols, meas = FA.parse_aseg(ASEG)
    assert FA.global_volumes(meas) == WANT_GLOBAL
    first = {f0 for f0, _, _ in meas}                             # v1.0.0 lookup: first fields
    assert [k in first for k in FA.ANAT_GLOBAL] == [True, False, False, False, False, False]


def test_block_a_layout_and_no_fill():
    a = FA.build_block_a(APARC, APARC, ASEG)
    assert a.size == 2 * 3 * 3 + 19 + 6
    assert np.allclose(a[-6:], WANT_GLOBAL)
    assert len(set(a[-5:].tolist())) == 5                         # not one repeated fill value
    for drop in ("# Measure TotalGray,", "# Measure EstimatedTotalIntraCranialVol,"):
        broken = "\n".join(l for l in ASEG.splitlines() if not l.startswith(drop))
        expect_fail(FA.build_block_a, APARC, APARC, broken, exc=ValueError)
    no_stem = "\n".join(l for l in ASEG.splitlines() if not l.rstrip().endswith("90.0") or "Brain-Stem" not in l)
    expect_fail(FA.build_block_a, APARC, APARC, no_stem, exc=ValueError)
    dup = ASEG + "# Measure Cortex, CortexVol, Total cortical gray matter volume, 1.0, mm^3\n"
    expect_fail(FA.build_block_a, APARC, APARC, dup, exc=ValueError)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"anatomy inputs: all {len(tests)} tests passed")


if __name__ == "__main__":
    main()
