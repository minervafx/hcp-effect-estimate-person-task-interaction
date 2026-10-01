"""Explicit CIFTI surface-vertex mapping onto the analysis grayordinate frame.

[release v1.0.1] New module. It replaces the positional padding used by v1.0.0's
anatomy builder (`full[:gray.size] = gray`), which is correct only when a surface file
lists exactly the same cortical vertices, in the same order, as the atlas.

Facts this module relies on, checked against the HCP Open Access files on 2026-10-01:
  * The analysis frame is the standard 91,282-grayordinate CIFTI space. Its cortex is
    CORTEX_LEFT (offset 0, 29,696 vertices) and CORTEX_RIGHT (offset 29,696, 29,716
    vertices) of 32,492-vertex 32k_fs_LR surfaces; the medial wall is excluded. The
    group HCP-MMP1.0 dlabel, the task COPE files and the resting-state dtseries carry
    identical cortical VertexIndices lists.
  * {subject}.corrThickness and {subject}.MyelinMap_BC 32k dscalars carry the same
    59,412 cortical vertices, in the same order.
  * {subject}.sulc 32k dscalars carry 64,984 columns: all 32,492 vertices of each
    hemisphere (VertexIndices 0..32491), including the medial wall.

A surface value must therefore be placed through the BrainModel VertexIndices of both
the source file and the reference (atlas) frame, never by position.
"""
import xml.etree.ElementTree as ET

import numpy as np

import cifti

CORTEX = ("CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT")


class SurfaceMapError(ValueError):
    pass


def _tag(el):
    return el.tag.split("}", 1)[-1]


def surface_models_from_xml(xml_bytes):
    """Every SURFACE BrainModel of a CIFTI-2 XML extension.

    Returns {structure: dict(offset, count, n_vertices, vertices)}, where `vertices` is
    the int64 VertexIndices array: column offset + k holds surface vertex vertices[k].
    """
    root = ET.fromstring(xml_bytes)
    out = {}
    for bm in root.iter():
        if _tag(bm) != "BrainModel" or not bm.get("ModelType", "").endswith("SURFACE"):
            continue
        vi = [c for c in bm if _tag(c) == "VertexIndices"]
        if len(vi) != 1:
            raise SurfaceMapError(f"{bm.get('BrainStructure')}: no VertexIndices element")
        verts = np.array((vi[0].text or "").split(), np.int64)
        count = int(bm.get("IndexCount"))
        if verts.size != count:
            raise SurfaceMapError(f"{bm.get('BrainStructure')}: {verts.size} VertexIndices, "
                                  f"IndexCount {count}")
        out[bm.get("BrainStructure")] = dict(offset=int(bm.get("IndexOffset")), count=count,
                                             n_vertices=int(bm.get("SurfaceNumberOfVertices")),
                                             vertices=verts)
    return out


def surface_models(header_bytes):
    """Surface BrainModels from the leading bytes of a CIFTI-2 file."""
    hdr = cifti.parse_header(header_bytes)
    ext = cifti.parse_extensions(header_bytes, hdr)
    if 32 not in ext:
        raise SurfaceMapError("no CIFTI-2 XML extension")
    return surface_models_from_xml(ext[32])


def same_cortex(a, b):
    """True if two model dicts describe identical cortical columns (offset, order, vertices)."""
    for s in CORTEX:
        if s not in a or s not in b:
            return False
        x, y = a[s], b[s]
        if (x["offset"], x["count"], x["n_vertices"]) != (y["offset"], y["count"], y["n_vertices"]):
            return False
        if not np.array_equal(x["vertices"], y["vertices"]):
            return False
    return True


def map_to_frame(values, src, ref, n_frame):
    """Place a surface map onto the reference grayordinate frame by vertex identity.

    values : 1-D array, one value per column of the source file
    src    : surface_models() of the source file (cortex only, see CORTEX)
    ref    : surface_models() of the reference frame (the atlas dlabel)
    Returns float32 (n_frame,), NaN outside the reference cortex.

    Raises if the hemispheres, mesh sizes or column count do not match, or if any
    reference vertex is absent from the source.
    """
    values = np.asarray(values)
    if values.ndim != 1:
        raise SurfaceMapError("values must be 1-D")
    if set(src) != set(CORTEX):
        raise SurfaceMapError(f"source surface structures {sorted(src)} != {list(CORTEX)}")
    if set(ref) != set(CORTEX):
        raise SurfaceMapError(f"reference surface structures {sorted(ref)} != {list(CORTEX)}")
    n_src = sum(m["count"] for m in src.values())
    if values.size != n_src:
        raise SurfaceMapError(f"{values.size} values for {n_src} source surface columns")
    out = np.full(int(n_frame), np.nan, np.float32)
    for s in CORTEX:
        a, r = src[s], ref[s]
        if a["n_vertices"] != r["n_vertices"]:
            raise SurfaceMapError(f"{s}: source mesh {a['n_vertices']} != reference {r['n_vertices']}")
        col = np.full(a["n_vertices"], -1, np.int64)            # surface vertex -> source column
        col[a["vertices"]] = a["offset"] + np.arange(a["count"])
        want = col[r["vertices"]]
        if (want < 0).any():
            raise SurfaceMapError(f"{s}: {(want < 0).sum()} reference vertices missing in source")
        if r["offset"] + r["count"] > n_frame:
            raise SurfaceMapError(f"{s}: reference columns exceed frame of {n_frame}")
        out[r["offset"]:r["offset"] + r["count"]] = values[want]
    return out
