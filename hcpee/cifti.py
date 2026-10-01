#!/usr/bin/env python
"""Minimal, dependency-free CIFTI-2 reader for Phase 2.9.

Rationale (PART R - resource rules).  Reading a `.dscalar.nii` needs almost nothing:
CIFTI-2 is a NIfTI-2 file whose voxel grid is degenerate and whose real structure lives
in an XML extension (code 32).  Installing `nibabel` (plus its dependency chain) into a
1.9 GB / 845 MB-free VPS to parse a 540-byte header is not justified, and a full
FSL/FreeSurfer install certainly is not.  This module implements exactly the three
operations Phase 2.9 needs:

    * parse the NIfTI-2 header and the CIFTI XML (map names, brain-model layout)
    * read the whole data matrix
    * extract one contrast map from that matrix by name

**Storage order - corrected 2026-09-03 against real HCP objects.**  An earlier version
of this module assumed a dscalar was stored one contiguous map after another, which
would have let a single HTTP range GET pull one COPE out of a multi-contrast file.
That is FALSE.  NIfTI stores the *first* dimension fastest, so for dim = [.., n_maps,
n_grayordinates, ..] the MAP index varies fastest and the maps are **interleaved**:
element (map m, grayordinate g) sits at flat offset g * n_maps + m.  One map is
therefore strided across the entire data block and cannot be range-extracted.

The error was invisible to `99_selftest.py` because `write_dscalar` below wrote its
synthetic fixtures in the same wrong order the reader used - self-consistent, and
wrong.  It was caught only by an algebraic identity on real data: HCP releases both
`2BK` and `neg_2BK`, and `neg_2BK` must equal `-(2BK)` exactly.  Under the contiguous
reading it correlated at r = 0.0001; under the interleaved reading it matches to
float32 rounding.  Both the reader and the writer now use the interleaved order, and
`99_selftest.py` asserts the identity that would have caught it.

NIfTI-2 layout used here (little-endian assumed, verified against the magic):
    off   0  int32   sizeof_hdr = 540
    off   4  char[8] magic  b'n+2\\0\\r\\n\\x1a\\n'
    off  12  int16   datatype
    off  14  int16   bitpix
    off  16  int64   dim[8]
    off 168  double  pixdim[8]
    off 232  int64   vox_offset
    off 540  ...     extension flag (4 bytes) then extensions

For a CIFTI-2 dscalar the matrix is stored row-major as dim[5] rows x dim[6] columns,
where a *row* is one map (one contrast) and a *column* is one grayordinate.
"""
import io
import struct
import xml.etree.ElementTree as ET

import numpy as np

NIFTI2_HDR_SIZE = 540
NIFTI2_MAGIC = b"n+2\x00\r\n\x1a\n"
NIFTI2_MAGIC_HDR = b"ni2\x00\r\n\x1a\n"          # detached .hdr form, not used by HCP

DTYPE = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64,
         256: np.int8, 512: np.uint16, 768: np.uint32, 1024: np.int64,
         1280: np.uint64}


class Nifti2Header(object):
    __slots__ = ("sizeof_hdr", "magic", "datatype", "bitpix", "dim", "pixdim",
                 "vox_offset", "endian")

    def __repr__(self):
        return (f"<Nifti2Header dim={self.dim} dtype={self.datatype} "
                f"vox_offset={self.vox_offset}>")


def parse_header(buf):
    """Parse the fixed 540-byte NIfTI-2 header from `buf` (bytes, >= 544)."""
    if len(buf) < NIFTI2_HDR_SIZE:
        raise ValueError(f"need >= {NIFTI2_HDR_SIZE} bytes, got {len(buf)}")
    for endian in ("<", ">"):
        (sizeof_hdr,) = struct.unpack_from(endian + "i", buf, 0)
        if sizeof_hdr == NIFTI2_HDR_SIZE:
            break
    else:
        raise ValueError("not a NIfTI-2 header (sizeof_hdr != 540 in either endianness)")
    magic = buf[4:12]
    if magic not in (NIFTI2_MAGIC, NIFTI2_MAGIC_HDR):
        raise ValueError(f"bad NIfTI-2 magic {magic!r}")
    h = Nifti2Header()
    h.endian = endian
    h.sizeof_hdr = sizeof_hdr
    h.magic = magic
    h.datatype, h.bitpix = struct.unpack_from(endian + "hh", buf, 12)
    h.dim = list(struct.unpack_from(endian + "8q", buf, 16))
    h.pixdim = list(struct.unpack_from(endian + "8d", buf, 168))
    (h.vox_offset,) = struct.unpack_from(endian + "q", buf, 232)
    if h.datatype not in DTYPE:
        raise ValueError(f"unsupported datatype {h.datatype}")
    return h


# Extension codes NIfTI defines; used to decide where the extension list ends when
# vox_offset is 0 (HCP writes 0, so the data offset must be derived by walking).
KNOWN_ECODES = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
                38, 40, 42, 44, 46}


class NeedMoreBytes(ValueError):
    """Raised when the supplied prefix is too short to contain header + extensions."""

    def __init__(self, needed, have):
        super().__init__(f"need {needed} leading bytes to parse the CIFTI header, "
                         f"have {have}")
        self.needed = int(needed)
        self.have = int(have)


def _walk_extensions(buf, hdr, collect):
    """Walk the extension list from 544. Returns (end_offset, {ecode: bytes}).

    NIfTI terminates the extension list at `vox_offset`, but HCP's level-2 dscalars
    write **vox_offset = 0**, so the end has to be found by walking. An entry is
    accepted only if its size is a sane positive multiple of 16 and its code is one
    NIfTI actually defines; the first entry failing that is treated as the start of
    the data block. The result is cross-checked against the object size by the caller,
    which is what makes this safe rather than merely plausible.
    """
    out = {}
    if len(buf) < NIFTI2_HDR_SIZE + 4:
        raise NeedMoreBytes(NIFTI2_HDR_SIZE + 4, len(buf))
    if buf[NIFTI2_HDR_SIZE] == 0:
        return NIFTI2_HDR_SIZE + 4, out                   # no extensions present
    hard_end = int(hdr.vox_offset) or None
    pos = NIFTI2_HDR_SIZE + 4
    while True:
        if hard_end is not None and pos >= hard_end:
            break
        if pos + 8 > len(buf):
            if hard_end is None:
                break                                     # ran out; treat as the end
            raise NeedMoreBytes(pos + 8, len(buf))
        esize, ecode = struct.unpack_from(hdr.endian + "ii", buf, pos)
        if esize <= 8 or esize % 16 != 0 or ecode not in KNOWN_ECODES:
            break                                         # not an extension -> data
        if pos + esize > len(buf):
            raise NeedMoreBytes(pos + esize, len(buf))
        if collect:
            out.setdefault(ecode, buf[pos + 8:pos + esize].rstrip(b"\x00"))
        pos += esize
    return pos, out


def parse_extensions(buf, hdr):
    """Return {ecode: bytes} for every extension after the 540-byte header."""
    return _walk_extensions(buf, hdr, True)[1]


def header_length(buf):
    """Bytes of `buf` occupied by header + extensions, i.e. where the data starts.

    Raises NeedMoreBytes (carrying `.needed`) if the prefix is too short, so a caller
    streaming over HTTP can issue exactly one more range request.
    """
    return _walk_extensions(buf, parse_header(buf), False)[0]


def _strip_ns(tag):
    return tag.split("}", 1)[-1] if "}" in tag else tag


def parse_cifti_xml(xml_bytes):
    """Extract map names and the brain-model layout from the CIFTI-2 XML extension.

    Returns dict with:
        map_names   list[str]  - one per row of the matrix (contrast names for dscalar)
        structures  list of dict(name, index_offset, index_count, model_type)
        n_columns   int        - total grayordinates declared by the brain models
    """
    root = ET.fromstring(xml_bytes)
    map_names, structures, n_columns = [], [], 0
    for mim in root.iter():
        if _strip_ns(mim.tag) != "MatrixIndicesMap":
            continue
        itype = mim.get("IndicesMapToDataType", "")
        if itype.endswith("SCALARS"):
            for nm in mim:
                if _strip_ns(nm.tag) == "NamedMap":
                    label = ""
                    for ch in nm:
                        if _strip_ns(ch.tag) == "MapName":
                            label = (ch.text or "").strip()
                    map_names.append(label)
        elif itype.endswith("BRAIN_MODELS"):
            for bm in mim:
                if _strip_ns(bm.tag) != "BrainModel":
                    continue
                off = int(bm.get("IndexOffset", 0))
                cnt = int(bm.get("IndexCount", 0))
                structures.append(dict(name=bm.get("BrainStructure", ""),
                                       index_offset=off, index_count=cnt,
                                       model_type=bm.get("ModelType", "")))
                n_columns = max(n_columns, off + cnt)
    return dict(map_names=map_names, structures=structures, n_columns=n_columns)


class Dscalar(object):
    """A parsed CIFTI-2 dscalar/dtseries header, without its data."""

    def __init__(self, hdr, xml, data_start=None):
        self.hdr = hdr
        self.xml = xml
        self.n_rows = int(hdr.dim[5]) if hdr.dim[0] >= 5 else 1
        self.n_cols = int(hdr.dim[6]) if hdr.dim[0] >= 6 else int(hdr.dim[5])
        self.dtype = np.dtype(DTYPE[hdr.datatype]).newbyteorder(hdr.endian)
        self.itemsize = self.dtype.itemsize
        # HCP writes vox_offset = 0; the real offset comes from the extension walk
        self.data_start = int(data_start if data_start is not None
                              else hdr.vox_offset)

    @property
    def map_names(self):
        return self.xml["map_names"]

    @property
    def structures(self):
        return self.xml["structures"]

    @property
    def data_nbytes(self):
        return self.n_rows * self.n_cols * self.itemsize

    def data_range(self):
        """(start, end_exclusive) of the whole data block.

        There is no per-map range: maps are interleaved (see the module docstring), so
        reading any single contrast requires the whole block.
        """
        return self.data_start, self.data_start + self.data_nbytes

    def decode_matrix(self, raw):
        """Decode the data block to (n_maps, n_grayordinates).

        The on-disk order is (n_grayordinates, n_maps) with the map index fastest, so
        the reshape is by n_cols rows of n_rows values, then transposed.
        """
        need = self.n_rows * self.n_cols
        a = np.frombuffer(raw, dtype=self.dtype, count=need)
        return np.ascontiguousarray(a.reshape(self.n_cols, self.n_rows).T,
                                    dtype=np.float32)

    def extract_map(self, raw, i):
        """One contrast map (n_grayordinates,) out of the decoded data block."""
        if not 0 <= i < self.n_rows:
            raise IndexError(f"map {i} out of range (n_maps={self.n_rows})")
        a = np.frombuffer(raw, dtype=self.dtype, count=self.n_rows * self.n_cols)
        return np.ascontiguousarray(a[i::self.n_rows], dtype=np.float32)

    def find_map(self, needle, exact=False):
        """Index of the first map whose name matches. Raises if ambiguous or absent."""
        names = self.map_names
        if exact:
            hits = [i for i, n in enumerate(names) if n == needle]
        else:
            k = needle.lower()
            hits = [i for i, n in enumerate(names) if k in n.lower()]
        if not hits:
            raise KeyError(f"no map matching {needle!r}; have {names}")
        if len(hits) > 1 and not exact:
            raise KeyError(f"{needle!r} is ambiguous: "
                           f"{[names[i] for i in hits]}")
        return hits[0]

    def __repr__(self):
        return (f"<Dscalar {self.n_rows} maps x {self.n_cols} grayordinates "
                f"dtype={self.dtype.str} data@{self.data_start}>")


def open_header(first_bytes):
    """Parse a CIFTI-2 header from the leading bytes of a file.

    Raises NeedMoreBytes if the prefix is too short to hold the XML extension - HCP's
    level-2 dscalar XML is ~720 KB, far larger than a NIfTI header, so a caller must be
    prepared to widen its range request.
    """
    hdr = parse_header(first_bytes)
    end, ext = _walk_extensions(first_bytes, hdr, True)
    if 32 not in ext:
        raise ValueError("no CIFTI-2 XML extension (ecode 32) found - not a CIFTI file")
    return Dscalar(hdr, parse_cifti_xml(ext[32]), data_start=end)


def read_file(path):
    """Read a whole local CIFTI file. Returns (Dscalar, ndarray[n_rows, n_cols])."""
    with open(path, "rb") as f:
        head = f.read(8192)
        while True:
            try:
                d = open_header(head)
                break
            except NeedMoreBytes as e:
                f.seek(0)
                head = f.read(e.needed)
        f.seek(d.data_start)
        raw = f.read(d.data_nbytes)
    return d, d.decode_matrix(raw)


# --------------------------------------------------------------------- writing
def write_dscalar(path, data, map_names, structures=None):
    """Write a minimal but standards-shaped CIFTI-2 dscalar.

    Used by 99_selftest.py to build synthetic fixtures, so the reader is exercised
    against real bytes rather than against a mock.
    """
    data = np.asarray(data, np.float32)
    if data.ndim != 2:
        raise ValueError("data must be 2-D (n_maps, n_grayordinates)")
    n_rows, n_cols = data.shape
    if len(map_names) != n_rows:
        raise ValueError("one map name per row required")
    structures = structures or [dict(name="CIFTI_STRUCTURE_CORTEX_LEFT",
                                     index_offset=0, index_count=n_cols,
                                     model_type="CIFTI_MODEL_TYPE_SURFACE")]
    bm = "".join(
        f'<BrainModel IndexOffset="{s["index_offset"]}" IndexCount="{s["index_count"]}" '
        f'ModelType="{s["model_type"]}" BrainStructure="{s["name"]}" '
        f'SurfaceNumberOfVertices="32492"><VertexIndices>'
        + " ".join(str(i) for i in range(s["index_count"]))
        + "</VertexIndices></BrainModel>"
        for s in structures)
    nm = "".join(f"<NamedMap><MapName>{n}</MapName></NamedMap>" for n in map_names)
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<CIFTI Version="2"><Matrix>'
        '<MatrixIndicesMap AppliesToMatrixDimension="0" '
        'IndicesMapToDataType="CIFTI_INDEX_TYPE_SCALARS">' + nm + '</MatrixIndicesMap>'
        '<MatrixIndicesMap AppliesToMatrixDimension="1" '
        'IndicesMapToDataType="CIFTI_INDEX_TYPE_BRAIN_MODELS">' + bm + '</MatrixIndicesMap>'
        '</Matrix></CIFTI>').encode()
    esize = ((8 + len(xml) + 15) // 16) * 16
    vox_offset = NIFTI2_HDR_SIZE + 4 + esize

    h = bytearray(NIFTI2_HDR_SIZE)
    struct.pack_into("<i", h, 0, NIFTI2_HDR_SIZE)
    h[4:12] = NIFTI2_MAGIC
    struct.pack_into("<hh", h, 12, 16, 32)                     # float32 / 32 bits
    struct.pack_into("<8q", h, 16, 6, 1, 1, 1, 1, n_rows, n_cols, 1)
    struct.pack_into("<8d", h, 168, 1, 1, 1, 1, 1, 1, 1, 1)
    struct.pack_into("<q", h, 232, vox_offset)
    struct.pack_into("<i", h, 344, 1)                          # scl_slope
    with open(path, "wb") as f:
        f.write(bytes(h))
        f.write(b"\x01\x00\x00\x00")
        f.write(struct.pack("<ii", esize, 32))
        f.write(xml.ljust(esize - 8, b"\x00"))
        # interleaved: grayordinate-major, map index fastest - the real CIFTI order
        f.write(np.ascontiguousarray(data.T, "<f4").tobytes())
    return vox_offset
