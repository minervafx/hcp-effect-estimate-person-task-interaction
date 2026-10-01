#!/usr/bin/env python
"""Remote zip member extraction over a SigV4-signed S3 object.

Phase 2.7 built `src/phase2_7/rzip.py` to pull single members out of a remote zip by
byte range, over plain HTTP with curl.  The HCP group-average package needs the same
trick, but the object is in a credentialed bucket, so the transport has to be the
signed client in `hcp_s3.py` rather than curl.  The zip logic below is Phase 2.7's,
with the transport swapped and zip64 support added - `HCP_S1200_GroupAvg_v1.zip` is
1.9 GB, which is under the 4 GB zip32 ceiling, but its central directory may still
carry zip64 extras.

Why this matters scientifically: it is what makes the PRIMARY pre-registered
parcellation (HCP-MMP1.0, 360 cortical parcels) reachable under Gate 1 without
downloading 1.9 GB, so Phase 2.9 does not have to fall back to the weaker
21-structure partition declared in PHASE_2_9_PREREGISTRATION.md section 3.
"""
import io
import struct
import zlib


class S3Zip(object):
    def __init__(self, s3, key):
        self.s3 = s3
        self.key = key
        self.size = s3.head_size(key)
        self.entries = {}
        self._read_central_directory()

    def _get(self, start, end_inclusive):
        return self.s3.get_range(self.key, start, end_inclusive + 1)

    def _read_central_directory(self):
        tail_n = min(self.size, 1 << 16)
        tail = self._get(self.size - tail_n, self.size - 1)
        i = tail.rfind(b"PK\x05\x06")
        if i < 0:
            raise RuntimeError("no EOCD found")
        n_ent, cd_size, cd_off = struct.unpack("<HII", tail[i + 10:i + 20])
        # zip64: the EOCD locator sits immediately before the EOCD record
        if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF or n_ent == 0xFFFF:
            j = tail.rfind(b"PK\x06\x07")
            if j < 0:
                raise RuntimeError("zip64 indicated but no EOCD64 locator")
            (eocd64_off,) = struct.unpack("<Q", tail[j + 8:j + 16])
            e64 = self._get(eocd64_off, eocd64_off + 55)
            if e64[:4] != b"PK\x06\x06":
                raise RuntimeError("bad EOCD64 signature")
            n_ent, cd_size, cd_off = struct.unpack("<QQQ", e64[32:56])
        cd = self._get(cd_off, cd_off + cd_size - 1)
        p = 0
        for _ in range(n_ent):
            if cd[p:p + 4] != b"PK\x01\x02":
                break
            (method, _t, _d, crc, csize, usize, nlen, elen, clen,
             _dsk, _ia, _ea, lho) = struct.unpack("<HHHIIIHHHHHII", cd[p + 10:p + 46])
            name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
            extra = cd[p + 46 + nlen:p + 46 + nlen + elen]
            if 0xFFFFFFFF in (csize, usize, lho):
                usize, csize, lho = self._zip64_extra(extra, usize, csize, lho)
            self.entries[name] = dict(method=method, crc=crc, csize=csize,
                                      usize=usize, lho=lho)
            p += 46 + nlen + elen + clen

    @staticmethod
    def _zip64_extra(extra, usize, csize, lho):
        q = 0
        while q + 4 <= len(extra):
            hid, hsz = struct.unpack("<HH", extra[q:q + 4])
            if hid == 0x0001:
                vals, r = [], q + 4
                for cur in (usize, csize, lho):
                    if cur == 0xFFFFFFFF and r + 8 <= q + 4 + hsz:
                        vals.append(struct.unpack("<Q", extra[r:r + 8])[0])
                        r += 8
                    else:
                        vals.append(cur)
                return vals[0], vals[1], vals[2]
            q += 4 + hsz
        return usize, csize, lho

    def namelist(self):
        return sorted(self.entries)

    def read(self, name):
        e = self.entries[name]
        hdr = self._get(e["lho"], e["lho"] + 29)
        if hdr[:4] != b"PK\x03\x04":
            raise RuntimeError("bad local header for " + name)
        nlen, elen = struct.unpack("<HH", hdr[26:30])
        start = e["lho"] + 30 + nlen + elen
        blob = self._get(start, start + e["csize"] - 1)
        if e["method"] == 0:
            data = blob
        elif e["method"] == 8:
            data = zlib.decompressobj(-15).decompress(blob)
        else:
            raise RuntimeError(f"unsupported compression method {e['method']}")
        if len(data) != e["usize"]:
            raise RuntimeError(f"size mismatch for {name}")
        if zlib.crc32(data) & 0xFFFFFFFF != e["crc"]:
            raise RuntimeError(f"CRC mismatch for {name}")
        return data

    def open(self, name):
        return io.BytesIO(self.read(name))
