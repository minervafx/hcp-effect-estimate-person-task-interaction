#!/usr/bin/env python
"""Compare a completed run ($HCPEE_OUT) against expected/expected_values.json.

Exit status 0 only if every value matches within the stated absolute tolerance.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "hcpee"))
import paths                                                          # noqa: E402


def get(d, dotted):
    for part in dotted.split("."):
        d = d[int(part)] if isinstance(d, list) else d[part]
    return d


def main():
    spec = json.load(open(os.path.join(paths.REPO, "expected", "expected_values.json")))
    tol = spec["tolerance_abs"]
    cache, rows, ok = {}, [], True
    for fn, key, want, label in spec["checks"]:
        if fn not in cache:
            cache[fn] = json.load(open(os.path.join(paths.OUT, fn)))
        got = get(cache[fn], key)
        good = abs(float(got) - float(want)) <= tol
        ok &= good
        rows.append(dict(file=fn, key=key, label=label, expected=want, got=got, match=good))
    A = cache["01_anatomy_interaction.json"]
    for label, expr, want in spec["derived"]:
        nm = "R_ORIG" if "R_ORIG" in expr else "R_REST"
        got = 100 * (1 - A[nm]["after"]["idiff"] / A[nm]["before"]["idiff"])
        good = abs(round(got, 4) - want) < 1e-9
        ok &= good
        rows.append(dict(file="01_anatomy_interaction.json", key=expr, label=label,
                         expected=want, got=got, match=good))
    for r in rows:
        print(("OK  " if r["match"] else "FAIL") + f"  {r['label']}: got {r['got']!r} "
              f"expected {r['expected']!r}")
    out = dict(all_match=bool(ok), n_checks=len(rows), rows=rows)
    json.dump(out, open(os.path.join(paths.OUT, "20_verification.json"), "w"), indent=1)
    print("ALL MATCH" if ok else "MISMATCH", f"({len(rows)} checks)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
