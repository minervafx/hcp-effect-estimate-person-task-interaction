#!/usr/bin/env python
"""PHASE 3.0 / STAGE 1 - stream HCP Open Access resting-state runs and reduce them to
379-parcel timeseries.

Pre-registration sections 2, 3, 4.2 and 10.  The dense 91,282-grayordinate timeseries is
never written to disk and never held whole in memory: the file is read in grayordinate
blocks by signed byte-range GET and reduced to parcel means on the fly.

Retained per run: `results/phase3_0/cache/{subject}_{session}_{run}.npz` holding the
(T, 379) float32 parcellated timeseries plus provenance.  ~1.8 MB each.

Credentials come from ~/.hcp_aws.env (or $HCP_CRED_FILE) via hcpee.hcp_s3 and are
never logged.

[release] Path/import wiring changed (hcpee/paths.py): the cache goes to
$HCPEE_REST_CACHE and the log to $HCPEE_OUT. Optional HCPEE_REST_ONLY
("subject:session:run,..." ) restricts the job list, e.g. for spot checks.
Parcellation and FC construction unchanged. See docs/PATCHES.md.
"""
import concurrent.futures as cf
import json
import os
import shutil
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "hcpee"))
import paths                                                          # noqa: E402
import p30                                                            # noqa: E402

N_WORKERS = 4
MIN_FREE_GB = 3.0                     # pre-registration section 10 abort threshold
_lock = threading.Lock()
_tl = threading.local()


def s3():
    if not hasattr(_tl, "s3"):
        _tl.s3 = p30.hcp_s3.S3()
    return _tl.s3


def free_gb(path=paths.REST_CACHE):
    st = shutil.disk_usage(path)
    return st.free / 1e9


def out_path(subject, session, run):
    return os.path.join(p30.CACHE, f"{subject}_{session}_{run}.npz")


def fetch_one(job, labels, state):
    subject, session, run = job
    dst = out_path(subject, session, run)
    if os.path.exists(dst):
        return dict(**dict(zip(("subject", "session", "run"), job)),
                    status="cached", bytes=0)
    key = p30.run_key(session, subject, run)
    rec = dict(subject=subject, session=session, run=run, key=key)
    t0 = time.time()
    try:
        before = s3().bytes_transferred
        ts, T = p30.stream_parcellate(s3(), key, labels)
        nbytes = s3().bytes_transferred - before
        # X3 / X4 - length and finiteness, applied here so the record is auditable
        if T < p30.MIN_TIMEPOINTS:
            rec.update(status="excluded", rule="X3", reason=f"T={T} < {p30.MIN_TIMEPOINTS}")
        elif not np.isfinite(ts).all():
            rec.update(status="excluded", rule="X4", reason="non-finite parcel timeseries")
        elif (ts.std(0) <= 0).any():
            rec.update(status="excluded", rule="X4",
                       reason=f"{int((ts.std(0) <= 0).sum())} zero-variance parcels")
        else:
            rec.update(status="ok")
        try:
            m = s3().get(p30.motion_key(session, subject, run))
            rms = float(m.decode().split()[0])
        except Exception:
            rms = float("nan")
        rec.update(n_timepoints=int(T), bytes=int(nbytes),
                   rel_rms_mean=rms, seconds=round(time.time() - t0, 2))
        if rec["status"] == "ok":
            np.savez_compressed(dst, timeseries=ts,
                                provenance=json.dumps(rec).encode("utf-8"))
    except p30.hcp_s3.AccessGateError:
        raise
    except Exception as e:                                            # noqa: BLE001
        rec.update(status="missing" if "404" in str(e) else "error",
                   rule="X1" if "404" in str(e) else None,
                   reason=f"{type(e).__name__}: {str(e)[:200]}",
                   bytes=0, seconds=round(time.time() - t0, 2))
    with _lock:
        state["done"] += 1
        state["bytes"] += rec.get("bytes", 0)
        print(f"[{state['done']:3d}/{state['total']}] {rec['status']:8s} "
              f"{subject} {session:6s} {run:15s} "
              f"T={rec.get('n_timepoints', '-')} "
              f"{rec.get('bytes', 0)/1e6:6.0f}MB {rec.get('seconds', 0):5.1f}s "
              f"| cum {state['bytes']/1e9:5.1f}GB", flush=True)
    return rec


def main():
    os.makedirs(p30.CACHE, exist_ok=True)
    if free_gb() < MIN_FREE_GB:
        raise SystemExit(f"ABORT: only {free_gb():.1f} GB free, need {MIN_FREE_GB}")
    subs = p30.subjects()
    labels = p30.atlas_labels()
    jobs = [(s, ses, r) for s in subs for ses in ("test", "retest")
            for r in p30.ALL_RUNS]
    only = os.environ.get("HCPEE_REST_ONLY")
    if only:
        want = {tuple(x.split(":")) for x in only.split(",") if x}
        jobs = [j for j in jobs if j in want]
    state = dict(done=0, total=len(jobs), bytes=0)
    print(f"Phase 3.0 fetch: {len(jobs)} runs, {len(subs)} subjects, "
          f"{N_WORKERS} workers, {free_gb():.1f} GB free", flush=True)
    t0 = time.time()
    recs = []
    with cf.ThreadPoolExecutor(N_WORKERS) as ex:
        futs = [ex.submit(fetch_one, j, labels, state) for j in jobs]
        for f in cf.as_completed(futs):
            recs.append(f.result())
    recs.sort(key=lambda r: (r["subject"], r["session"], r["run"]))
    summary = dict(
        n_jobs=len(jobs), n_subjects=len(subs),
        n_ok=sum(r["status"] == "ok" for r in recs),
        n_cached=sum(r["status"] == "cached" for r in recs),
        n_missing=sum(r["status"] == "missing" for r in recs),
        n_excluded=sum(r["status"] == "excluded" for r in recs),
        n_error=sum(r["status"] == "error" for r in recs),
        bytes_streamed=state["bytes"], seconds=round(time.time() - t0, 1),
        free_gb_after=round(free_gb(), 2), runs=recs)
    os.makedirs(paths.OUT, exist_ok=True)
    p = os.path.join(paths.OUT, "01_fetch_rest_log.json")
    json.dump(summary, open(p, "w"), indent=2)
    print(f"\nstreamed {state['bytes']/1e9:.2f} GB in {summary['seconds']/60:.1f} min; "
          f"ok={summary['n_ok']} cached={summary['n_cached']} "
          f"missing={summary['n_missing']} excluded={summary['n_excluded']} "
          f"error={summary['n_error']}")
    print("wrote", p)


if __name__ == "__main__":
    main()
