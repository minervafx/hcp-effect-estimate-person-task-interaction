#!/usr/bin/env python
"""PHASE 3.0 EFFECT-ESTIMATE PORT - Resting-state control on effect-estimate interaction.

Mechanical port of src/phase3_0/02_rest_control.py to effect-estimate space.
Only change: task-fMRI interaction input uses effect-estimate maps instead of z-statistic maps.
Everything else preserved exactly from canonical Phase 3.0.

[release] Path/import wiring changed (hcpee/paths.py); the task cache is accepted only
after hcpee.map_type_guard verifies it holds COPE effect estimates. Estimation code,
seeds and permutation counts are unchanged. See docs/PATCHES.md.
"""
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "hcpee"))
import paths                                                          # noqa: E402
import map_type_guard                                                 # noqa: E402
import ix                                                             # noqa: E402
import p30                                                            # noqa: E402
P29 = p30.P29

RES = paths.OUT
GUARD = {}
PREREG = "HCP_EFFECT_ESTIMATE_CONTROLS_PREREG.md"
T = list(P29.TRANSITIONS)
PT = P29.PRIMARY_TRANSITION
K_LADDER = (5, 10, 20)
K_PRIMARY = 10
SEED = P29.SEED
NPERM = 10000


def prereg_sha():
    return paths.prereg_sha(PREREG)


def load_rest():
    """Return (fc, avail) -- fc[(subject, session, group)] = Fisher-z vector."""
    fc, avail = {}, {}
    for f in sorted(glob.glob(os.path.join(p30.CACHE, "*.npz"))):
        stem = os.path.basename(f)[:-4]
        subject, session, run = stem.split("_", 2)
        z = np.load(f)
        ts = z["timeseries"]
        prov = json.loads(bytes(z["provenance"]).decode("utf-8"))
        group = "REST1" if "REST1" in run else "REST2"
        avail.setdefault((subject, session, group), []).append(
            dict(run=run, n_timepoints=int(ts.shape[0]),
                 rel_rms_mean=prov.get("rel_rms_mean")))
        fc.setdefault((subject, session, group), []).append(p30.fc_vector(ts))
    out = {}
    for key, vecs in fc.items():
        out[key] = np.mean(np.stack(vecs), 0)
    return out, avail


def rest_block(fc, subs, session, group):
    return np.stack([fc[(s, session, group)] for s in subs]).astype(np.float64)


def removal(Pd, Qd, Bt, Br, k, fit="test", n_perm=2000):
    """One rung: fit PCA on `fit` session's nuisance block, residualise, re-evaluate."""
    src = Bt if fit == "test" else Br
    mu, comps = P29.pca_fit_test_only(src, k)
    Xp, Xq = P29.pca_apply(Bt, mu, comps), P29.pca_apply(Br, mu, comps)
    Pr = {t: P29.residualise(Pd[t], Xp) for t in T}
    Qr = {t: P29.residualise(Qd[t], Xq) for t in T}
    tr = P29.transfer(Pr[PT], Qr[PT], n_perm=n_perm)
    sp = P29.specificity(Pr, Qr, n_perm=NPERM)
    return dict(k=int(k), fit_session=fit, transfer=tr, specificity=sp,
                cv_r2_block_predicts_delta_test=P29.cv_r2(Xp, Pd[PT]),
                cv_r2_block_predicts_delta_retest=P29.cv_r2(Xq, Qd[PT])), (Xp, Xq)


def permuted_nuisance_null(Pd, Qd, Bt, Br, k, n_draws=200, seed=SEED):
    """NC2: shuffle the nuisance block's subject labels, then remove."""
    rng = np.random.default_rng(seed)
    n = Bt.shape[0]
    out = []
    for _ in range(n_draws):
        pi = rng.permutation(n)
        mu, comps = P29.pca_fit_test_only(Bt[pi], k)
        Xp = P29.pca_apply(Bt[pi], mu, comps)
        Xq = P29.pca_apply(Br[pi], mu, comps)
        out.append(P29.transfer(P29.residualise(Pd[PT], Xp),
                                P29.residualise(Qd[PT], Xq), n_perm=200)["rank1"])
    a = np.asarray(out)
    return dict(k=int(k), n_draws=int(n_draws), mean=float(a.mean()),
                p5=float(np.quantile(a, .05)), p95=float(np.quantile(a, .95)))


def simmat_corr(A_t, A_r, B_t, B_r, n_perm=NPERM, seed=SEED):
    """S5: correlation of two cross-session similarity matrices, off-diagonal only."""
    Sa, Sb = P29.simmat(A_t, A_r), P29.simmat(B_t, B_r)
    n = Sa.shape[0]
    off = ~np.eye(n, dtype=bool)
    a, b = Sa[off].ravel(), Sb[off].ravel()
    r = float(np.corrcoef(a, b)[0, 1])
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        pi = rng.permutation(n)
        rp = float(np.corrcoef(Sa[np.ix_(pi, pi)][off].ravel(), b)[0, 1])
        ge += abs(rp) >= abs(r)
    return dict(r=r, p_subject_permutation=float((ge + 1) / (n_perm + 1)),
                n_perm=int(n_perm), n_pairs=int(off.sum()))


def load_task_effect_estimate(subs):
    """Load effect-estimate (COPE) cache, read-only."""
    data = {}
    cache_dir = paths.EE_CACHE
    GUARD.update(map_type_guard.assert_effect_estimate_cache(cache_dir, T, subs))
    for s in subs:
        for ses in ("test", "retest"):
            f = os.path.join(cache_dir, f"{s}_{ses}.npz")
            data[(s, ses)] = dict(np.load(f))
    return data


def main():
    t0 = time.time()
    os.makedirs(RES, exist_ok=True)
    subs_all = p30.subjects()
    task = load_task_effect_estimate(subs_all)
    fc, avail = load_rest()

    # ------------------------------------------------- exclusions X5 / X6
    def usable(s, group):
        return all((s, ses, group) in fc for ses in ("test", "retest"))
    subs = [s for s in subs_all if usable(s, "REST1")]
    subs_r2 = [s for s in subs_all if usable(s, "REST2")]
    n = len(subs)
    out = {"prereg_sha256": prereg_sha(),
           "prereg_expected_sha256": paths.PREREG_EXPECTED_SHA256[PREREG],
           "map_type_guard": GUARD,
           "seed": SEED, "n_perm": NPERM, "k_ladder": list(K_LADDER),
           "k_primary": K_PRIMARY, "primary_transition": PT, "transitions": T,
           "n_subjects_task": len(subs_all), "n_subjects_primary": n,
           "n_subjects_rest2": len(subs_r2), "chance_rank1": 1.0 / n,
           "subjects_primary": subs,
           "excluded_from_primary": sorted(set(subs_all) - set(subs)),
           "excluded_from_rest2": sorted(set(subs_all) - set(subs_r2)),
           "run_availability": {f"{a}|{b}|{c}": v for (a, b, c), v in avail.items()}}
    print(f"n(task)={len(subs_all)}  n(REST1 both sessions)={n}  "
          f"n(REST2 both sessions)={len(subs_r2)}  chance={1/n:.5f}", flush=True)

    M = lambda ses, key, ss: np.stack([task[(s, ses)][key] for s in ss]).astype(float)
    Pd = {t: M("test", f"{t}__diff", subs) for t in T}
    Qd = {t: M("retest", f"{t}__diff", subs) for t in T}

    # ------------------------------------------------- baselines on this subset
    out["baseline"] = dict(
        transfer=P29.transfer(Pd[PT], Qd[PT], n_perm=NPERM),
        specificity=P29.specificity(Pd, Qd, n_perm=NPERM)["pooled"])
    print("baseline rank1=%.4f auc=%.4f | SS=%+.4f SD=%+.4f DS=%+.4f "
          "AUC(SS,SD)=%.4f AUC(SS,DS)=%.4f" % (
              out["baseline"]["transfer"]["rank1"], out["baseline"]["transfer"]["auc"],
              out["baseline"]["specificity"]["ss"], out["baseline"]["specificity"]["sd"],
              out["baseline"]["specificity"]["ds"],
              out["baseline"]["specificity"]["auc_ss_sd"],
              out["baseline"]["specificity"]["auc_ss_ds"]), flush=True)

    # ------------------------------------------------- nuisance blocks
    zr = P29.zscore_rows
    othertask = {ses: np.concatenate(
        [zr(M(ses, f"{t}__{l}", subs)) for t in T if t != PT for l in ("A", "B")], 1)
        for ses in ("test", "retest")}
    R1 = {ses: rest_block(fc, subs, ses, "REST1") for ses in ("test", "retest")}
    R2 = ({ses: rest_block(fc, subs_r2, ses, "REST2") for ses in ("test", "retest")}
          if subs_r2 else None)

    def zblock(X):
        return (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
    combined = {ses: np.concatenate([zblock(R1[ses]), zblock(othertask[ses])], 1)
                for ses in ("test", "retest")}

    # ------------------------------------------------- NC1 nuisance-model power
    out["NC1_nuisance_power"] = dict(
        rest_REST1=P29.transfer(R1["test"], R1["retest"], n_perm=NPERM),
        othertask_static=P29.transfer(othertask["test"], othertask["retest"],
                                      n_perm=NPERM),
        threshold_rank1=0.50)
    if R2 is not None:
        out["NC1_nuisance_power"]["rest_REST2"] = P29.transfer(
            R2["test"], R2["retest"], n_perm=NPERM)
    print("NC1 rest REST1 rank1=%.4f auc=%.4f | othertask static rank1=%.4f" % (
        out["NC1_nuisance_power"]["rest_REST1"]["rank1"],
        out["NC1_nuisance_power"]["rest_REST1"]["auc"],
        out["NC1_nuisance_power"]["othertask_static"]["rank1"]), flush=True)

    # ------------------------------------------------- PRIMARY + S1 (k ladder)
    out["df_matched_null"] = {}
    out["primary_REST1"] = {}
    for k in K_LADDER:
        r, _ = removal(Pd, Qd, R1["test"], R1["retest"], k, fit="test")
        nl = P29.df_matched_null(Pd[PT], Qd[PT], k)
        r["df_matched_null"] = nl
        r["below_null_p5"] = bool(r["transfer"]["rank1"] < nl["p5"])
        r["inside_null"] = bool(nl["p5"] <= r["transfer"]["rank1"] <= nl["p95"])
        out["df_matched_null"][f"k{k}"] = nl
        out["primary_REST1"][f"k{k}"] = r
        p = r["specificity"]["pooled"]
        print("REST1 k=%-2d rank1=%.4f auc=%.4f dfnull=%.4f[%.4f,%.4f] below=%s | "
              "SS=%+.4f SD=%+.4f DS=%+.4f AUC_SD=%.4f AUC_DS=%.4f cvR2=%+.4f" % (
                  k, r["transfer"]["rank1"], r["transfer"]["auc"], nl["mean"],
                  nl["p5"], nl["p95"], r["below_null_p5"], p["ss"], p["sd"], p["ds"],
                  p["auc_ss_sd"], p["auc_ss_ds"],
                  r["cv_r2_block_predicts_delta_retest"]), flush=True)

    # ------------------------------------------------- S3 REST2 replication
    if R2 is not None:
        Pd2 = {t: M("test", f"{t}__diff", subs_r2) for t in T}
        Qd2 = {t: M("retest", f"{t}__diff", subs_r2) for t in T}
        out["S3_REST2"] = {"n_subjects": len(subs_r2),
                           "baseline": P29.transfer(Pd2[PT], Qd2[PT], n_perm=NPERM)}
        for k in K_LADDER:
            r, _ = removal(Pd2, Qd2, R2["test"], R2["retest"], k, fit="test")
            nl = P29.df_matched_null(Pd2[PT], Qd2[PT], k)
            r["df_matched_null"] = nl
            r["below_null_p5"] = bool(r["transfer"]["rank1"] < nl["p5"])
            out["S3_REST2"][f"k{k}"] = r
            p = r["specificity"]["pooled"]
            print("REST2 k=%-2d rank1=%.4f dfnull=%.4f[%.4f,%.4f] below=%s | "
                  "AUC_SD=%.4f AUC_DS=%.4f" % (
                      k, r["transfer"]["rank1"], nl["mean"], nl["p5"], nl["p95"],
                      r["below_null_p5"], p["auc_ss_sd"], p["auc_ss_ds"]), flush=True)

    # ------------------------------------------------- S4 combined block
    out["S4_combined_block"] = {}
    for k in K_LADDER:
        r, _ = removal(Pd, Qd, combined["test"], combined["retest"], k, fit="test")
        r["df_matched_null"] = out["df_matched_null"][f"k{k}"]
        r["below_null_p5"] = bool(
            r["transfer"]["rank1"] < out["df_matched_null"][f"k{k}"]["p5"])
        out["S4_combined_block"][f"k{k}"] = r
        p = r["specificity"]["pooled"]
        print("COMB  k=%-2d rank1=%.4f below=%s | AUC_SD=%.4f AUC_DS=%.4f" % (
            k, r["transfer"]["rank1"], r["below_null_p5"],
            p["auc_ss_sd"], p["auc_ss_ds"]), flush=True)

    # ------------------------------------------------- S5 similarity structure
    out["S5_simmat_corr"] = dict(
        rest_vs_delta=simmat_corr(R1["test"], R1["retest"], Pd[PT], Qd[PT]),
        othertask_vs_delta=simmat_corr(othertask["test"], othertask["retest"],
                                       Pd[PT], Qd[PT]))
    print("S5 corr(rest simmat, delta simmat) = %+.4f (p=%.4g)" % (
        out["S5_simmat_corr"]["rest_vs_delta"]["r"],
        out["S5_simmat_corr"]["rest_vs_delta"]["p_subject_permutation"]), flush=True)

    # ------------------------------------------------- NC2 permuted nuisance
    out["NC2_permuted_nuisance"] = {
        f"k{k}": permuted_nuisance_null(Pd, Qd, R1["test"], R1["retest"], k)
        for k in K_LADDER}
    for k in K_LADDER:
        a, b = out["NC2_permuted_nuisance"][f"k{k}"], out["df_matched_null"][f"k{k}"]
        a["inside_df_matched_null"] = bool(b["p5"] <= a["mean"] <= b["p95"])
        print("NC2 k=%-2d permuted-rest removal mean rank1=%.4f [%.4f,%.4f] "
              "inside df-null=%s" % (k, a["mean"], a["p5"], a["p95"],
                                     a["inside_df_matched_null"]), flush=True)

    # ------------------------------------------------- NC3 / NC4
    rng = np.random.default_rng(SEED)
    pi = rng.permutation(Pd[PT].shape[1])
    out["NC3_parcel_scramble"] = P29.transfer(Pd[PT][:, pi], Qd[PT], n_perm=NPERM)
    out["NC4_wrong_transition"] = P29.transfer(Pd[PT], Qd["LANG_STORY_minus_MATH"],
                                               n_perm=NPERM)
    print("NC3 parcel scramble rank1=%.4f | NC4 wrong transition rank1=%.4f | "
          "chance=%.4f" % (out["NC3_parcel_scramble"]["rank1"],
                           out["NC4_wrong_transition"]["rank1"], 1 / n), flush=True)

    # ------------------------------------------------- NC5 fit-direction leakage
    out["NC5_retest_fit"] = {}
    for k in K_LADDER:
        r, _ = removal(Pd, Qd, R1["test"], R1["retest"], k, fit="retest")
        r["df_matched_null"] = out["df_matched_null"][f"k{k}"]
        r["below_null_p5"] = bool(
            r["transfer"]["rank1"] < out["df_matched_null"][f"k{k}"]["p5"])
        out["NC5_retest_fit"][f"k{k}"] = r
        p = r["specificity"]["pooled"]
        print("NC5 k=%-2d (retest-fit) rank1=%.4f below=%s AUC_SD=%.4f AUC_DS=%.4f" % (
            k, r["transfer"]["rank1"], r["below_null_p5"],
            p["auc_ss_sd"], p["auc_ss_ds"]), flush=True)

    # ------------------------------------------------- S7 motion sensitivity
    def mean_rms(s, ses, group):
        v = [d["rel_rms_mean"] for d in avail.get((s, ses, group), [])
             if d["rel_rms_mean"] is not None and np.isfinite(d["rel_rms_mean"])]
        return float(np.mean(v)) if v else float("nan")
    keep = [s for s in subs
            if all(np.isfinite(mean_rms(s, ses, "REST1")) and
                   mean_rms(s, ses, "REST1") <= 0.2 for ses in ("test", "retest"))]
    out["S7_motion"] = dict(threshold_rel_rms_mm=0.2, n_kept=len(keep),
                            n_dropped=n - len(keep),
                            dropped=sorted(set(subs) - set(keep)))
    if len(keep) >= 20:
        Pk = {t: M("test", f"{t}__diff", keep) for t in T}
        Qk = {t: M("retest", f"{t}__diff", keep) for t in T}
        Rk = {ses: rest_block(fc, keep, ses, "REST1") for ses in ("test", "retest")}
        out["S7_motion"]["baseline"] = P29.transfer(Pk[PT], Qk[PT], n_perm=NPERM)
        r, _ = removal(Pk, Qk, Rk["test"], Rk["retest"], K_PRIMARY, fit="test")
        nl = P29.df_matched_null(Pk[PT], Qk[PT], K_PRIMARY)
        r["df_matched_null"] = nl
        r["below_null_p5"] = bool(r["transfer"]["rank1"] < nl["p5"])
        out["S7_motion"][f"k{K_PRIMARY}"] = r
        p = r["specificity"]["pooled"]
        print("S7 motion n=%d rank1=%.4f below=%s AUC_SD=%.4f AUC_DS=%.4f" % (
            len(keep), r["transfer"]["rank1"], r["below_null_p5"],
            p["auc_ss_sd"], p["auc_ss_ds"]), flush=True)
    else:
        out["S7_motion"]["skipped"] = "fewer than 20 subjects survive the threshold"

    out["NC7_restricted_data_accessed"] = False
    out["seconds"] = round(time.time() - t0, 1)
    out_path = os.path.join(RES, "02_rest_control.json")
    json.dump(out, open(out_path, "w"), indent=2, default=float)
    print("\nwrote", out_path, "in %.1f min" % (out["seconds"] / 60))


if __name__ == "__main__":
    main()