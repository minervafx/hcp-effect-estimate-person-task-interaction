#!/usr/bin/env python
"""PHASE 3.2 EFFECT-ESTIMATE PORT - Anatomy control in interaction space on effect-estimate maps.

Mechanical port of src/phase3_2/01_anatomy_interaction.py to effect-estimate space.
Only change: task-fMRI interaction input uses effect-estimate maps instead of z-statistic maps.
Everything else preserved exactly from canonical Phase 3.2 (corrected modern anatomy model).

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
import an                                                             # noqa: E402
import ix                                                             # noqa: E402
import p30                                                            # noqa: E402
P29 = p30.P29

RES = paths.OUT
GUARD = {}
PREREG = "HCP_EFFECT_ESTIMATE_CONTROLS_PREREG.md"
T = list(P29.TRANSITIONS)
SEED = P29.SEED
NPERM = int(os.environ.get("NPERM", "10000"))
NBOOT = int(os.environ.get("NBOOT", "10000"))
BSHUF = int(os.environ.get("BSHUF", "1000"))
K_PRIMARY = 20
LAM_PRIMARY = 1.0
K_LADDER = (5, 10, 20)
LAM_LADDER = (1.0, 10.0, 50.0)


def prereg_sha():
    return paths.prereg_sha(PREREG)


def stack(task, ses, subs):
    return np.stack([[task[(s, ses)][f"{t}__diff"] for s in subs]
                     for t in T]).astype(float)


def load_rest():
    fc = {}
    for f in sorted(glob.glob(os.path.join(p30.CACHE, "*.npz"))):
        subject, session, run = os.path.basename(f)[:-4].split("_", 2)
        z = np.load(f)
        g = "REST1" if "REST1" in run else "REST2"
        fc.setdefault((subject, session, g), []).append(p30.fc_vector(z["timeseries"]))
    return {k: np.mean(np.stack(v), 0) for k, v in fc.items()}


def load_anatomy(subs):
    out = []
    for ses in ("test", "retest"):
        sfx = "" if ses == "test" else "_retest"
        z = np.load(os.path.join(paths.ANAT_DIR, f"anatomy{sfx}.npz"),
                    allow_pickle=True)
        idx = {s: i for i, s in enumerate([str(x) for x in z["subjects"]])}
        A = np.concatenate([z["blockA"], z["blockB"]], 1).astype(float)
        out.append(A[[idx[s] for s in subs]])
    (At, Ar), keep = an.clean_anatomy(out)
    return At, Ar, int(keep.sum())


def spec_check(Xt, Xr):
    p = P29.specificity({t: Xt[i] for i, t in enumerate(T)},
                        {t: Xr[i] for i, t in enumerate(T)}, n_perm=1000)["pooled"]
    return {k: round(float(p[k]), 4) for k in
            ("ss", "sd", "ds", "auc_ss_sd", "auc_ss_ds")}


def null_person_from_S(S, n_perm, seed):
    rng = np.random.default_rng(seed)
    n = S.shape[1]
    r1 = np.empty(n_perm)
    idf = np.empty(n_perm)
    for b in range(n_perm):
        pi = rng.permutation(n)
        r1[b] = ix._rank1_from_S_perm(S, pi)
        idf[b] = ix.idiff_from_S_perm(S, pi)
    return r1, idf


def survival(S, tag, n_perm=NPERM, full=True):
    st = ix.primary_stats(S)
    r1n, idn = null_person_from_S(S, n_perm, SEED)
    out = dict(tag=tag, **st)
    out["null_person_rank1"] = dict(n_perm=int(n_perm), mean=float(r1n.mean()),
                                    sd=float(r1n.std()),
                                    p95=float(np.quantile(r1n, .95)),
                                    max=float(r1n.max()))
    out["null_person_idiff"] = dict(n_perm=int(n_perm), mean=float(idn.mean()),
                                    sd=float(idn.std()),
                                    p99=float(np.quantile(idn, .99)),
                                    max=float(idn.max()))
    out["p_idiff"] = ix.perm_p(st["idiff"], idn)
    out["p_rank1"] = ix.perm_p(st["rank1"], r1n)
    if full:
        ptn = ix.per_transition_null_fast(S, n_perm, SEED)
        ps = [ix.perm_p(st["per_transition_rank1"][i], ptn[:, i]) for i in range(len(T))]
        qs = ix.bh_fdr(ps)
        out["per_transition"] = [
            dict(transition=T[i], rank1=st["per_transition_rank1"][i],
                 null_mean=float(ptn[:, i].mean()), p=float(ps[i]),
                 q_bh_fdr=float(qs[i])) for i in range(len(T))]
        out.update(ix.boot_ci(S, n_boot=NBOOT, seed=SEED))
    out["PASSES"] = bool(
        out["p_idiff"] <= 0.001 and st["idiff"] > 0
        and out["p_rank1"] <= 0.01 and st["rank1"] > st["chance"]
        and st["auc"] >= 0.55
        and sum(x > st["chance"] for x in st["per_transition_rank1"]) >= 3
        and (out["idiff_ci"][0] > 0 if full else True))
    return out


def var_removed(I, J):
    return float(1.0 - (np.asarray(J) ** 2).sum() / max((np.asarray(I) ** 2).sum(), 1e-20))


def run_removal(It, Ir, Ut, Ur, tau=None):
    n = It.shape[1]
    tau = np.arange(n) if tau is None else tau
    return (an.remove_perm_fast(It, Ut, tau), an.remove_perm_fast(Ir, Ur, tau))


def attribution(It, Ir, Ut, Ur, before, after, b_shuf, seed):
    rng = np.random.default_rng(seed)
    n = It.shape[1]
    d_id = before["idiff"] - after["idiff"]
    d_r1 = before["rank1"] - after["rank1"]
    nid = np.empty(b_shuf)
    nr1 = np.empty(b_shuf)
    for b in range(b_shuf):
        tau = rng.permutation(n)
        st = ix.primary_stats(ix.cell_simmats(*run_removal(It, Ir, Ut, Ur, tau)))
        nid[b] = before["idiff"] - st["idiff"]
        nr1[b] = before["rank1"] - st["rank1"]
    return dict(b_shuf=int(b_shuf), d_idiff=float(d_id), d_rank1=float(d_r1),
                null_d_idiff=dict(mean=float(nid.mean()), sd=float(nid.std()),
                                  p95=float(np.quantile(nid, .95)),
                                  max=float(nid.max())),
                null_d_rank1=dict(mean=float(nr1.mean()), sd=float(nr1.std()),
                                  p95=float(np.quantile(nr1, .95)),
                                  max=float(nr1.max())),
                p_attrib_idiff=float((1 + (nid >= d_id - 1e-12).sum()) / (1 + b_shuf)),
                p_attrib_rank1=float((1 + (nr1 >= d_r1 - 1e-12).sum()) / (1 + b_shuf)))


def analyse(Xt, Xr, At, Ar, tag, k=K_PRIMARY, lam=LAM_PRIMARY, loso=False,
            b_shuf=BSHUF, full=True, n_perm=NPERM):
    t0 = time.time()
    It, Ir = ix.two_way_residual(Xt, loso), ix.two_way_residual(Xr, loso)
    Ut = an.precompute_smoothers(At, k, lam)
    Ur = an.precompute_smoothers(Ar, k, lam)
    S_before = ix.cell_simmats(It, Ir)
    before = survival(S_before, tag + " BEFORE", n_perm=n_perm, full=full)
    Jt, Jr = run_removal(It, Ir, Ut, Ur)
    S_after = ix.cell_simmats(Jt, Jr)
    after = survival(S_after, tag + " AFTER", n_perm=n_perm, full=full)
    out = dict(tag=tag, k=int(k), lam=float(lam),
               loso_transition_mean=bool(loso), before=before, after=after,
               variance_removed_test=var_removed(It, Jt),
               variance_removed_retest=var_removed(Ir, Jr),
               cv_r2_test=an.crossfit_r2(It, At, k, lam),
               cv_r2_retest=an.crossfit_r2(Ir, Ar, k, lam))
    if b_shuf:
        out["attribution"] = attribution(It, Ir, Ut, Ur, before, after, b_shuf, SEED)
    out["seconds"] = round(time.time() - t0, 1)
    a = out.get("attribution", {})
    print("  [%-34s] rank1 %.4f->%.4f  Idiff %+7.3f->%+7.3f  AUC %.4f->%.4f | "
          "PASS %s->%s | p_attrib %s | cvR2 %+.4f | %.0fs"
          % (tag, before["rank1"], after["rank1"], before["idiff"], after["idiff"],
             before["auc"], after["auc"], before["PASSES"], after["PASSES"],
             ("%.4f" % a["p_attrib_idiff"]) if a else "n/a",
             out["cv_r2_test"]["pooled"], out["seconds"]), flush=True)
    return out


def diagnostics(Xt, Xr, At, Ar, tag):
    It, Ir = ix.two_way_residual(Xt), ix.two_way_residual(Xr)
    S_I = ix.cell_simmats(It, Ir)
    S_anat = an.anat_simmat(At, Ar)
    rng = np.random.default_rng(SEED)
    n = S_I.shape[1]
    per = []
    for t in range(len(T)):
        r = an.mantel_offdiag(S_anat, S_I[t])
        nullr = []
        for _ in range(2000):
            pi = rng.permutation(n)
            nullr.append(an.mantel_offdiag(S_anat, S_I[t][:, pi][pi]))
        nullr = np.asarray(nullr)
        per.append(dict(transition=T[t], mantel_offdiag_r=float(r),
                        null_mean=float(nullr.mean()), null_sd=float(nullr.std()),
                        p_two_sided=float((1 + (np.abs(nullr) >= abs(r)).sum())
                                          / (1 + nullr.size))))
    imp = an.impostor_rank(S_I, S_anat)
    print("  [%-20s] Mantel off-diag r: %s | impostor rank %.4f (null 0.5), "
          "spearman %+.4f" % (tag, ["%.4f" % x["mantel_offdiag_r"] for x in per],
                              imp["mean_normalised_rank"], imp["mean_spearman"]),
          flush=True)
    return dict(B1_mantel_offdiag=per, B2_impostor=imp)


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
    subs = p30.subjects()
    task = load_task_effect_estimate(subs)
    fc = load_rest()
    At, Ar, q = load_anatomy(subs)
    fp1 = an.assert_shuffle_fast_path_exact()
    fp2 = an.assert_smoother_fast_path_exact()
    assert fp1["ok"] and fp2["ok"], (fp1, fp2)

    out = dict(prereg_sha256=prereg_sha(),
               prereg_expected_sha256=paths.PREREG_EXPECTED_SHA256[PREREG],
               map_type_guard=GUARD, seed=SEED, n_perm=NPERM, n_boot=NBOOT,
               b_shuf=BSHUF, k_primary=K_PRIMARY, lam_primary=LAM_PRIMARY,
               transitions=T, n_subjects=len(subs), subjects=subs,
               chance_rank1=1.0 / len(subs), anatomy_features_kept=q,
               shuffle_fast_path_exact=fp1, smoother_fast_path_exact=fp2,
               NC8_restricted_data_accessed=False)
    print("n=%d  anatomy kept=%d  k=%d lam=%.1f  fastpaths=%s/%s"
          % (len(subs), q, K_PRIMARY, LAM_PRIMARY, fp1["ok"], fp2["ok"]), flush=True)

    # NC1 - anatomical nuisance power
    out["NC1_anatomy_identification"] = {
        k: (float(v) if isinstance(v, (int, float)) else v)
        for k, v in P29.transfer(At, Ar, n_perm=2000).items()}
    print("NC1 anatomy cross-session rank1 = %.4f, AUC = %.4f"
          % (out["NC1_anatomy_identification"]["rank1"],
             out["NC1_anatomy_identification"]["auc"]), flush=True)

    Xt, Xr = stack(task, "test", subs), stack(task, "retest", subs)
    out["execution_check_baseline"] = spec_check(Xt, Xr)
    Bt = np.stack([fc[(s, "test", "REST1")] for s in subs]).astype(float)
    Br = np.stack([fc[(s, "retest", "REST1")] for s in subs]).astype(float)
    mu, comps = P29.pca_fit_test_only(Bt, 10)
    Pp, Qq = P29.pca_apply(Bt, mu, comps), P29.pca_apply(Br, mu, comps)
    Rt = np.stack([P29.residualise(Xt[i], Pp) for i in range(len(T))])
    Rr = np.stack([P29.residualise(Xr[i], Qq) for i in range(len(T))])
    out["execution_check_rest_k10"] = spec_check(Rt, Rr)
    print("check baseline:", out["execution_check_baseline"], flush=True)
    print("check rest k10:", out["execution_check_rest_k10"], flush=True)

    print("\nPRIMARY", flush=True)
    out["R_ORIG"] = analyse(Xt, Xr, At, Ar, "R-ORIG")
    out["R_REST"] = analyse(Rt, Rr, At, Ar, "R-REST")

    print("\nDIAGNOSTICS", flush=True)
    out["diag_R_ORIG"] = diagnostics(Xt, Xr, At, Ar, "R-ORIG")
    out["diag_R_REST"] = diagnostics(Rt, Rr, At, Ar, "R-REST")

    print("\nNEGATIVE CONTROLS", flush=True)
    rng = np.random.default_rng(SEED)

    def rebuild(X):
        Pm, Tm, G = X.mean(0), X.mean(1), X.mean((0, 1))
        add = Pm[None] + Tm[:, None] - G[None, None]
        return add + rng.standard_normal(X.shape) * float((X - add).std())
    out["NC4_additive_rebuild"] = analyse(rebuild(Xt), rebuild(Xr), At, Ar,
                                          "NC4 additive rebuild", full=False,
                                          b_shuf=200)
    out["NC7_session_swap_R_ORIG"] = analyse(Xr, Xt, Ar, At, "NC7 swap R-ORIG",
                                             full=False, b_shuf=200)
    out["NC7_session_swap_R_REST"] = analyse(Rr, Rt, Ar, At, "NC7 swap R-REST",
                                             full=False, b_shuf=200)
    out["NC7_testfit_R_ORIG"] = analyse(Xt, Xr, At, At, "NC7 test-anat both R-ORIG",
                                        full=False, b_shuf=200)
    out["NC7_testfit_R_REST"] = analyse(Rt, Rr, At, At, "NC7 test-anat both R-REST",
                                        full=False, b_shuf=200)

    print("\nNC6 - k x lambda ladder", flush=True)
    lad = {}
    for k in K_LADDER:
        for lam in LAM_LADDER:
            for nm, (A_, B_) in (("R_ORIG", (Xt, Xr)), ("R_REST", (Rt, Rr))):
                lad["%s_k%d_lam%g" % (nm, k, lam)] = analyse(
                    A_, B_, At, Ar, "NC6 %s k=%d lam=%g" % (nm, k, lam),
                    k=k, lam=lam, full=False, b_shuf=200)
    out["NC6_ladder"] = lad

    print("\nSECONDARY REPRESENTATIONS", flush=True)
    sec = {}
    for nm, (A_, B_) in (("R_ORIG", (Xt, Xr)), ("R_REST", (Rt, Rr))):
        sec["S_LOSOCEN_" + nm] = analyse(A_, B_, At, Ar, "S-LOSOCEN " + nm,
                                         loso=True, full=False, b_shuf=200)
        It, Ir = ix.two_way_residual(A_), ix.two_way_residual(B_)
        # S-POOLED: the REPAIRED Phase 2.9 pooled subspace removal
        Jt = an.pooled_subspace_removal(It, At, K_PRIMARY)
        Jr = an.pooled_subspace_removal(Ir, Ar, K_PRIMARY)
        sp = survival(ix.cell_simmats(Jt, Jr), "S-POOLED " + nm, full=False)
        sec["S_POOLED_" + nm] = dict(after=sp,
                                     variance_removed_test=var_removed(It, Jt))
        print("  [%-34s] rank1 %.4f  Idiff %+7.3f  PASS %s"
              % ("S-POOLED " + nm, sp["rank1"], sp["idiff"], sp["PASSES"]), flush=True)
        # S-RBF: nonlinear variant
        Kt = It - an.crossfit_predict_rbf(It, At, K_PRIMARY, LAM_PRIMARY)
        Kr = Ir - an.crossfit_predict_rbf(Ir, Ar, K_PRIMARY, LAM_PRIMARY)
        sr = survival(ix.cell_simmats(Kt, Kr), "S-RBF " + nm, full=False)
        sec["S_RBF_" + nm] = dict(after=sr, variance_removed_test=var_removed(It, Kt))
        print("  [%-34s] rank1 %.4f  Idiff %+7.3f  PASS %s"
              % ("S-RBF " + nm, sr["rank1"], sr["idiff"], sr["PASSES"]), flush=True)
    out["secondary"] = sec

    out["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(RES, "01_anatomy_interaction.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nWROTE %s/01_anatomy_interaction.json  (%.0fs)"
          % (RES, out["seconds"]), flush=True)


if __name__ == "__main__":
    main()