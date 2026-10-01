#!/usr/bin/env python
"""PHASE 3.1 EFFECT-ESTIMATE FALSIFICATION - Person x transition interaction test on COPE maps.

Identical to src/phase3_1/01_interaction.py but loads from the effect-estimate cache
(results/hcp_effect_estimate_falsification/cache/) instead of the z-stat cache.

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
PREREG = "HCP_EFFECT_ESTIMATE_FALSIFICATION_PREREG.md"
T = list(P29.TRANSITIONS)
PTn = P29.PRIMARY_TRANSITION
SEED = P29.SEED
NPERM = 10000
NBOOT = 10000


def prereg_sha():
    return paths.prereg_sha(PREREG)


def stack(task, ses, subs, suffix="diff"):
    return np.stack([[task[(s, ses)][f"{t}__{suffix}"] for s in subs]
                     for t in T]).astype(float)


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


def removal_scores(Bt, Br, k, fit="test"):
    src = Bt if fit == "test" else Br
    mu, comps = P29.pca_fit_test_only(src, k)
    return P29.pca_apply(Bt, mu, comps), P29.pca_apply(Br, mu, comps)


def resid_stack(X, Xp):
    return np.stack([P29.residualise(X[i], Xp) for i in range(X.shape[0])])


def spec_check(Xt, Xr):
    p = P29.specificity({t: Xt[i] for i, t in enumerate(T)},
                        {t: Xr[i] for i, t in enumerate(T)}, n_perm=1000)["pooled"]
    return {k: round(float(p[k]), 4) for k in
            ("ss", "sd", "ds", "auc_ss_sd", "auc_ss_ds")}


def evaluate(Xt, Xr, tag, full=True, loso=False, n_perm=NPERM):
    t0 = time.time()
    null, nulli, S = ix.null_person_fast2(Xt, Xr, n_perm, SEED, loso=loso)
    st = ix.primary_stats(S)
    out = dict(tag=tag, loso_transition_mean=bool(loso), n_subjects=int(S.shape[1]), **st)
    out["null_person"] = dict(n_perm=int(n_perm), mean=float(null.mean()),
                              sd=float(null.std()), p5=float(np.quantile(null, .05)),
                              p95=float(np.quantile(null, .95)), max=float(null.max()))
    out["null_person_idiff"] = dict(n_perm=int(n_perm), mean=float(nulli.mean()),
                                    sd=float(nulli.std()),
                                    p95=float(np.quantile(nulli, .95)),
                                    p99=float(np.quantile(nulli, .99)),
                                    max=float(nulli.max()))
    out["p_idiff_vs_null_person"] = ix.perm_p(st["idiff"], nulli)
    out["p_rank1_vs_null_person"] = ix.perm_p(st["rank1"], null)
    if full:
        ptn = ix.per_transition_null_fast(S, n_perm, SEED)
        ps = [ix.perm_p(st["per_transition_rank1"][i], ptn[:, i]) for i in range(len(T))]
        qs = ix.bh_fdr(ps)
        out["per_transition"] = [
            dict(transition=T[i], rank1=st["per_transition_rank1"][i],
                 null_mean=float(ptn[:, i].mean()), p=float(ps[i]), q_bh_fdr=float(qs[i]))
            for i in range(len(T))]
        out.update(ix.boot_ci(S, n_boot=NBOOT, seed=SEED))
        nt = ix.null_transition(Xt, Xr, loso=loso)
        out["null_transition_derangements"] = dict(
            n=int(nt.size), values=[float(x) for x in nt], mean=float(nt.mean()),
            max=float(nt.max()), p=ix.perm_p(st["rank1"], nt),
            smallest_attainable_p=1.0 / (nt.size + 1))
        S4, A = ix.factorial_cells(Xt, Xr)
        n = S.shape[1]
        b, ss, sd, ds, dd = ix.bPT_from_perm(S4, A, np.arange(n))
        rng = np.random.default_rng(SEED)
        fn = np.asarray([ix.bPT_from_perm(S4, A, rng.permutation(n))[0]
                         for _ in range(n_perm)])
        out["confirmatory_factorial"] = dict(
            SS=ss, SD=sd, DS=ds, DD=dd, b0=dd, bP=sd - dd, bT=ds - dd, bPT=b,
            null_person_mean=float(fn.mean()), null_person_sd=float(fn.std()),
            null_person_p95=float(np.quantile(fn, .95)),
            p_bPT_vs_null_person=ix.perm_p(b, fn))
        It, Ir = ix.two_way_residual(Xt, loso), ix.two_way_residual(Xr, loso)
        wrong = []
        for a in range(len(T)):
            for c in range(len(T)):
                if a == c:
                    continue
                M = ix.unit(It[a]) @ ix.unit(Ir[c]).T
                d = np.diag(M)
                wrong.append(float(((M > d[None, :]).sum(0) == 0).mean()))
        out["NC3_wrong_transition_rank1"] = float(np.mean(wrong))
        out["NC3_wrong_transition_per_pair"] = [round(x, 5) for x in wrong]
    out["seconds"] = round(time.time() - t0, 1)
    print("  [%-28s] Idiff=%+7.3f p=%.2e | rank1=%.4f (chance %.4f) p=%.2e | "
          "AUC=%.4f | null r1=%.4f Id=%+.3f  %.0fs"
          % (tag, st["idiff"], out["p_idiff_vs_null_person"], st["rank1"], st["chance"],
             out["p_rank1_vs_null_person"], st["auc"], null.mean(), nulli.mean(),
             out["seconds"]), flush=True)
    return out


def main():
    t0 = time.time()
    os.makedirs(RES, exist_ok=True)
    subs = p30.subjects()
    task = load_task_effect_estimate(subs)
    ix_ok = ix.assert_fast_path_exact()

    out = dict(prereg_sha256=prereg_sha(),
               prereg_expected_sha256=paths.PREREG_EXPECTED_SHA256[PREREG],
               map_type_guard=GUARD, seed=SEED, n_perm=NPERM, n_boot=NBOOT,
               transitions=T, n_subjects=len(subs), chance_rank1=1.0 / len(subs),
               subjects=subs, fast_path_equivalence_check=ix_ok,
               NC7_restricted_data_accessed=False)
    print("n=%d  T=%d  chance=%.5f  fastpath_ok=%s"
          % (len(subs), len(T), 1 / len(subs), ix_ok["ok"]), flush=True)
    assert ix_ok["ok"], ix_ok

    Xt, Xr = stack(task, "test", subs), stack(task, "retest", subs)
    out["execution_check_baseline"] = spec_check(Xt, Xr)
    print("check baseline:", out["execution_check_baseline"], flush=True)

    # ---------------- R-ORIG (PRIMARY level 1)
    out["R_ORIG"] = evaluate(Xt, Xr, "R-ORIG")

    # ---------------- negative controls (same as Phase 3.1)
    out["NC4_swap_R_ORIG"] = evaluate(Xr, Xt, "NC4 swap R-ORIG", full=False)
    out["NC5_R1_loso_R_ORIG"] = evaluate(Xt, Xr, "NC5 R1-LOSO R-ORIG", full=False, loso=True)

    rng = np.random.default_rng(SEED)
    def rebuild(X):
        Pm, Tm, G = X.mean(0), X.mean(1), X.mean((0, 1))
        add = Pm[None] + Tm[:, None] - G[None, None]
        return add + rng.standard_normal(X.shape) * float((X - add).std())
    out["NC6_additive_rebuild_R_ORIG"] = evaluate(rebuild(Xt), rebuild(Xr),
                                                  "NC6 additive rebuild", full=False)

    out["seconds"] = round(time.time() - t0, 1)
    out_path = os.path.join(RES, "01_interaction_effect_estimate.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)
    print("WROTE %s  (%.0fs)" % (out_path, out["seconds"]), flush=True)


if __name__ == "__main__":
    main()