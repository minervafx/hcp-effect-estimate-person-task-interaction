#!/usr/bin/env python
"""Phase 2.9 analysis helpers - fMRI.

Every retrieval / specificity / permutation estimator is the Phase 2.5 estimator in
`src/phase2_5/contrast.py`, imported read-only and called on Phase 2.9 matrices, so
the fMRI numbers are produced by exactly the code that produced the ds004148
(Phase 2/2.5/2.6), Shin2017 (Phase 2.7) and OpenBMI (Phase 2.8) numbers.  Only three
things are new, because the modality is new:

  * the representation builder (parcellated task-contrast COPE vectors, not spectra),
  * the anatomical nuisance model (Part J).

[release note] Vendored from the project's source tree. Only path/import wiring was
changed and the unused twin estimators (historical Parts K/L/O) were removed; see
docs/PATCHES.md. The estimator functions used by the scripts are unchanged.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import contrast as K                                                 # noqa: E402
SEED = 20260823
N_PERM = 10000

# ------------------------------------------------------- frozen task vocabulary
# PART E: chosen before any subject retrieval quantity was computed. See
# notes/PHASE_2_9_PREREGISTRATION.md section 5 for the selection criteria.
PRIMARY_TRANSITION = "WM_2BK_minus_0BK"

TRANSITIONS = {
    # name                    task          B (state after)   A (state before)
    "WM_2BK_minus_0BK":      ("WM",         "2BK",            "0BK"),
    "LANG_STORY_minus_MATH": ("LANGUAGE",   "STORY",          "MATH"),
    "MOTOR_AVG_minus_CUE":   ("MOTOR",      "AVG",            "CUE"),
    "REL_REL_minus_MATCH":   ("RELATIONAL", "REL",            "MATCH"),
}
SECONDARY_TRANSITIONS = [t for t in TRANSITIONS if t != PRIMARY_TRANSITION]

# HCP level-2 map names that carry each state-vs-implicit-baseline COPE, used for the
# STATIC control and for the SS/SD/DS construction. Resolved at fetch time against the
# CIFTI XML map names; these are the documented contrast labels.
STATE_COPES = {
    "WM":         {"0BK": "0BK", "2BK": "2BK"},
    "LANGUAGE":   {"STORY": "STORY", "MATH": "MATH"},
    "MOTOR":      {"CUE": "CUE", "AVG": "AVG"},
    "RELATIONAL": {"REL": "REL", "MATCH": "MATCH"},
}
DIFFERENTIAL_COPES = {
    "WM_2BK_minus_0BK":      ("WM", "2BK-0BK"),
    "LANG_STORY_minus_MATH": ("LANGUAGE", "STORY-MATH"),
    "MOTOR_AVG_minus_CUE":   ("MOTOR", "AVG-CUE"),
    "REL_REL_minus_MATCH":   ("RELATIONAL", "REL-MATCH"),
}

SESSIONS = ("test", "retest")


# ------------------------------------------------------------- representations
def zscore_rows(X):
    """Per-subject standardisation of a feature vector (removes global scale/offset)."""
    X = np.asarray(X, np.float64)
    mu = X.mean(1, keepdims=True)
    sd = np.maximum(X.std(1, keepdims=True), 1e-12)
    return (X - mu) / sd


def parcellate(gray, labels, n_parcels):
    """Mean of `gray` (n_maps, n_grayordinates) within each parcel of `labels`.

    `labels` is an integer array over grayordinates, identical for every subject
    (a GROUP atlas on the common 32k_fs_LR mesh).  Using a group atlas rather than each
    subject's own FreeSurfer parcellation is deliberate and pre-registered: a
    subject-specific parcellation would inject individual anatomy directly into the
    functional feature, which is the fMRI form of the per-subject-electrode-coordinate
    leak that Phase 2 avoided by using a template montage.
    """
    gray = np.atleast_2d(np.asarray(gray, np.float64))
    out = np.zeros((gray.shape[0], n_parcels), np.float64)
    for p in range(n_parcels):
        m = labels == (p + 1)
        if m.any():
            out[:, p] = gray[:, m].mean(1)
        else:
            out[:, p] = np.nan
    return out


def delta(state_b, state_a):
    """Delta_i^{A->B} = F(B) - F(A). The Phase 2.5 object, in fMRI units."""
    return np.asarray(state_b, np.float64) - np.asarray(state_a, np.float64)


# ------------------------------------------------------------------ estimators
def transfer(P, Q, n_perm=N_PERM, seed=SEED):
    """Phase 2.5 retrieval estimator: standardise on P only, cosine, permutation null."""
    r = K.evaluate(P, Q, n_perm=n_perm, seed=seed, metrics=("cosine",))["cosine"]
    r.pop("ranks", None)
    _, same, diff = K.similarity_blocks(P, Q)
    r.update(same_person_sim=float(same.mean()), diff_person_sim=float(diff.mean()),
             auc=K.auc(same, diff), cohens_d=K.cohens_d(same, diff))
    return r


def _unit(Z):
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)


def _standardise_pair(P, Q):
    """Session-1-only standardisation, then row-normalisation. Never fits on Q."""
    mu, sd = P.mean(0), np.maximum(P.std(0), 1e-9)
    return _unit((P - mu) / sd), _unit((Q - mu) / sd)


def simmat(P, Q):
    """Cosine similarity matrix S[i, j] = sim(P_i, Q_j), P-only standardisation."""
    Pz, Qz = _standardise_pair(np.asarray(P, float), np.asarray(Q, float))
    return Pz @ Qz.T


def specificity(P_by_trans, Q_by_trans, n_perm=N_PERM, seed=SEED):
    """PART I. Phase 2.5 section-6 estimator, verbatim logic, on fMRI transitions.

      SS  sim(D_i^{T,test}, D_i^{T,retest})   same person, same transition
      SD  sim(D_i^{T,test}, D_i^{U,retest})   same person, DIFFERENT transition
      DS  sim(D_i^{T,test}, D_j^{T,retest})   different person, same transition
    """
    trans = list(P_by_trans)
    n = P_by_trans[trans[0]].shape[0]
    allP = np.concatenate([P_by_trans[t] for t in trans], axis=0)
    mu, sd = allP.mean(0), np.maximum(allP.std(0), 1e-9)
    Pz = {t: _unit((P_by_trans[t] - mu) / sd) for t in trans}
    Qz = {t: _unit((Q_by_trans[t] - mu) / sd) for t in trans}
    off = ~np.eye(n, dtype=bool)
    per, SSa, SDa, DSa = {}, [], [], []
    for t in trans:
        S = Pz[t] @ Qz[t].T
        ss = np.diag(S).copy()
        ds = S[off].reshape(n, n - 1).mean(axis=1)
        sdv = [np.einsum("ij,ij->i", Pz[t], Qz[u]) for u in trans if u != t]
        sdm = np.stack(sdv).mean(axis=0) if sdv else np.full(n, np.nan)
        SSa.append(ss); SDa.append(sdm); DSa.append(ds)
        per[t] = dict(ss=float(ss.mean()), sd=float(np.nanmean(sdm)),
                      ds=float(ds.mean()),
                      d_ss_sd=K.cohens_d(ss, sdm), d_ss_ds=K.cohens_d(ss, ds),
                      auc_ss_sd=K.auc(ss, sdm), auc_ss_ds=K.auc(ss, S[off]),
                      test_ss_vs_sd=K.paired_permutation(ss, sdm, n_perm, seed),
                      test_ss_vs_ds=K.paired_permutation(ss, ds, n_perm, seed))
    if len(trans) > 1:
        q1 = K.bh_fdr([per[t]["test_ss_vs_sd"]["p_one_sided"] for t in trans])
        q2 = K.bh_fdr([per[t]["test_ss_vs_ds"]["p_one_sided"] for t in trans])
        for t, a, b in zip(trans, q1, q2):
            per[t]["test_ss_vs_sd"]["p_bh_fdr"] = float(a)
            per[t]["test_ss_vs_ds"]["p_bh_fdr"] = float(b)
    SS, SD, DS = np.concatenate(SSa), np.concatenate(SDa), np.concatenate(DSa)
    pooled = dict(ss=float(SS.mean()), sd=float(np.nanmean(SD)), ds=float(DS.mean()),
                  d_ss_sd=K.cohens_d(SS, SD), d_ss_ds=K.cohens_d(SS, DS),
                  auc_ss_sd=K.auc(SS, SD), auc_ss_ds=K.auc(SS, DS),
                  test_ss_vs_sd=K.paired_permutation(SS, SD, n_perm, seed),
                  test_ss_vs_ds=K.paired_permutation(SS, DS, n_perm, seed))
    return dict(transitions=trans, per_transition=per, pooled=pooled, n_subjects=n)


# ------------------------------------------------------------ anatomy (PART J)
def cv_r2(Xp, Y, n_folds=6, seed=SEED, lam=1.0):
    """Cross-validated R^2 of ridge(Xp -> Y). Phase 2.8 estimator verbatim."""
    rng = np.random.default_rng(seed)
    n = Xp.shape[0]
    order = rng.permutation(n)
    folds = np.array_split(order, n_folds)
    sse, sst = 0.0, 0.0
    for f in folds:
        tr = np.setdiff1d(order, f)
        mx, sx = Xp[tr].mean(0), np.maximum(Xp[tr].std(0), 1e-9)
        my = Y[tr].mean(0)
        Am = (Xp[tr] - mx) / sx
        Bm = Y[tr] - my
        W = np.linalg.solve(Am.T @ Am + lam * np.eye(Am.shape[1]), Am.T @ Bm)
        pred = ((Xp[f] - mx) / sx) @ W + my
        sse += float(((Y[f] - pred) ** 2).sum())
        sst += float(((Y[f] - my) ** 2).sum())
    return 1.0 - sse / max(sst, 1e-20)


def residualise(Y, Xp):
    """Regress Xp out of Y (ridge-stabilised OLS). Phase 2.8 estimator verbatim."""
    Am = np.concatenate([np.ones((Xp.shape[0], 1)),
                         (Xp - Xp.mean(0)) / np.maximum(Xp.std(0), 1e-9)], 1)
    W = np.linalg.solve(Am.T @ Am + 1e-6 * np.eye(Am.shape[1]), Am.T @ Y)
    return Y - Am @ W


def df_matched_null(P, Q, k, n_draws=200, seed=SEED, n_perm=200):
    """Retrieval after removing k RANDOM directions - the degrees-of-freedom control.

    Any nuisance removal must be read against this, never against the un-residualised
    number: removing k dimensions changes retrieval on its own.
    """
    rng = np.random.default_rng(seed)
    n = P.shape[0]
    out = []
    for _ in range(n_draws):
        Z = rng.standard_normal((n, k))
        out.append(transfer(residualise(P, Z), residualise(Q, Z),
                            n_perm=n_perm)["rank1"])
    a = np.asarray(out)
    return dict(k=int(k), n_draws=int(n_draws), mean=float(a.mean()),
                p5=float(np.quantile(a, .05)), p95=float(np.quantile(a, .95)))


def pca_fit_test_only(X, k, seed=SEED):
    """PCA basis fitted on the TEST session only. Returns (mean, components)."""
    mu = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    k = min(k, Vt.shape[0])
    return mu, Vt[:k]


def pca_apply(X, mu, comps):
    return (np.asarray(X, float) - mu) @ comps.T


# [release] Parts K/L/O (monozygotic-twin estimators) removed: unused by this
# study, which accessed no HCP Restricted Data and performed no twin analysis.


def summarise(r):
    return (f"rank1={r['rank1']:.4f} rank5={r['rank5']:.4f} mrr={r['mrr']:.4f} "
            f"medR={r['median_rank']:5.1f} AUC={r['auc']:.3f} p={r['p_rank1']:.1e}")
