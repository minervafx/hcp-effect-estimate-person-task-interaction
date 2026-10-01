"""Person x transition INTERACTION estimators (Run 11 / Phase 3.1).

Design frozen before any real-data interaction outcome was inspected.

Object:  X[s][t] is an (n_subjects, n_features) array of transition-contrast vectors
         for session s in {test, retest} and transition t in 1..T.

Two-way decomposition, computed INDEPENDENTLY WITHIN EACH SESSION:

    I[i,t] = X[i,t] - Pmean[i] - Tmean[t] + Gmean

    Pmean[i] = mean_t X[i,t]      Tmean[t] = mean_i X[i,t]      Gmean = mean_{i,t}

This annihilates ANY additive model X[i,t] = P[i] + T[t] EXACTLY, so a purely
additive representation leaves I = 0 up to noise, and noise does not correspond
across sessions.  No cross-fitting is required for that annihilation to be exact;
a leave-one-transition-out person mean would in fact REINTRODUCE a transition main
effect (see prereg 5.3).  A leave-one-SUBJECT-out transition mean is a genuine
cross-fit and is carried as robustness variant R1.
"""
import numpy as np


# ------------------------------------------------------------- decomposition
def two_way_residual(Xt, loso_transition_mean=False):
    """Xt: (T, n, p) stack for ONE session.  Returns I with the same shape."""
    X = np.asarray(Xt, np.float64)
    T, n, p = X.shape
    Pm = X.mean(0)                                   # (n, p)  person mean over T
    G = X.mean((0, 1))                               # (p,)
    if not loso_transition_mean:
        Tm = X.mean(1)                               # (T, p)
        return X - Pm[None] - Tm[:, None] + G[None, None]
    # R1: leave-one-subject-out column (transition) mean and grand mean.
    Tm_loso = (X.sum(1)[:, None, :] - X) / (n - 1.0)          # (T, n, p)
    G_loso = (X.sum((0, 1))[None, :] - X.sum(0)) / ((n - 1.0) * T)   # (n, p)
    return X - Pm[None] - Tm_loso + G_loso[None]


def unit(Z, axis=-1):
    return Z / np.maximum(np.linalg.norm(Z, axis=axis, keepdims=True), 1e-12)


# ------------------------------------------------------------- primary stat
def cell_simmats(It, Ir):
    """Per-transition cross-session cosine similarity matrices in interaction space.

    Returns S with S[t][i, j] = cos(I[i,t,test], I[j,t,retest]).
    No refitting of any mean: I is already centred.  Cosine only.
    """
    A, B = unit(np.asarray(It, float)), unit(np.asarray(Ir, float))
    return np.einsum("tip,tjp->tij", A, B)


def primary_stats(S):
    """S: (T, n, n).  Pooled over T*n cells.

    rank1  : fraction of cells whose correct subject is top-ranked WITHIN its
             transition (chance = 1/n).  This is the headline effect size.
    auc    : AUC(same cell, different-person-same-transition).
    idiff  : mean(self) - mean(different person, same transition), x100.
    """
    S = np.asarray(S, float)
    T, n, _ = S.shape
    eye = np.eye(n, dtype=bool)
    self_s, ds_s, r1, ranks = [], [], [], []
    for t in range(T):
        M = S[t]
        d = np.diag(M)
        self_s.append(d)
        ds_s.append(M[~eye])
        # column j of M holds similarities of every test prototype to retest j
        rk = 1 + (M > d[None, :]).sum(0)             # rank of the true prototype
        ranks.append(rk)
        r1.append(rk == 1)
    self_s = np.concatenate(self_s)
    ds_s = np.concatenate(ds_s)
    ranks = np.concatenate(ranks)
    r1 = np.concatenate(r1)
    return dict(rank1=float(r1.mean()), chance=1.0 / n,
                mrr=float((1.0 / ranks).mean()), median_rank=float(np.median(ranks)),
                self_sim=float(self_s.mean()), ds_sim=float(ds_s.mean()),
                idiff=float((self_s.mean() - ds_s.mean()) * 100.0),
                auc=float(_auc(self_s, ds_s)),
                per_transition_rank1=[float(x.mean()) for x in np.split(r1, T)])


def _auc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    x = np.concatenate([a, b])
    r = np.empty(x.size, float)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    i = 0
    while i < xs.size:                                # average ranks over ties
        j = i
        while j + 1 < xs.size and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return (r[:a.size].sum() - a.size * (a.size + 1) / 2.0) / (a.size * b.size)


# ------------------------------------------------------- structure-preserving null
def null_person(Xt_test, Xt_retest, n_perm, seed, loso=False, stat="rank1"):
    """N_person: relabel SUBJECTS in the retest session with ONE permutation sigma
    applied UNIFORMLY ACROSS ALL TRANSITIONS, then re-run the whole decomposition.

    Because sigma is the same for every transition, the retest person main effects
    and transition main effects are both preserved EXACTLY as sets - only their
    correspondence to the test session's persons is destroyed.  The person x
    transition pairing within retest is also preserved; what is destroyed is
    person identity across sessions.  This is the null for "the interaction
    residual carries a stable PERSON-SPECIFIC component".
    """
    rng = np.random.default_rng(seed)
    Xr = np.asarray(Xt_retest, float)
    It = two_way_residual(Xt_test, loso)
    out = []
    for _ in range(n_perm):
        pi = rng.permutation(Xr.shape[1])
        Ir = two_way_residual(Xr[:, pi], loso)
        out.append(primary_stats(cell_simmats(It, Ir))[stat])
    return np.asarray(out, float)


def null_transition(Xt_test, Xt_retest, loso=False, stat="rank1", derangements_only=True):
    """N_transition: relabel TRANSITIONS in the retest session with ONE permutation pi
    applied UNIFORMLY ACROSS ALL SUBJECTS.  Person and transition main effects are
    preserved exactly (transitions merely relabelled); the person x transition
    pairing across sessions is destroyed.  Exhaustively enumerated (T! is small)."""
    from itertools import permutations
    Xr = np.asarray(Xt_retest, float)
    T = Xr.shape[0]
    It = two_way_residual(Xt_test, loso)
    out = []
    for pi in permutations(range(T)):
        if derangements_only and any(pi[i] == i for i in range(T)):
            continue
        if not derangements_only and pi == tuple(range(T)):
            continue
        Ir = two_way_residual(Xr[list(pi)], loso)
        out.append(primary_stats(cell_simmats(It, Ir))[stat])
    return np.asarray(out, float)


def perm_p(obs, null):
    null = np.asarray(null, float)
    return float((1 + (null >= obs - 1e-12).sum()) / (1 + null.size))


def boot_ci(S, n_boot=10000, seed=0, alpha=0.05):
    """Percentile bootstrap over SUBJECTS (the unit of exchangeability) for the
    primary rank-1 and for Idiff.  Subjects are resampled with replacement; the
    similarity matrix is re-indexed on both axes so a resampled subject keeps its
    own prototype AND its own distractor structure."""
    S = np.asarray(S, float)
    T, n, _ = S.shape
    rng = np.random.default_rng(seed)
    r1, idf = [], []
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        Sb = S[:, b][:, :, b]
        st = primary_stats(Sb)
        r1.append(st["rank1"]); idf.append(st["idiff"])
    q = lambda a: (float(np.quantile(a, alpha / 2)), float(np.quantile(a, 1 - alpha / 2)))
    return dict(rank1_ci=q(r1), idiff_ci=q(idf), n_boot=int(n_boot))


# ----------------------------------------------- CONFIRMATORY factorial model
def factorial_beta(Xt_test, Xt_retest):
    """Similarity-level 2x2 factorial on the RAW (uncentred) representation.

        S[(i,t),(j,u)] = b0 + bP*1{i=j} + bT*1{t=u} + bPT*1{i=j & t=u}

    Saturated, so the OLS solution is exactly the cell-mean contrast

        bPT = SS - SD - DS + DD

    which needs no iid assumption at all.  Only the NULL needs to be
    dependence-aware, and it is (the same structure-preserving permutations).
    Standardisation is the Phase 2.9 one: a single mean/sd fitted on the TEST
    session pooled over transitions, then row-normalisation, then cosine.
    """
    P = np.asarray(Xt_test, float); Q = np.asarray(Xt_retest, float)
    T, n, _ = P.shape
    flat = P.reshape(T * n, -1)
    mu, sd = flat.mean(0), np.maximum(flat.std(0), 1e-9)
    A = unit((P - mu) / sd); B = unit((Q - mu) / sd)
    S = np.einsum("tip,ujp->tiuj", A, B)                       # (T,n,T,n)
    ti = np.arange(T)[:, None, None, None] == np.arange(T)[None, None, :, None]
    ii = np.arange(n)[None, :, None, None] == np.arange(n)[None, None, None, :]
    same_t = np.broadcast_to(ti, S.shape); same_i = np.broadcast_to(ii, S.shape)
    SS = S[same_t & same_i].mean(); SD = S[~same_t & same_i].mean()
    DS = S[same_t & ~same_i].mean(); DD = S[~same_t & ~same_i].mean()
    return dict(SS=float(SS), SD=float(SD), DS=float(DS), DD=float(DD),
                b0=float(DD), bP=float(SD - DD), bT=float(DS - DD),
                bPT=float(SS - SD - DS + DD))


def factorial_null(Xt_test, Xt_retest, n_perm, seed):
    rng = np.random.default_rng(seed)
    Q = np.asarray(Xt_retest, float)
    out = []
    for _ in range(n_perm):
        pi = rng.permutation(Q.shape[1])
        out.append(factorial_beta(Xt_test, Q[:, pi])["bPT"])
    return np.asarray(out, float)


# ===========================================================================
# EXACT FAST PATH for the N_PERSON null.
#
# The two-way residual operator is EQUIVARIANT under a subject permutation that
# is applied uniformly across transitions:
#
#     I( X[:, pi] )[t, i]  ==  I(X)[t, pi[i]]      exactly
#
# because Pmean permutes with the subjects while Tmean and G are unchanged
# (a permutation does not change a mean over subjects).  The same holds for the
# leave-one-subject-out variant.  Therefore the whole null can be evaluated by
# re-indexing ONE precomputed similarity array instead of re-running the
# decomposition 10,000 times.  `assert_fast_path_exact` verifies this
# numerically rather than trusting the argument.
# ===========================================================================
def _rank1_from_S_perm(S, pi):
    """rank-1 of the permuted design, from the unpermuted similarity array."""
    T, n, _ = S.shape
    hit = 0
    for t in range(T):
        M = S[t]
        col = M[:, pi]                       # retest label j now holds subject pi[j]
        true = col[np.arange(n), np.arange(n)]
        hit += int(((col > true[None, :]).sum(0) == 0).sum())
    return hit / float(T * n)


def null_person_fast(Xt_test, Xt_retest, n_perm, seed, loso=False):
    It = two_way_residual(Xt_test, loso)
    Ir = two_way_residual(Xt_retest, loso)
    S = cell_simmats(It, Ir)
    rng = np.random.default_rng(seed)
    n = S.shape[1]
    return np.asarray([_rank1_from_S_perm(S, rng.permutation(n))
                       for _ in range(n_perm)], float), S


def per_transition_null_fast(S, n_perm, seed):
    """Per-transition rank-1 null under the SAME uniform subject permutation."""
    T, n, _ = S.shape
    rng = np.random.default_rng(seed)
    out = np.empty((n_perm, T))
    for b in range(n_perm):
        pi = rng.permutation(n)
        for t in range(T):
            col = S[t][:, pi]
            true = col[np.arange(n), np.arange(n)]
            out[b, t] = ((col > true[None, :]).sum(0) == 0).mean()
    return out


def bh_fdr(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty_like(p)
    m = p.size
    prev = 1.0
    for r in range(m - 1, -1, -1):
        prev = min(prev, p[o[r]] * m / (r + 1))
        q[o[r]] = prev
    return q


def factorial_cells(Xt_test, Xt_retest):
    """Return (d, A) sufficient statistics for the factorial model under any uniform
    subject permutation of the retest session.

        d[t,u,i] = S[t, i, u, i-th retest subject]      (matched-subject slice)
        A[t,u]   = mean over ALL (i, j) of S[t,i,u,j]   (permutation-invariant)
    """
    P = np.asarray(Xt_test, float); Q = np.asarray(Xt_retest, float)
    T, n, _ = P.shape
    flat = P.reshape(T * n, -1)
    mu, sd = flat.mean(0), np.maximum(flat.std(0), 1e-9)
    A_ = unit((P - mu) / sd); B_ = unit((Q - mu) / sd)
    S4 = np.einsum("tip,ujp->tiuj", A_, B_)
    A = S4.mean((1, 3))
    return S4, A


def bPT_from_perm(S4, A, pi):
    T, n = S4.shape[0], S4.shape[1]
    idx = np.arange(n)
    d = S4[:, idx, :, :][:, :, :, pi][:, idx, :, idx]      # (n, T, T) matched slice
    d = d.transpose(1, 2, 0).mean(2)                       # (T, T)
    eyeT = np.eye(T, dtype=bool)
    SS = d[eyeT].mean(); SD = d[~eyeT].mean()
    off = (n * A - d) / (n - 1.0)
    DS = off[eyeT].mean(); DD = off[~eyeT].mean()
    return float(SS - SD - DS + DD), float(SS), float(SD), float(DS), float(DD)


def assert_fast_path_exact(seed=0, n=12, T=4, p=40, n_check=25, tol=1e-9):
    """Numerically verify the fast paths against the brute-force implementations."""
    rng = np.random.default_rng(seed)
    Xt = rng.standard_normal((T, n, p)) + rng.standard_normal((1, n, p))
    Xr = rng.standard_normal((T, n, p)) + rng.standard_normal((1, n, p))
    bad = []
    for loso in (False, True):
        a = null_person(Xt, Xr, n_check, 7, loso=loso)
        b, _ = null_person_fast(Xt, Xr, n_check, 7, loso=loso)
        if not np.allclose(a, b, atol=tol):
            bad.append(f"null_person loso={loso}: maxdiff {np.abs(a-b).max()}")
    S4, A = factorial_cells(Xt, Xr)
    r = rng2 = np.random.default_rng(11)
    for _ in range(n_check):
        pi = rng2.permutation(n)
        f = factorial_beta(Xt, Xr[:, pi])
        g = bPT_from_perm(S4, A, pi)
        for name, x, y in (("bPT", f["bPT"], g[0]), ("SS", f["SS"], g[1]),
                           ("SD", f["SD"], g[2]), ("DS", f["DS"], g[3]),
                           ("DD", f["DD"], g[4])):
            if abs(x - y) > 1e-9:
                bad.append(f"factorial {name}: {x} vs {y}")
    return dict(ok=not bad, problems=bad, n_check=int(n_check), n=int(n), T=int(T), p=int(p))


def idiff_from_S_perm(S, pi):
    """Idiff in interaction space under a uniform retest subject permutation, exactly.

    self' = mean over cells of S[t][j, pi[j]];  DS' = mean of everything else.
    The total sum of S is permutation-invariant, so DS' follows by subtraction.
    """
    T, n, _ = S.shape
    idx = np.arange(n)
    matched = S[:, idx, :][:, idx, pi] if False else np.stack(
        [S[t][idx, pi] for t in range(T)])
    self_m = matched.mean()
    tot = S.sum()
    ds_m = (tot - matched.sum()) / float(T * n * (n - 1))
    return (self_m - ds_m) * 100.0


def null_person_fast2(Xt_test, Xt_retest, n_perm, seed, loso=False):
    """N_PERSON null for BOTH primary statistics from one pass."""
    It = two_way_residual(Xt_test, loso)
    Ir = two_way_residual(Xt_retest, loso)
    S = cell_simmats(It, Ir)
    rng = np.random.default_rng(seed)
    n = S.shape[1]
    r1, idf = np.empty(n_perm), np.empty(n_perm)
    for b in range(n_perm):
        pi = rng.permutation(n)
        r1[b] = _rank1_from_S_perm(S, pi)
        idf[b] = idiff_from_S_perm(S, pi)
    return r1, idf, S
