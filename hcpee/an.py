"""Anatomical nuisance control IN INTERACTION SPACE (Run 12 / Phase 3.2).

Object
------
I[t, i, :]   the two-way (person x transition) interaction residual of Phase 3.1,
             computed independently within each session.
a[i, :]      a stable, subject-level anatomical feature vector (FreeSurfer
             morphometry), available for BOTH sessions.

Nuisance model under test
-------------------------
    A[i, t] = W_t @ g(a_i)

i.e. stable person-specific anatomy acting through a TRANSITION-SPECIFIC linear
map, which is exactly the mundane account that survives removal of the additive
person main effect and the additive transition main effect.  W_t is fitted
SEPARATELY FOR EACH TRANSITION, because the whole point of the alternative is
that the same anatomy expresses itself differently on contrasts with different
spatial patterns.

Why this is not the Phase 2.9 estimator
---------------------------------------
Phase 2.9 Part J ran `p29.pca_fit_test_only` on the RAW, UNSTANDARDISED 1,366
anatomical features, whose per-column SDs span six orders of magnitude
(cortical thickness ~0.1 mm, EstimatedTotalIntraCranialVol ~2e5 mm^3).  The
retained components are therefore a basis for head size and a few large volumes:
PC1 correlates 1.000 with the largest-SD column, and PCs 1-5 place 100 % of their
weight mass inside the 229-dimensional volumetric sub-block, leaving the
1,137-dimensional surface block (thickness / myelin / sulcal depth per parcel)
with no representation at all.  That block is NOT inherited.  Here every feature
is standardised inside the training fold before the PCA is fitted, and
zero-variance columns are dropped.

Leakage discipline
------------------
Scaler, PCA and ridge are all fitted inside a leave-one-subject-out training
fold.  The held-out subject contributes nothing to the nuisance mapping that is
subtracted from it.  Predictions are therefore out-of-fold everywhere.
"""
import numpy as np


# --------------------------------------------------------------- anatomy block
def clean_anatomy(A_list):
    """Drop columns with no between-subject variance in ANY session.

    Returns the list of cleaned arrays and the kept-column index.
    """
    A_list = [np.asarray(A, float) for A in A_list]
    keep = np.ones(A_list[0].shape[1], bool)
    for A in A_list:
        keep &= A.std(0) > 1e-12
    return [A[:, keep] for A in A_list], keep


# ------------------------------------------------------- fold-internal nuisance
def _fit_basis(Atr, k):
    """Standardise then PCA, both fitted on the TRAINING rows only."""
    mu, sd = Atr.mean(0), np.maximum(Atr.std(0), 1e-9)
    Z = (Atr - mu) / sd
    Zc = Z - Z.mean(0)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    comps = Vt[:min(k, Vt.shape[0])]
    return dict(mu=mu, sd=sd, gmu=Z.mean(0), comps=comps)


def _apply_basis(B, A):
    return ((np.asarray(A, float) - B["mu"]) / B["sd"] - B["gmu"]) @ B["comps"].T


def _ridge(Gtr, Ytr, lam):
    """Ridge on standardised PC scores.  Returns a callable on new scores."""
    gm, gs = Gtr.mean(0), np.maximum(Gtr.std(0), 1e-9)
    Zt = (Gtr - gm) / gs
    ym = Ytr.mean(0)
    W = np.linalg.solve(Zt.T @ Zt + lam * np.eye(Zt.shape[1]), Zt.T @ (Ytr - ym))
    return lambda G: ((np.asarray(G, float) - gm) / gs) @ W + ym


def crossfit_predict(I, A, k, lam, folds=None):
    """Out-of-fold anatomical prediction of the interaction residual.

    I : (T, n, p) interaction residual for ONE session
    A : (n, q)    anatomical features for the SAME session
    Returns Ihat with the same shape as I.

    W_t is fitted per transition.  The scaler and PCA are shared across
    transitions within a fold (they depend on anatomy only) but are refitted for
    every fold.
    """
    I = np.asarray(I, float)
    A = np.asarray(A, float)
    T, n, p = I.shape
    if folds is None:
        folds = [np.array([i]) for i in range(n)]          # leave-one-subject-out
    Ihat = np.empty_like(I)
    for te in folds:
        tr = np.setdiff1d(np.arange(n), te)
        B = _fit_basis(A[tr], k)
        Gtr, Gte = _apply_basis(B, A[tr]), _apply_basis(B, A[te])
        for t in range(T):
            f = _ridge(Gtr, I[t, tr], lam)
            Ihat[t, te] = f(Gte)
    return Ihat


def crossfit_r2(I, A, k, lam, folds=None):
    """Out-of-fold R^2 of the anatomical prediction, per transition and pooled.

    R^2 = 1 - SSE / SST with SST taken about the TRAINING mean of each fold, so a
    model no better than the group mean scores 0 and a worse one scores < 0.
    """
    I = np.asarray(I, float)
    A = np.asarray(A, float)
    T, n, p = I.shape
    if folds is None:
        folds = [np.array([i]) for i in range(n)]
    sse = np.zeros(T)
    sst = np.zeros(T)
    for te in folds:
        tr = np.setdiff1d(np.arange(n), te)
        B = _fit_basis(A[tr], k)
        Gtr, Gte = _apply_basis(B, A[tr]), _apply_basis(B, A[te])
        for t in range(T):
            f = _ridge(Gtr, I[t, tr], lam)
            pred = f(Gte)
            sse[t] += float(((I[t, te] - pred) ** 2).sum())
            sst[t] += float(((I[t, te] - I[t, tr].mean(0)) ** 2).sum())
    per = 1.0 - sse / np.maximum(sst, 1e-20)
    return dict(per_transition=[float(x) for x in per],
                pooled=float(1.0 - sse.sum() / max(sst.sum(), 1e-20)))


def remove(I, A, k, lam, folds=None):
    return np.asarray(I, float) - crossfit_predict(I, A, k, lam, folds)


# ------------------------------------------------- pooled (Phase 2.9 style) removal
def pooled_subspace_removal(I, A, k):
    """The REPAIRED Phase 2.9 estimator: standardise -> PCA on all n subjects ->
    project the resulting scores out of I.  No cross-fitting; reported as a
    secondary, because the held-out subject contributes to the subspace."""
    I = np.asarray(I, float)
    A = np.asarray(A, float)
    Z = (A - A.mean(0)) / np.maximum(A.std(0), 1e-9)
    Zc = Z - Z.mean(0)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    G = Zc @ Vt[:min(k, Vt.shape[0])].T
    D = np.concatenate([np.ones((G.shape[0], 1)), (G - G.mean(0))
                        / np.maximum(G.std(0), 1e-9)], 1)
    out = np.empty_like(I)
    for t in range(I.shape[0]):
        W = np.linalg.solve(D.T @ D + 1e-6 * np.eye(D.shape[1]), D.T @ I[t])
        out[t] = I[t] - D @ W
    return out


# ------------------------------------------------------------- diagnostics
def anat_simmat(At, Ar):
    """Cross-session cosine similarity of standardised anatomy (test x retest)."""
    At, Ar = np.asarray(At, float), np.asarray(Ar, float)
    mu, sd = At.mean(0), np.maximum(At.std(0), 1e-9)
    P = (At - mu) / sd
    Q = (Ar - mu) / sd
    P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
    return P @ Q.T


def anat_within_session_simmat(A):
    A = np.asarray(A, float)
    Z = (A - A.mean(0)) / np.maximum(A.std(0), 1e-9)
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    return Z @ Z.T


def mantel_offdiag(S1, S2):
    """Pearson r between two n x n similarity matrices, OFF-DIAGONAL ONLY.

    The diagonal is excluded on purpose.  Including it would let the near-perfect
    anatomical self-similarity act as an indicator of i == j, so the association
    would measure 'anatomy identifies people', not 'anatomy explains the
    interaction'.  That is precisely the confusion SIM4 exists to catch.
    """
    S1, S2 = np.asarray(S1, float), np.asarray(S2, float)
    off = ~np.eye(S1.shape[0], dtype=bool)
    return float(np.corrcoef(S1[off], S2[off])[0, 1])


def impostor_rank(S_I, S_anat):
    """Among the 41 NON-SELF retest candidates for each cell, where does the
    anatomically nearest other subject sit in the interaction-similarity ranking?

    Model-free: if I[i,t] were a function of anatomy_i and t, the anatomically
    nearest impostor would be pulled toward the top.  Returns the mean normalised
    rank (0 = top, 1 = bottom; 0.5 under the null) and the mean Spearman
    correlation, over cells, between anatomical and interaction similarity across
    the non-self candidates.
    """
    S_I = np.asarray(S_I, float)
    S_anat = np.asarray(S_anat, float)
    T, n, _ = S_I.shape
    nr, rho = [], []
    for t in range(T):
        for j in range(n):                      # retest subject j, its 41 impostors
            m = np.ones(n, bool)
            m[j] = False
            si = S_I[t][m, j]
            sa = S_anat[m, j]
            nn = int(np.argmax(sa))             # anatomically nearest impostor
            nr.append(float((si > si[nn]).sum()) / (si.size - 1.0))
            rho.append(_spearman(sa, si))
    return dict(mean_normalised_rank=float(np.mean(nr)),
                mean_spearman=float(np.mean(rho)), n_cells=len(nr))


def _spearman(a, b):
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def _rankdata(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(x.size, float)
    xs = x[o]
    i = 0
    while i < xs.size:
        j = i
        while j + 1 < xs.size and xs[j + 1] == xs[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


# ===========================================================================
# EXACT FAST PATH for the N_ANATSHUF null.
#
# N_ANATSHUF permutes the rows of the anatomy block by tau, applied identically
# to both sessions, and re-runs the whole leave-one-subject-out removal.  Done
# naively that refits 42 PCA bases per draw.  It does not have to.
#
# Under tau, fold i pairs functional row i with anatomy row j = tau(i), and its
# TRAINING anatomy rows are A[tau(tr_i)] = every anatomy row except A[j].  So the
# fold's scaler and PCA depend on the permutation ONLY through j, and the 42
# possible bases - "leave anatomy row j out" - can be precomputed once and reused
# for every draw.  Only the ridge targets change.
#
# `assert_shuffle_fast_path_exact` verifies this numerically rather than trusting
# the argument.
# ===========================================================================
def precompute_bases(A, k):
    """For each j, the scaler+PCA fitted on A with row j held out, plus the
    projection of ALL rows onto that basis."""
    A = np.asarray(A, float)
    n = A.shape[0]
    out = []
    for j in range(n):
        tr = np.setdiff1d(np.arange(n), [j])
        B = _fit_basis(A[tr], k)
        out.append(_apply_basis(B, A))                 # (n, k) scores on basis j
    return out


def remove_perm(I, G_by_j, tau, lam):
    """Anatomy-removed interaction residual under anatomy permutation `tau`.

    tau[i] = the anatomy row assigned to functional subject i.  tau = identity
    reproduces the unpermuted primary removal exactly.
    """
    I = np.asarray(I, float)
    T, n, p = I.shape
    out = np.empty_like(I)
    idx = np.arange(n)
    for i in range(n):
        j = int(tau[i])
        G = G_by_j[j]                                   # scores of every row on basis j
        tr = idx[idx != i]
        Gtr = G[tau[tr]]
        gte = G[j][None, :]
        for t in range(T):
            f = _ridge(Gtr, I[t, tr], lam)
            out[t, i] = I[t, i] - f(gte)[0]
    return out


def assert_shuffle_fast_path_exact(seed=0, n=14, q=60, T=3, p=25, k=4, lam=1.0,
                                   n_check=6, tol=1e-8):
    """Verify remove_perm(tau) against the brute-force `remove` on permuted anatomy."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, q)) * rng.uniform(0.5, 4.0, q)
    I = rng.standard_normal((T, n, p))
    G_by_j = precompute_bases(A, k)
    bad = []
    # identity must reproduce the primary removal
    d0 = np.abs(remove_perm(I, G_by_j, np.arange(n), lam) - remove(I, A, k, lam)).max()
    if d0 > tol:
        bad.append(f"identity: maxdiff {d0}")
    for _ in range(n_check):
        tau = rng.permutation(n)
        fast = remove_perm(I, G_by_j, tau, lam)
        slow = remove(I, A[tau], k, lam)
        d = np.abs(fast - slow).max()
        if d > tol:
            bad.append(f"tau: maxdiff {d}")
    return dict(ok=not bad, problems=bad, n_check=int(n_check), tol=tol,
                identity_maxdiff=float(d0))


# ---------------------------------------------------------------------------
# Linear-smoother form of remove_perm.
#
# For a fixed basis j, the standardiser of the PC scores, the Gram matrix and the
# ridge inverse all depend on the SET of training anatomy rows, which is always
# "every row except j", and therefore not on the permutation.  Only the pairing
# of anatomy rows to functional rows changes.  The ridge prediction is a linear
# smoother of the training targets:
#
#     pred[t,i] = sum_{r != i} u_j[tau(r)] * (I[t,r] - ybar_t) + ybar_t
#
# with u_j = Z_j @ (Z_j[j] @ M_j)^T, M_j = (Z_j[-j]^T Z_j[-j] + lam I)^{-1}.
# Precomputing u_j collapses each fold from a (k x n_tr) x (n_tr x p) product to
# a single length-n_tr weighted sum per transition.
# ---------------------------------------------------------------------------
def precompute_smoothers(A, k, lam):
    """Returns U with U[j] the length-n weight vector of basis j (see above)."""
    A = np.asarray(A, float)
    n = A.shape[0]
    U = np.empty((n, n))
    for j in range(n):
        tr = np.setdiff1d(np.arange(n), [j])
        B = _fit_basis(A[tr], k)
        G = _apply_basis(B, A)
        gm, gs = G[tr].mean(0), np.maximum(G[tr].std(0), 1e-9)
        Z = (G - gm) / gs
        M = np.linalg.inv(Z[tr].T @ Z[tr] + lam * np.eye(Z.shape[1]))
        U[j] = Z @ (Z[j] @ M)
    return U


def remove_perm_fast(I, U, tau):
    """Identical to remove_perm(I, precompute_bases(A, k), tau, lam) with U from
    precompute_smoothers(A, k, lam)."""
    I = np.asarray(I, float)
    T, n, p = I.shape
    out = np.empty_like(I)
    tau = np.asarray(tau, int)
    w_all = U[tau]                                  # w_all[i, r] = U[tau[i], tau[r]]
    w_all = w_all[:, tau]
    S = I.sum(1)                                    # (T, p)
    for i in range(n):
        w = w_all[i].copy()
        w[i] = 0.0
        ybar = (S - I[:, i]) / (n - 1.0)            # (T, p) training mean
        sw = w.sum()
        pred = np.einsum("r,trp->tp", w, I) - sw * ybar + ybar
        out[:, i] = I[:, i] - pred
    return out


def assert_smoother_fast_path_exact(seed=3, n=13, q=50, T=3, p=17, k=4, lam=1.0,
                                    n_check=6, tol=1e-8):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, q)) * rng.uniform(0.5, 4.0, q)
    I = rng.standard_normal((T, n, p))
    U = precompute_smoothers(A, k, lam)
    G_by_j = precompute_bases(A, k)
    bad = []
    d0 = np.abs(remove_perm_fast(I, U, np.arange(n)) - remove(I, A, k, lam)).max()
    if d0 > tol:
        bad.append(f"identity vs remove: {d0}")
    for _ in range(n_check):
        tau = rng.permutation(n)
        d = np.abs(remove_perm_fast(I, U, tau)
                   - remove_perm(I, G_by_j, tau, lam)).max()
        if d > tol:
            bad.append(f"tau vs remove_perm: {d}")
        d2 = np.abs(remove_perm_fast(I, U, tau) - remove(I, A[tau], k, lam)).max()
        if d2 > tol:
            bad.append(f"tau vs brute remove: {d2}")
    return dict(ok=not bad, problems=bad, n_check=int(n_check), tol=tol,
                identity_maxdiff=float(d0))


# ------------------------------------------------------- nonlinear variant (S-RBF)
def crossfit_predict_rbf(I, A, k, lam, gamma_scale=1.0):
    """Leave-one-subject-out kernel ridge with an RBF kernel on the anatomy PC
    scores.  Carried as a robustness variant because the primary estimator is
    linear in anatomy, and a real geometric effect need not be.

    The bandwidth is the median pairwise squared distance of the TRAINING scores
    (median heuristic), so it too is fitted inside the fold.
    """
    I = np.asarray(I, float)
    A = np.asarray(A, float)
    T, n, p = I.shape
    out = np.empty_like(I)
    for i in range(n):
        tr = np.setdiff1d(np.arange(n), [i])
        B = _fit_basis(A[tr], k)
        Gtr, Gte = _apply_basis(B, A[tr]), _apply_basis(B, A[[i]])
        gm, gs = Gtr.mean(0), np.maximum(Gtr.std(0), 1e-9)
        Ztr, Zte = (Gtr - gm) / gs, (Gte - gm) / gs
        d2 = ((Ztr[:, None] - Ztr[None]) ** 2).sum(-1)
        med = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
        g = gamma_scale / max(med, 1e-9)
        K = np.exp(-g * d2)
        kte = np.exp(-g * ((Zte[:, None] - Ztr[None]) ** 2).sum(-1))
        Kinv = np.linalg.inv(K + lam * np.eye(K.shape[0]))
        for t in range(T):
            ym = I[t, tr].mean(0)
            out[t, i] = (kte @ (Kinv @ (I[t, tr] - ym)))[0] + ym
    return out
