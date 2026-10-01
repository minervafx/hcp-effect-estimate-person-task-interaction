"""Phase 2D-2J analysis primitives.

Leakage policy, enforced by construction in every function here:
  * every scaler, nuisance-regression coefficient and PCA basis is fitted on the
    TRAINING rows only and then applied to the test rows;
  * segments from one recording are never split across train and test -- the train
    and test sets are always different (session, state) recordings;
  * the headline train/test pair is fixed before any result is inspected.
"""
import numpy as np

EPS = 1e-9
METRICS = ("cosine", "correlation", "seuclidean")


# ---------------------------------------------------------------- train-fitted
class Standardizer:
    def fit(self, X):
        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0)
        self.keep_ = self.sd_ > EPS
        return self

    def transform(self, X):
        return (X[:, self.keep_] - self.mu_[self.keep_]) / self.sd_[self.keep_]


class NuisanceRegressor:
    """Residualise features on a nuisance block. Coefficients from TRAIN rows only."""

    def fit(self, X, N):
        A = np.concatenate([np.ones((N.shape[0], 1)), N], axis=1)
        self.beta_, *_ = np.linalg.lstsq(A, X, rcond=None)
        return self

    def transform(self, X, N):
        A = np.concatenate([np.ones((N.shape[0], 1)), N], axis=1)
        return X - A @ self.beta_


# ------------------------------------------------------------------- distances
def _dist(Q, P, metric):
    """Q (n_q, d) queries, P (n_p, d) prototypes -> (n_q, n_p) distance."""
    if metric == "cosine":
        Qn = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), EPS)
        Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), EPS)
        return 1.0 - Qn @ Pn.T
    if metric == "correlation":
        Qc = Q - Q.mean(axis=1, keepdims=True)
        Pc = P - P.mean(axis=1, keepdims=True)
        Qn = Qc / np.maximum(np.linalg.norm(Qc, axis=1, keepdims=True), EPS)
        Pn = Pc / np.maximum(np.linalg.norm(Pc, axis=1, keepdims=True), EPS)
        return 1.0 - Qn @ Pn.T
    if metric == "seuclidean":
        # features are already standardised by the train-fitted Standardizer,
        # so plain Euclidean here IS standardised Euclidean
        return np.sqrt(np.maximum(
            (Q * Q).sum(1)[:, None] + (P * P).sum(1)[None, :] - 2.0 * Q @ P.T, 0.0))
    raise ValueError(metric)


def rank_matrix(D):
    """R[i, j] = rank of prototype j for query i (1 = closest). Ties get average rank.
    Precomputing this makes the permutation null exact and essentially free."""
    n_q, n_p = D.shape
    R = np.empty((n_q, n_p))
    for i in range(n_q):
        d = D[i]
        o = np.argsort(d, kind="mergesort")
        ds = d[o]
        r = np.empty(n_p)
        j = 0
        while j < n_p:
            k = j
            while k + 1 < n_p and ds[k + 1] == ds[j]:
                k += 1
            r[o[j:k + 1]] = 0.5 * (j + k) + 1.0
            j = k + 1
        R[i] = r
    return R


def ranks_from_dist(D, true_idx):
    return rank_matrix(D)[np.arange(D.shape[0]), true_idx]


def retrieval_metrics(D, true_idx, R=None):
    R = rank_matrix(D) if R is None else R
    n_q, n_p = D.shape
    r = R[np.arange(n_q), true_idx]
    t = D[np.arange(n_q), true_idx]
    Dm = D.copy()
    Dm[np.arange(n_q), true_idx] = np.inf
    best_imp = Dm.min(axis=1)
    imp_sd = np.array([np.std(Dm[i][np.isfinite(Dm[i])]) for i in range(n_q)])
    margins = (best_imp - t) / np.maximum(imp_sd, EPS)
    return dict(n_queries=int(n_q), n_classes=int(n_p),
                rank1=float((r == 1).mean()), rank5=float((r <= 5).mean()),
                mrr=float((1.0 / r).mean()), median_rank=float(np.median(r)),
                mean_rank=float(r.mean()),
                margin_sd_units=float(margins.mean()),
                ranks=r)


def permutation_null(D, true_idx, n_perm=10000, seed=0, R=None):
    """Null = the query -> prototype correspondence is destroyed by permuting the
    prototype identity labels. The distance matrix is held fixed, so this is an exact
    randomisation test of the correspondence rather than a resampling approximation."""
    rng = np.random.default_rng(seed)
    R = rank_matrix(D) if R is None else R
    n_q, n_p = D.shape
    obs = retrieval_metrics(D, true_idx, R)
    perms = np.stack([rng.permutation(n_p) for _ in range(n_perm)])     # (n_perm, n_p)
    idx = perms[:, true_idx]                                            # (n_perm, n_q)
    rr = R[np.arange(n_q)[None, :], idx]                                # (n_perm, n_q)
    r1 = (rr == 1).mean(axis=1)
    mrr = (1.0 / rr).mean(axis=1)
    return dict(
        obs_rank1=obs["rank1"], obs_mrr=obs["mrr"],
        null_rank1_mean=float(r1.mean()), null_rank1_p95=float(np.quantile(r1, 0.95)),
        null_mrr_mean=float(mrr.mean()), null_mrr_p95=float(np.quantile(mrr, 0.95)),
        p_rank1=float((1 + (r1 >= obs["rank1"]).sum()) / (1 + n_perm)),
        p_mrr=float((1 + (mrr >= obs["mrr"]).sum()) / (1 + n_perm)),
        n_perm=n_perm, seed=seed)


# --------------------------------------------------------------- the pipeline
def transfer(Xtr, subtr, Xte, subte, subjects, Ntr=None, Nte=None,
             metrics=METRICS, query_level="recording", n_perm=10000, seed=0):
    """Prototype-from-train / query-from-test identification.

    Xtr, Xte    (n_seg, d) segment feature matrices
    subtr,subte (n_seg,)   subject label per segment
    subjects    ordered list of the subject labels defining the prototype index
    Ntr, Nte    optional nuisance blocks; if given they are regressed out with
                coefficients fitted on TRAIN only.
    """
    if Ntr is not None:
        nr = NuisanceRegressor().fit(Xtr, Ntr)
        Xtr, Xte = nr.transform(Xtr, Ntr), nr.transform(Xte, Nte)
    sc = Standardizer().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)

    sidx = {s: i for i, s in enumerate(subjects)}
    P = np.stack([Ztr[subtr == s].mean(axis=0) for s in subjects])

    if query_level == "recording":
        Q = np.stack([Zte[subte == s].mean(axis=0) for s in subjects])
        true_idx = np.arange(len(subjects))
    else:
        Q = Zte
        true_idx = np.array([sidx[s] for s in subte])

    out = {}
    for m in metrics:
        D = _dist(Q, P, m)
        R = rank_matrix(D)
        met = retrieval_metrics(D, true_idx, R)
        ranks = met.pop("ranks")
        met.update(permutation_null(D, true_idx, n_perm=n_perm, seed=seed, R=R))
        met["theoretical_chance_rank1"] = 1.0 / len(subjects)
        out[m] = met
        out[m + "_ranks"] = ranks.tolist()
    return out
