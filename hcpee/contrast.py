"""Phase 2.5 shared machinery for state-CONTRAST identity transfer.

The object under test is not a person's feature vector but a person's *state
transformation*:

    Delta_i^{A->B}(S) = X_{i,B,S} - X_{i,A,S}

evaluated on log-power features. Two facts make this the interesting quantity.

1. If a person-specific volume-conduction / skull filter acts multiplicatively on
   sensor POWER,  P_{i,s,c}(f) = a_{i,c}(f) * p_{i,s,c}(f),  then in log units it is
   additive and the within-session difference between two states cancels it EXACTLY.
   The Phase 2 contrast features are log10 band powers, so the Phase 2 contrast test
   already was the log-ratio formulation. Section 7 of Phase 2.5 makes that explicit
   by comparing it against formulations that do NOT cancel a multiplicative gain.
2. The group-level state term is common to every subject and so contributes nothing
   to between-subject identification.

Scaling convention (identical to Phase 2 `11_state_contrast.py`): mean and SD are
taken from the Session-1 (prototype) side only, never from the query side.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis as A                                                 # noqa: E402
# [release] The EEG feature-table loader (`data as D`) and the two EEG-only helpers that
# used it (rec_mean, deltas) are not shipped; see docs/PATCHES.md.

METRICS = ("cosine", "correlation")


def evaluate(P, Q, n_perm=10000, seed=20260820, metrics=METRICS):
    """P prototypes (row i = subject i, from the earlier session), Q queries.

    Standardisation uses P only. Returns retrieval metrics + exact permutation null
    for each distance metric.
    """
    P = np.asarray(P, np.float64)
    Q = np.asarray(Q, np.float64)
    mu, sd = P.mean(0), np.maximum(P.std(0), 1e-9)
    Pz, Qz = (P - mu) / sd, (Q - mu) / sd
    n = P.shape[0]
    true = np.arange(n)
    out = {}
    for m in metrics:
        Dm = A._dist(Qz, Pz, m)
        R = A.rank_matrix(Dm)
        met = A.retrieval_metrics(Dm, true, R)
        ranks = met.pop("ranks")
        met.update(A.permutation_null(Dm, true, n_perm=n_perm, seed=seed, R=R))
        met["theoretical_chance_rank1"] = 1.0 / n
        met["ranks"] = ranks.tolist()
        out[m] = met
    return out


def similarity_blocks(P, Q):
    """Cosine similarity matrix between standardised P (rows) and Q (rows),
    plus the same-person diagonal and the impostor off-diagonal."""
    mu, sd = P.mean(0), np.maximum(P.std(0), 1e-9)
    Pz, Qz = (P - mu) / sd, (Q - mu) / sd
    a = Pz / np.maximum(np.linalg.norm(Pz, axis=1, keepdims=True), 1e-12)
    b = Qz / np.maximum(np.linalg.norm(Qz, axis=1, keepdims=True), 1e-12)
    S = a @ b.T
    n = S.shape[0]
    return S, np.diag(S).copy(), S[~np.eye(n, dtype=bool)]


def paired_permutation(a, b, n_perm=10000, seed=20260820):
    """Two-sided-safe paired randomisation test of mean(a) > mean(b), a and b paired
    per subject. Sign flips of the within-pair difference."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    obs = d.mean()
    flips = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    null = (flips * d[None, :]).mean(axis=1)
    return dict(mean_a=float(a.mean()), mean_b=float(b.mean()),
                mean_diff=float(obs), sd_diff=float(d.std(ddof=1)),
                cohens_dz=float(obs / max(d.std(ddof=1), 1e-12)),
                p_one_sided=float((1 + (null >= obs).sum()) / (1 + n_perm)),
                n=int(d.size), n_perm=n_perm)


def boot_ci(x, stat=np.mean, n_boot=5000, seed=20260820):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    vals = np.array([stat(x[rng.integers(0, x.size, x.size)]) for _ in range(n_boot)])
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float((a.mean() - b.mean())
                 / np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1))))


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        adj[i] = prev
    return adj
