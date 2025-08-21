
import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy.linalg import qr as scipy_qr
from scipy.spatial.distance import pdist, squareform

def _center_cols(X):
    return X - X.mean(axis=0, keepdims=True)

def _J(n):
    return np.eye(n) - np.ones((n, n))/n

def _make_design(labels):
    labels = np.asarray(labels)
    cats, inv = np.unique(labels, return_inverse=True)
    n, L = len(labels), len(cats)
    Z = np.zeros((n, L), dtype=np.float64)
    Z[np.arange(n), inv] = 1.0
    Z -= Z.mean(axis=0, keepdims=True)
    return Z, cats

def _gram_from_metric(X, metric="euclidean"):
    n = X.shape[0]
    if metric == "euclidean":
        Xc = _center_cols(X.astype(np.float64))
        return Xc @ Xc.T
    elif metric == "cosine":
        X = X.astype(np.float64)
        row_norms = np.linalg.norm(X, axis=1, keepdims=True)
        row_norms = np.maximum(row_norms, 1e-12)
        Xn = X / row_norms
        return _J(n) @ (Xn @ Xn.T) @ _J(n)
    else:
        D = squareform(pdist(X, metric=metric))
        D2 = D**2
        return -0.5 * _J(n) @ D2 @ _J(n)

def _projector_qr(Z, tol=1e-10):
    if Z.size == 0:
        return np.zeros((0,0)), 0, None
    Q, R, piv = scipy_qr(Z, mode="economic", pivoting=True)
    diag = np.abs(np.diag(R))
    if diag.size == 0:
        return np.zeros((Z.shape[0], Z.shape[0])), 0, None
    thresh = diag[0] * tol if diag[0] > 0 else tol
    r = int((diag > thresh).sum())
    if r == 0:
        return np.zeros((Z.shape[0], Z.shape[0])), 0, None
    Qr = Q[:, :r]
    return Qr @ Qr.T, r, Qr

def variance_partition_with_effects(X, features, metric="euclidean", n_perm=0, random_state=0, robust=True, tol=1e-10, verbose=True):
    """
    Compute variance partitioning stats AND an additive per-level effect model (for metric='euclidean').
    Returns: stats_df, intercept, effects, levels_map
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape

    Zs = {}
    levels_map = {}
    for name, lab in features.items():
        Zs[name], levels_map[name] = _make_design(lab)

    Z_all = np.concatenate([Zs[k] for k in Zs.keys()], axis=1) if len(Zs)>0 else np.zeros((n,0))

    if robust:
        P_all, r_all, _ = _projector_qr(Z_all, tol=tol) if Z_all.size else (np.zeros((n,n)), 0, None)
        df_resid = n - 1 - r_all
    else:
        G = Z_all.T @ Z_all if Z_all.size else np.zeros((0,0))
        P_all = Z_all @ np.linalg.pinv(G) @ Z_all.T if Z_all.size else np.zeros((n,n))
        r_all = int(np.round(np.linalg.matrix_rank(Z_all))) if Z_all.size else 0
        df_resid = n - 1 - r_all

    A = _gram_from_metric(X, metric=metric)
    SS_total = float(np.trace(A))
    SSE_full = float(np.trace(A @ (np.eye(n) - P_all)))
    SSR_all = SS_total - SSE_full
    R2_total = SSR_all / SS_total if SS_total > 0 else np.nan
    if verbose:
        print(f"Total R2 (all features): {R2_total:.6f}")

    rows = []
    for name in Zs.keys():
        Zf = Zs[name]
        if robust:
            P_f_marg, r_f, _ = _projector_qr(Zf, tol=tol)
            Z_others = np.concatenate([Zs[k] for k in Zs.keys() if k != name], axis=1) if len(Zs)>1 else np.zeros((n,0))
            P_others, _, _ = _projector_qr(Z_others, tol=tol)
        else:
            Gf = Zf.T @ Zf
            P_f_marg = Zf @ np.linalg.pinv(Gf) @ Zf.T
            Z_others = np.concatenate([Zs[k] for k in Zs.keys() if k != name], axis=1) if len(Zs)>1 else np.zeros((n,0))
            G_oth = Z_others.T @ Z_others if Z_others.size else np.zeros((0,0))
            P_others = Z_others @ np.linalg.pinv(G_oth) @ Z_others.T if Z_others.size else np.zeros((n,n))
            r_f = int(np.round(np.linalg.matrix_rank(Zf)))

        P_f_partial = P_all - P_others
        SSR_marg = float(np.trace(A @ P_f_marg))
        SSR_part = float(np.trace(A @ P_f_partial))
        R2_marg = SSR_marg / SS_total if SS_total > 0 else np.nan
        R2_part = SSR_part / SS_total if SS_total > 0 else np.nan
        eta2_part = SSR_part / (SSR_part + SSE_full) if (SSR_part + SSE_full) > 0 else np.nan

        pval = None
        if n_perm and r_f > 0:
            ge = 0
            labels = np.array(features[name])
            for _ in range(n_perm):
                perm = rng.permutation(n)
                Zf_perm, _ = _make_design(labels[perm])
                if robust:
                    Z_all_perm = np.concatenate([Zf_perm] + [Zs[k] for k in Zs.keys() if k != name], axis=1)
                    P_all_perm, _, _ = _projector_qr(Z_all_perm, tol=tol)
                else:
                    Z_all_perm = np.concatenate([Zf_perm] + [Zs[k] for k in Zs.keys() if k != name], axis=1)
                    G_allp = Z_all_perm.T @ Z_all_perm
                    P_all_perm = Z_all_perm @ np.linalg.pinv(G_allp) @ Z_all_perm.T
                SSR_perm = float(np.trace(A @ (P_all_perm - P_others)))
                if SSR_perm >= SSR_part - 1e-12:
                    ge += 1
            pval = (ge + 1) / (n_perm + 1)

        rows.append({
            "feature": name,
            "levels": len(levels_map[name]),
            "df_effect": r_f,
            "df_resid": df_resid,
            "SS_total": SS_total,
            "SSR_marginal": SSR_marg,
            "R2_marginal": R2_marg,
            "SSR_partial": SSR_part,
            "R2_partial": R2_part,
            "eta2_partial": eta2_part,
            "p_partial_perm": pval,
        })

    stats_df = pd.DataFrame(rows).sort_values("R2_partial", ascending=False).reset_index(drop=True)

    # Additive effects (Euclidean only)
    effects = {}
    intercept = X.mean(axis=0)
    if metric == "euclidean" and Z_all.size:
        Xc = X - intercept
        if robust:
            Q, R, piv = scipy_qr(Z_all, mode="economic", pivoting=True)
            diag = np.abs(np.diag(R))
            if diag.size == 0 or diag[0] == 0:
                r = 0
            else:
                r = int((diag > diag[0]*tol).sum())
            if r == 0:
                B_hat = np.zeros((Z_all.shape[1], Xc.shape[1]))
            else:
                Qr, Rr, pivr = Q[:, :r], R[:r, :r], piv[:r]
                beta_piv = np.linalg.solve(Rr, Qr.T @ Xc)
                B_hat = np.zeros((Z_all.shape[1], Xc.shape[1]))
                B_hat[pivr, :] = beta_piv
        else:
            B_hat, *_ = lstsq(Z_all, Xc, rcond=None)

        offset = 0
        for name in Zs.keys():
            L = Zs[name].shape[1]
            Ef = B_hat[offset:offset+L, :]
            Ef = Ef - Ef.mean(axis=0, keepdims=True)  # enforce sum-to-zero numerically
            effects[name] = Ef
            offset += L
    else:
        effects = {}

    # Compute total R² for the full additive model
    SSR_all = SS_total - float(np.trace(A @ (np.eye(n) - P_all)))
    R2_total = SSR_all / SS_total if SS_total > 0 else np.nan
    print(f'Total R² (all features): {R2_total:.4f}')
    
    return stats_df, intercept, effects, levels_map, R2_total
