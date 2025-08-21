"""
# Variance Partitioning – API & Practical Guide

This guide documents two functions for quantifying how multiple categorical features explain variance in high‑dimensional vectors (embeddings, features, etc.). It supports Euclidean and distance‑based (PERMANOVA‑style) analyses, permutation tests, and robust numerics for very high dimensions.

* `variance_partition` — standard, concise implementation.
* `variance_partition_robust` — numerically robust (QR projectors, cosine via normalized Gram). Recommended for large/ill‑conditioned cases.

---

## 1) `variance_partition`

**Signature**

```python
variance_partition(
    X: np.ndarray,
    features: Dict[str, ArrayLike],
    metric: str = "euclidean",
    n_perm: int = 0,
    random_state: int = 0,
) -> pandas.DataFrame
```

**Purpose**
Compute **marginal** and **partial** variance explained by each categorical feature in a unified projection/trace framework. With `metric="euclidean"` it matches classical (M)ANOVA on centered data; with other metrics it performs a PERMANOVA‑style analysis via Gower centering.

**Parameters**

* `X` `(n, d)`: Data matrix (rows = samples). Will be column‑centered automatically for Euclidean.
* `features`: Dict mapping feature name → label array of length `n`. Each feature may have arbitrary categorical levels.
* `metric`: Distance/affinity metric.

  * `"euclidean"` (default): uses centered Gram `A = X_c X_c^T`.
  * Other strings supported by `scipy.spatial.distance.pdist` (e.g., `"cosine"`, `"cityblock"`) are converted using **Gower centering**: `A = -1/2 * J D^2 J`.
* `n_perm`: Number of label permutations per feature for a partial‑effect p‑value (0 = skip).
* `random_state`: RNG seed for reproducibility when permuting.

**Returns**
`pandas.DataFrame` with one row per feature and columns:

* `feature`: feature name
* `levels`: number of unique categories
* `df_effect`: effective degrees of freedom (rank of that feature’s design)
* `df_resid`: residual df for the full model
* `SS_total`: total sum of squares (trace of `A`)
* `SSR_marginal`: sum of squares explained by the feature **alone**
* `R2_marginal = SSR_marginal / SS_total`
* `SSR_partial`: sum of squares **beyond** other features (unique contribution)
* `R2_partial = SSR_partial / SS_total`
* `eta2_partial = SSR_partial / (SSR_partial + SSE_full)`
* `p_partial_perm`: permutation p‑value for partial effect (if `n_perm>0`)

**Notes**

* Projectors are computed as `P = Z (Z^T Z)^+ Z^T` (Moore‑Penrose). For extremely high‑dimensional or ill‑conditioned `Z`, consider the robust version below.

**Example**

```python
from variance_partition import variance_partition
res = variance_partition(
    X,
    {"obj_id": obj_id, "shape1": shape1, "shape2": shape2,
     "spatial_relationship": spatial, "color1": color1, "color2": color2},
    metric="euclidean",
    n_perm=999,
    random_state=0,
)
print(res.sort_values("R2_partial", ascending=False))
```

---

## 2) `variance_partition_robust`

**Signature**

```python
variance_partition_robust(
    X: np.ndarray,
    features: Dict[str, ArrayLike],
    metric: str = "euclidean",
    n_perm: int = 0,
    random_state: int = 0,
    tol: float = 1e-10,
) -> pandas.DataFrame
```

**Purpose**
Drop‑in replacement that avoids SVD and is more stable at large scale/high collinearity. It:

* Builds projectors via **QR with column pivoting** (`P = QQ^T`), selecting rank using `tol`.
* Implements a **cosine** variant that normalizes rows (avoid NaN from zero rows) and centers a cosine Gram: `A = J (X_n X_n^T) J`.
* Cleans non‑finite rows; distance fallbacks set non‑finite to 0 to stabilize Gower centering.

**Parameters**

* Same as `variance_partition`, plus:
* `tol`: tolerance for rank determination from QR (`1e-10` → tighten/loosen as needed, e.g., `1e-8`).

**Returns**
Same schema as `variance_partition`.

**Example**

```python
from variance_partition_robust import variance_partition_robust
res = variance_partition_robust(
    X,
    {"obj_id": obj_id, "shape1": shape1, "shape2": shape2,
     "spatial_relationship": spatial, "color1": color1, "color2": color2},
    metric="cosine",
    n_perm=199,
    random_state=0,
    tol=1e-8,
)
```

---

## 3) Mathematical Summary

Let $X \in \mathbb{R}^{n\times d}$. Define a centered Gram $A$ on samples:

* **Euclidean**: $A = X_c X_c^\top$, where $X_c$ column‑centered; equivalently $A = -\tfrac12 J D^2 J$ with Euclidean distances.
* **General metric**: $A = -\tfrac12 J D^2 J$ (Gower centering), $J = I - \tfrac{1}{n}\mathbf{11}^\top$.

For each categorical feature $f$, build a **centered** one‑hot design $Z_f$ (columns sum to 0). Let $Z_{all} = [Z_1, Z_2, \dots]$. Define orthogonal projectors onto the design column spaces:
$P_f,\; P_{all},\; P_{others}$

Effects are computed as traces over $A$:

* **Marginal** SSR: $\mathrm{SSR}_f^{\text{marg}} = \mathrm{tr}(A P_f)$
* **Partial** SSR (unique): $\mathrm{SSR}_f^{\text{part}} = \mathrm{tr}\big(A (P_{all} - P_{others})\big)$
* **Totals**: $ \mathrm{SS}_{tot} = \mathrm{tr}(A)$, $\mathrm{SSE}_{full} = \mathrm{tr}(A(I-P_{all})) $
* **Indices**: $R^2 = \mathrm{SSR}/\mathrm{SS}_{tot}$, $\eta^2_{partial} = \mathrm{SSR}_f^{\text{part}} / (\mathrm{SSR}_f^{\text{part}} + \mathrm{SSE}_{full})$.

Permutation p‑values are computed by permuting labels of $f$ while keeping others fixed and recomputing the partial SSR.

---

## 4) Interpreting Results

* **Use `R2_partial`** (and `eta2_partial`) to rank features by **unique** variance explained beyond the others.
* **`R2_marginal` vs `R2_partial`**: If marginal ≫ partial, the feature’s signal overlaps heavily with other features.
* **Negative `R2_partial`** may occur from numerical noise; treat as 0.
* **`p_partial_perm`**: small values (e.g., `< 0.05`) indicate the unique effect is unlikely under the null of label exchangeability.

---

## 5) Practical Tips & Pitfalls

* **Ill‑conditioning / huge designs**: prefer `variance_partition_robust` (QR projectors), increase `tol` to drop near‑collinear columns.
* **Cosine metric**: normalize rows; robust version already handles this. Beware zero‑norm rows.
* **Rare levels**: merge very small categories (e.g., <5 samples) to stabilize rank and p‑values.
* **Speed**: permutation `n_perm` dominates runtime; start with 49–199 and increase if needed.
* **Memory**: avoid forming big projectors explicitly when rolling your own — use trace identities with an orthonormal basis `Q` so `tr(A QQ^T) = ||A^{1/2} Q||_F^2`.

---

## 6) End‑to‑End Example (Euclidean)

```python
from variance_partition import variance_partition

res = variance_partition(
    X,  # (n, d)
    {
      "obj_id": obj_id,
      "shape1": shape1,
      "shape2": shape2,
      "spatial_relationship": spatial,
      "color1": color1,
      "color2": color2,
    },
    metric="euclidean",
    n_perm=999,
    random_state=0,
)
# Rank by unique contribution
res = res.sort_values("R2_partial", ascending=False)
print(res[["feature","R2_partial","eta2_partial","p_partial_perm"]])
```

---

## 7) Troubleshooting

* **`LinAlgError: SVD did not converge`**

  * Switch to `variance_partition_robust`.
  * Or: clean NaN/Inf rows, merge rare levels, add tiny ridge to `(Z^T Z)`, or increase `tol`.
* **`cosine` gives NaN**

  * Ensure no zero‑norm rows; use robust version which normalizes and centers a cosine Gram.
* **Huge runtime**

  * Reduce `n_perm`, or pre‑aggregate identical rows/labels when appropriate.

---

## 8) Versioning & Reproducibility

* Always fix `random_state` when reporting permutation p‑values.
* Record `metric`, `n_perm`, and `tol` in your Methods.

---

**Credits**: Built around projection‑trace ANOVA in RKHS / PERMANOVA frameworks, with QR‑based stabilization for high‑dimensional designs.

"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import pinv

def _center_columns(X):
    return X - X.mean(axis=0, keepdims=True)

def _make_design(labels):
    labels = np.asarray(labels)
    cats, inv = np.unique(labels, return_inverse=True)
    n, L = len(labels), len(cats)
    Z = np.zeros((n, L))
    Z[np.arange(n), inv] = 1.0
    Z = Z - Z.mean(axis=0, keepdims=True)
    return Z, cats

def _block_concat(mats):
    return np.concatenate(mats, axis=1) if len(mats) > 0 else np.zeros((0,0))

def _projection(Z):
    if Z.size == 0:
        return np.zeros((0,0))
    G = Z.T @ Z
    return Z @ pinv(G) @ Z.T

def _gram_from_metric(X, metric="euclidean"):
    n = X.shape[0]
    J = np.eye(n) - np.ones((n,n))/n
    if metric == "euclidean":
        Xc = _center_columns(X)
        return Xc @ Xc.T
    else:
        D = squareform(pdist(X, metric=metric))
        D2 = D**2
        return -0.5 * J @ D2 @ J

def variance_partition(X, features, metric="euclidean", n_perm=0, random_state=0):
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    A = _gram_from_metric(X, metric=metric)
    SS_total = np.trace(A)

    Zs = {}
    cats_by = {}
    for name, lab in features.items():
        Zs[name], cats_by[name] = _make_design(lab)

    Z_all = _block_concat([Zs[k] for k in Zs.keys()]) if len(Zs) > 0 else np.zeros((n,0))
    P_all = _projection(Z_all) if Z_all.size else np.zeros((n,n))
    SSE_full = np.trace(A @ (np.eye(n) - P_all))
    df_all = int(np.round(np.linalg.matrix_rank(Z_all))) if Z_all.size else 0
    df_resid = n - 1 - df_all

    rows = []
    for name in Zs.keys():
        Z_f = Zs[name]
        P_f_marg = _projection(Z_f)
        SSR_marg = np.trace(A @ P_f_marg)
        R2_marg = SSR_marg / SS_total if SS_total > 0 else np.nan

        Z_others = _block_concat([Zs[k] for k in Zs.keys() if k != name])
        P_others = _projection(Z_others) if Z_others.size else np.zeros((n,n))
        P_f_partial = P_all - P_others
        SSR_partial = np.trace(A @ P_f_partial)
        R2_partial = SSR_partial / SS_total if SS_total > 0 else np.nan
        eta2_partial = SSR_partial / (SSR_partial + SSE_full) if (SSR_partial + SSE_full) > 0 else np.nan

        df_f = int(np.round(np.linalg.matrix_rank(Z_f)))
        pval = None
        if n_perm and df_f > 0:
            greater_equal = 0
            for _ in range(n_perm):
                perm_idx = rng.permutation(n)
                Z_f_perm, _ = _make_design(np.array(features[name])[perm_idx])
                Z_all_perm = _block_concat([Z_f_perm] + [Zs[k] for k in Zs.keys() if k != name])
                P_all_perm = _projection(Z_all_perm)
                P_f_partial_perm = P_all_perm - P_others
                SSR_perm = np.trace(A @ P_f_partial_perm)
                if SSR_perm >= SSR_partial - 1e-12:
                    greater_equal += 1
            pval = (greater_equal + 1) / (n_perm + 1)

        rows.append({
            "feature": name,
            "levels": len(cats_by[name]),
            "df_effect": df_f,
            "df_resid": df_resid,
            "SS_total": SS_total,
            "SSR_marginal": SSR_marg,
            "R2_marginal": R2_marg,
            "SSR_partial": SSR_partial,
            "R2_partial": R2_partial,
            "eta2_partial": eta2_partial,
            "p_partial_perm": pval,
        })

    df = pd.DataFrame(rows).sort_values("R2_partial", ascending=False).reset_index(drop=True)
    return df
