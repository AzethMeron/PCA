"""
Full audit: custom PCA vs sklearn, then export reference data for C++ audit.

Settings: SEED=143, N=3000, D=100, K=20, N_TX=50
"""
import sys
import numpy as np
from pca import PCA

SEED  = 143
N     = 3000
D     = 100
K     = 20
N_TX  = 50

# ---------------------------------------------------------------------------
def lcg(seed: int) -> int:
    return (1664525 * seed + 1013904223) & 0xFFFF_FFFF

def gen_lcg(seed: int, n: int, d: int):
    out = np.empty(n * d, dtype=np.float64)
    for i in range(n * d):
        seed = lcg(seed)
        out[i] = seed / 4294967296.0
    return out.reshape(n, d), seed

def hdr(title: str):
    print(f"\n{'='*64}")
    print(f"  {title}")
    print(f"{'='*64}")

_all_pass = True

def check(name: str, ok: bool, detail: str = "") -> bool:
    global _all_pass
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        _all_pass = False
    return ok

# ---------------------------------------------------------------------------
def main():
    from sklearn.decomposition import PCA as SkPCA

    # ---- data ---------------------------------------------------------------
    X_raw, _ = gen_lcg(SEED, N, D)
    col_mean  = X_raw.mean(axis=0)           # (D,)
    X         = X_raw - col_mean             # centered, what we pass to our PCA

    # ---- fit ----------------------------------------------------------------
    my_pca = PCA(K)
    my_pca.fit(X)

    sk_pca = SkPCA(n_components=K, svd_solver="full")
    sk_pca.fit(X_raw)                        # sklearn centers internally

    hdr("1. Eigenvalues / explained variance")

    my_ev = my_pca.explained_variance_values()   # (K,)
    sk_ev = sk_pca.explained_variance_           # (K,)
    rel   = np.abs(my_ev - sk_ev) / (np.abs(sk_ev) + 1e-300)
    check("Eigenvalues  rel_err < 1e-10", rel.max() < 1e-10,
          f"max_rel={rel.max():.3e}")

    my_er = my_pca.explained_variance_ratio()    # (K,)
    sk_er = sk_pca.explained_variance_ratio_     # (K,)
    check("Explained ratio abs_err < 1e-10",
          np.abs(my_er - sk_er).max() < 1e-10,
          f"max_abs={np.abs(my_er - sk_er).max():.3e}")

    print(f"  eigenvalues top-5 (ours):   {my_ev[:5]}")
    print(f"  eigenvalues top-5 (sklearn): {sk_ev[:5]}")
    print(f"  explained ratio sum: {my_er.sum():.8f}")

    # ---- components (sign-aligned) -----------------------------------------
    hdr("2. Principal components")

    # sklearn.components_ shape (K, D); ours (D, K)
    my_comp = my_pca.components_        # (D, K)
    sk_comp = sk_pca.components_.T      # (D, K)

    # sign convention: align each column of our components to sklearn's
    signs = np.sign(np.einsum("dk,dk->k", my_comp, sk_comp))
    signs[signs == 0] = 1.0
    my_aligned = my_comp * signs        # (D, K)

    abs_comp = np.abs(my_aligned - sk_comp).max()
    check("Components  abs_err < 1e-10", abs_comp < 1e-10,
          f"max_abs={abs_comp:.3e}")

    # ---- transform ----------------------------------------------------------
    hdr("3. Transform (N_TX=%d samples)" % N_TX)

    X_tx     = X[:N_TX]                 # centered
    my_tx    = my_pca.transform(X_tx)   # (N_TX, K)
    sk_tx    = sk_pca.transform(X_raw[:N_TX])  # (N_TX, K)

    # sign-align transform columns (same signs as components)
    my_tx_al = my_tx * signs            # (N_TX, K)
    abs_tx   = np.abs(my_tx_al - sk_tx).max()
    check("Transform  abs_err < 1e-10", abs_tx < 1e-10,
          f"max_abs={abs_tx:.3e}")

    # ---- whitened transform -------------------------------------------------
    hdr("4. Whitened transform (N_TX samples)")

    my_wtx   = my_pca.transform(X_tx, whiten=True)   # (N_TX, K)
    sk_wtx   = sk_pca.transform(X_raw[:N_TX]) / np.sqrt(sk_pca.explained_variance_)
    my_wtx_al = my_wtx * signs
    abs_wtx  = np.abs(my_wtx_al - sk_wtx).max()
    check("Whitened transform  abs_err < 1e-10", abs_wtx < 1e-10,
          f"max_abs={abs_wtx:.3e}")

    # ---- whitening variance on full dataset ---------------------------------
    hdr("5. Whitening output variance")

    wtx_full = my_pca.transform(X, whiten=True)   # (N, K)
    # np.var uses N in denominator; eigenvalues use N-1 → expect ~(N-1)/N
    var_ddof0    = np.var(wtx_full, axis=0)
    expected_var = (N - 1) / N
    abs_wvar     = np.abs(var_ddof0 - expected_var).max()
    check("Whitened var ≈ (N-1)/N  abs_err < 1e-10", abs_wvar < 1e-10,
          f"max_abs={abs_wvar:.3e}, expected≈{expected_var:.7f}")

    # sample variance (ddof=1) should be ≈ 1.0
    var_ddof1 = np.var(wtx_full, axis=0, ddof=1)
    abs_wvar1 = np.abs(var_ddof1 - 1.0).max()
    check("Whitened sample-var ≈ 1.0  abs_err < 1e-10", abs_wvar1 < 1e-10,
          f"max_abs={abs_wvar1:.3e}")

    # ---- error handling -----------------------------------------------------
    hdr("6. Error handling")

    unfitted = PCA(K)
    try:
        unfitted.transform(X[:5])
        check("transform before fit raises", False)
    except RuntimeError:
        check("transform before fit raises", True)

    try:
        unfitted.explained_variance_ratio()
        check("explained_variance_ratio before fit raises", False)
    except RuntimeError:
        check("explained_variance_ratio before fit raises", True)

    try:
        my_pca.transform(X[:5, :D//2])     # wrong D
        check("transform wrong-D raises", False)
    except ValueError:
        check("transform wrong-D raises", True)

    try:
        PCA(-1)
        check("PCA(n_components=-1) raises", False)
    except ValueError:
        check("PCA(n_components=-1) raises", True)

    # ---- export -------------------------------------------------------------
    hdr("7. Export for C++ audit")

    my_pca.export_to_hpp("lcg_full_pca_params.hpp", namespace="full_pca_params")
    print("  Exported lcg_full_pca_params.hpp")

    with open("lcg_full_ref.txt", "w") as f:
        # block 1: normal transforms
        f.write(f"{N_TX} {K}\n")
        for row in my_tx:
            f.write(" ".join(f"{v:.17g}" for v in row) + "\n")
        # block 2: whitened transforms
        f.write(f"{N_TX} {K}\n")
        for row in my_wtx:
            f.write(" ".join(f"{v:.17g}" for v in row) + "\n")
        # block 3: eigenvalues (for C++ Jacobi comparison)
        f.write(f"{K}\n")
        f.write(" ".join(f"{v:.17g}" for v in my_ev) + "\n")
    print("  Exported lcg_full_ref.txt")

    # ---- final --------------------------------------------------------------
    hdr("Summary")
    print(f"  Result: {'ALL PASS' if _all_pass else 'SOME FAILURES'}")
    if not _all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
