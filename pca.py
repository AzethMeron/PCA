import numpy as np
from typing import Optional


class PCA:
    """
    PCA with batched fitting, no internal centering.

    Assumes inputs are already centered/standardized externally.

    fit:    batched Gram matrix X^T X → sample covariance → symmetric eigh → descending sort.
    transform: X @ components_  (optionally batched).
    export_to_hpp: embed fitted params as constexpr arrays in a C++ header.

    Args:
        n_components: number of principal components to keep.
    """

    def __init__(self, n_components: int):
        if n_components <= 0:
            raise ValueError("n_components must be a positive integer.")
        self.n_components = int(n_components)
        self.components_: Optional[np.ndarray] = None    # (D, K)
        self.explained_variance_: Optional[np.ndarray] = None   # (K,) raw eigenvalues
        self._explained_ratio_: Optional[np.ndarray] = None     # (K,) ratios
        self.total_variance_: Optional[float] = None
        self.n_features_: Optional[int] = None
        self._fitted = False

    def fit(self, X: np.ndarray, batch_size: int = 64) -> "PCA":
        """
        Fit PCA on pre-centered data X using batched processing.

        Args:
            X:          (N, D) array, assumed already centered.
            batch_size: rows per batch for Gram accumulation.
        Returns:
            self
        """
        if np.ndim(X) != 2:
            raise ValueError("X must be 2D (N, D).")
        N, D = X.shape
        if N < 2:
            raise ValueError("Need at least 2 samples to compute covariance.")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        X = np.asarray(X, dtype=np.float64)
        K = min(self.n_components, D)

        # Batched Gram matrix G = X^T X  (assumes X is mean-centered)
        gram = np.zeros((D, D), dtype=np.float64)
        for start in range(0, N, batch_size):
            batch = X[start:start + batch_size]
            gram += batch.T @ batch

        # Sample covariance
        cov = gram / max(N - 1, 1)

        # Symmetric eigendecomposition (returns ascending order)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Descending sort
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx].clip(0)   # numerical safety
        eigenvectors = eigenvectors[:, idx]

        self.total_variance_ = float(eigenvalues.sum())
        self.components_ = eigenvectors[:, :K]                    # (D, K)
        self.explained_variance_ = eigenvalues[:K].copy()
        if self.total_variance_ > 0.0:
            self._explained_ratio_ = self.explained_variance_ / self.total_variance_
        else:
            self._explained_ratio_ = np.zeros(K, dtype=np.float64)
        self.n_features_ = D
        self._fitted = True
        return self

    def transform(self, X: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Project pre-centered X onto learned components.

        Args:
            X:          (N, D) array, centered the same way as fit data.
            batch_size: if provided, processes in chunks.
        Returns:
            (N, K) array.
        """
        if not self._fitted:
            raise RuntimeError("PCA must be fit before calling transform().")
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError("X must be 2D (N, D).")
        N, D = X.shape
        if D != self.n_features_:
            raise ValueError(
                f"X has D={D} features, but PCA was fit with D={self.n_features_}."
            )
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be a positive integer or None.")

        if batch_size is None or batch_size >= N:
            return X @ self.components_

        K = self.components_.shape[1]
        out = np.empty((N, K), dtype=np.float64)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            out[start:end] = X[start:end] @ self.components_
        return out

    def explained_variance_ratio(self) -> np.ndarray:
        """Ratio of variance explained by each kept component (K,), in [0, 1]."""
        if not self._fitted:
            raise RuntimeError("PCA must be fit before calling explained_variance_ratio().")
        return self._explained_ratio_

    def explained_variance_values(self) -> np.ndarray:
        """Raw variances (eigenvalues) for each kept component (K,)."""
        if not self._fitted:
            raise RuntimeError("PCA must be fit before calling explained_variance_values().")
        return self.explained_variance_

    def export_to_hpp(
        self,
        filepath: str,
        dtype: str = "double",
        namespace: str = "pca_params",
    ) -> None:
        """
        Export fitted parameters to a C++ header for compile-time import.

        Include the generated file and call <namespace>::make_pca() to obtain
        a ready-to-use pca::PCA<dtype> object loaded with the trained parameters.

        Args:
            filepath:  output .hpp path.
            dtype:     C++ scalar type — "double" or "float".
            namespace: C++ namespace wrapping the exported symbols.
        """
        if not self._fitted:
            raise RuntimeError("PCA must be fit before calling export_to_hpp().")
        if dtype not in ("double", "float"):
            raise ValueError("dtype must be 'double' or 'float'.")

        D, K = self.components_.shape
        # Flatten row-major: flat[d * K + k] = components_[d, k]
        comp_flat = self.components_.flatten(order="C")
        ev = self.explained_variance_
        er = self._explained_ratio_

        def _fmt(arr: np.ndarray) -> str:
            return ", ".join(f"{v:.17g}" for v in arr)

        lines = [
            "// generated by pca.py — do not edit",
            "#pragma once",
            "#include <vector>",
            '#include "pca.hpp"',
            "",
            f"namespace {namespace} {{",
            "",
            f"    constexpr int D = {D};",
            f"    constexpr int K = {K};",
            f"    constexpr {dtype} total_variance = {self.total_variance_:.17g};",
            "",
            "    // components[d * K + k] = component of feature d in eigenvector k",
            f"    constexpr {dtype} components[] = {{{_fmt(comp_flat)}}};",
            "",
            f"    constexpr {dtype} explained_variance[] = {{{_fmt(ev)}}};",
            f"    constexpr {dtype} explained_ratio[]    = {{{_fmt(er)}}};",
            "",
            f"    inline pca::PCA<{dtype}> make_pca() {{",
            f"        pca::PCA<{dtype}> p(K);",
            f"        p.load(D, K,",
            f"               std::vector<{dtype}>(components,         components         + D * K),",
            f"               std::vector<{dtype}>(explained_variance, explained_variance + K),",
            f"               std::vector<{dtype}>(explained_ratio,    explained_ratio    + K),",
            f"               total_variance);",
            f"        return p;",
            f"    }}",
            "",
            f"}} // namespace {namespace}",
            "",
        ]
        with open(filepath, "w") as f:
            f.write("\n".join(lines))
