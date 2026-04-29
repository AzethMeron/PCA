#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace pca {

namespace detail {

// ---------------------------------------------------------------------------
// Flat row-major matrix helper
// ---------------------------------------------------------------------------
template<typename T>
struct Matrix {
    int rows, cols;
    std::vector<T> data;

    Matrix() : rows(0), cols(0) {}
    Matrix(int r, int c, T init = T{0})
        : rows(r), cols(c), data(static_cast<std::size_t>(r) * c, init) {}

    T&       at(int i, int j)       noexcept { return data[i * cols + j]; }
    const T& at(int i, int j) const noexcept { return data[i * cols + j]; }
};

// Reorder eigenvalues descending and permute eigenvector columns accordingly.
template<typename T>
void sort_descending(std::vector<T>& vals, Matrix<T>& vecs) {
    const int D = static_cast<int>(vals.size());
    std::vector<int> idx(D);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return vals[a] > vals[b]; });

    std::vector<T> sv(D);
    Matrix<T> sm(D, D);
    for (int k = 0; k < D; ++k) {
        sv[k] = vals[idx[k]];
        for (int d = 0; d < D; ++d)
            sm.at(d, k) = vecs.at(d, idx[k]);
    }
    vals = std::move(sv);
    vecs = std::move(sm);
}

// ---------------------------------------------------------------------------
// Cyclic Jacobi eigendecomposition for a real symmetric matrix A.
//
// On exit:  A is diagonal (eigenvalues on diagonal);
//           columns of V are the corresponding orthonormal eigenvectors.
//
// Uses the tangent-half-angle formulation (Golub & Van Loan §8.4) which
// avoids catastrophic cancellation when |tau| is large.
// ---------------------------------------------------------------------------
template<typename T>
void jacobi_eigen(Matrix<T>& A, Matrix<T>& V, int max_sweeps = 100) {
    const int D   = A.rows;
    const T   eps = std::numeric_limits<T>::epsilon() * T{8};

    // V = identity
    for (int i = 0; i < D; ++i)
        for (int j = 0; j < D; ++j)
            V.at(i, j) = (i == j) ? T{1} : T{0};

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        bool any = false;

        for (int p = 0; p < D - 1; ++p) {
            for (int q = p + 1; q < D; ++q) {
                const T apq = A.at(p, q);
                // Skip if pivot is negligible relative to the diagonal entries
                if (std::abs(apq) <=
                        eps * std::sqrt(std::abs(A.at(p, p)) * std::abs(A.at(q, q))))
                    continue;

                any = true;

                // Compute Givens rotation coefficients
                const T tau = (A.at(q, q) - A.at(p, p)) / (T{2} * apq);
                const T t   = std::copysign(T{1}, tau) /
                              (std::abs(tau) + std::sqrt(T{1} + tau * tau));
                const T c   = T{1} / std::sqrt(T{1} + t * t);
                const T s   = t * c;

                // Zero the pivot and update diagonal
                A.at(p, q) = T{0};
                A.at(q, p) = T{0};
                A.at(p, p) -= t * apq;
                A.at(q, q) += t * apq;

                // Update off-diagonal rows / cols  r ∉ {p, q}
                for (int r = 0; r < D; ++r) {
                    if (r == p || r == q) continue;
                    const T arp = A.at(r, p), arq = A.at(r, q);
                    A.at(r, p) = A.at(p, r) = c * arp - s * arq;
                    A.at(r, q) = A.at(q, r) = s * arp + c * arq;
                }

                // Accumulate rotation into eigenvector matrix
                for (int r = 0; r < D; ++r) {
                    const T vrp = V.at(r, p), vrq = V.at(r, q);
                    V.at(r, p) = c * vrp - s * vrq;
                    V.at(r, q) = s * vrp + c * vrq;
                }
            }
        }

        if (!any) break;  // converged: all off-diagonal elements below threshold
    }
}

} // namespace detail

// ---------------------------------------------------------------------------
// PCA — principal component analysis, no internal centering.
//
// Assumes inputs are already centered / standardized externally.
//
// fit:       batched Gram matrix X^T X → sample covariance → Jacobi eigh
//            → descending sort → keep K leading components.
// transform: X @ components_  (optionally chunked).
// load:      restore from pre-fitted parameters (e.g. from a generated header).
//
// Template parameter T: scalar type (default double).
// ---------------------------------------------------------------------------
template<typename T = double>
class PCA {
public:
    explicit PCA(int n_components)
        : n_components_(n_components), d_(0), k_(0), total_var_(T{0}), fitted_(false) {
        if (n_components <= 0)
            throw std::invalid_argument("n_components must be a positive integer.");
    }

    // ------------------------------------------------------------------
    // Fit on pre-centered (N, D) data.
    // ------------------------------------------------------------------
    void fit(const std::vector<std::vector<T>>& X, int batch_size = 64) {
        if (X.empty())
            throw std::invalid_argument("X must not be empty.");
        if (batch_size <= 0)
            throw std::invalid_argument("batch_size must be a positive integer.");

        const int N = static_cast<int>(X.size());
        const int D = static_cast<int>(X[0].size());
        if (N < 2)
            throw std::invalid_argument("Need at least 2 samples to compute covariance.");
        for (const auto& row : X)
            if (static_cast<int>(row.size()) != D)
                throw std::invalid_argument("All rows of X must have the same length.");

        d_ = D;
        k_ = std::min(n_components_, D);

        // Batched Gram matrix G = X^T X
        detail::Matrix<T> gram(D, D);
        for (int start = 0; start < N; start += batch_size) {
            const int end = std::min(start + batch_size, N);
            for (int n = start; n < end; ++n)
                for (int d1 = 0; d1 < D; ++d1)
                    for (int d2 = 0; d2 < D; ++d2)
                        gram.at(d1, d2) += X[n][d1] * X[n][d2];
        }

        // Sample covariance (assumes X is mean-centered)
        const T denom = T(std::max(N - 1, 1));
        for (auto& v : gram.data) v /= denom;

        // Symmetric eigendecomposition
        detail::Matrix<T> V(D, D);
        detail::jacobi_eigen(gram, V);

        // Eigenvalues are on the diagonal; clamp floating-point negatives
        std::vector<T> evals(D);
        for (int i = 0; i < D; ++i)
            evals[i] = std::max(gram.at(i, i), T{0});

        // Descending sort
        detail::sort_descending(evals, V);

        // Total variance = sum of all D eigenvalues
        total_var_ = T{0};
        for (const T e : evals) total_var_ += e;

        // Store K leading components; layout: components_[d * K + k]
        components_.resize(static_cast<std::size_t>(D) * k_);
        expl_var_.resize(k_);
        expl_ratio_.resize(k_);
        for (int k = 0; k < k_; ++k) {
            expl_var_[k]   = evals[k];
            expl_ratio_[k] = (total_var_ > T{0}) ? evals[k] / total_var_ : T{0};
            for (int d = 0; d < D; ++d)
                components_[d * k_ + k] = V.at(d, k);
        }
        fitted_ = true;
    }

    // ------------------------------------------------------------------
    // Project a single pre-centered sample x (length D) → vector of K scores.
    // ------------------------------------------------------------------
    std::vector<T> transform_one(const std::vector<T>& x) const {
        check_fitted();
        if (static_cast<int>(x.size()) != d_)
            throw std::invalid_argument("x size does not match fitted dimension.");
        return transform_one_impl(x.data());
    }

    // ------------------------------------------------------------------
    // Same as transform_one but accepts a raw pointer + explicit size.
    // Useful when the caller already manages a contiguous buffer.
    // ------------------------------------------------------------------
    std::vector<T> transform_one_raw(const T* data, std::size_t size) const {
        check_fitted();
        if (static_cast<int>(size) != d_)
            throw std::invalid_argument("data size does not match fitted dimension.");
        return transform_one_impl(data);
    }

    // ------------------------------------------------------------------
    // Project pre-centered (N, D) data → (N, K) using transform_one().
    // ------------------------------------------------------------------
    std::vector<std::vector<T>> transform(
            const std::vector<std::vector<T>>& X, int batch_size = 0) const {
        check_fitted();
        if (X.empty())
            throw std::invalid_argument("X must not be empty.");
        (void)batch_size;  // row-by-row via transform_one; batching is a no-op

        const int N = static_cast<int>(X.size());
        if (static_cast<int>(X[0].size()) != d_)
            throw std::invalid_argument(
                "X feature dimension does not match fitted dimension.");

        std::vector<std::vector<T>> out(N);
        for (int i = 0; i < N; ++i)
            out[i] = transform_one(X[i]);
        return out;
    }

    // ------------------------------------------------------------------
    // Load pre-fitted parameters from a generated header.
    //
    // components:         D*K row-major vector; index d*K + k.
    //                     Matches Python's components_.flatten(order='C').
    // explained_variance: K raw eigenvalues.
    // explained_ratio:    K per-component variance ratios.
    // total_variance:     sum of all D eigenvalues.
    // ------------------------------------------------------------------
    void load(int D, int K,
              const std::vector<T>& components,
              const std::vector<T>& explained_variance,
              const std::vector<T>& explained_ratio,
              T                     total_variance) {
        if (D <= 0 || K <= 0)
            throw std::invalid_argument("D and K must be positive.");
        if (static_cast<int>(components.size())         != D * K)
            throw std::invalid_argument("components size must equal D*K.");
        if (static_cast<int>(explained_variance.size()) != K)
            throw std::invalid_argument("explained_variance size must equal K.");
        if (static_cast<int>(explained_ratio.size())    != K)
            throw std::invalid_argument("explained_ratio size must equal K.");
        d_         = D;
        k_         = K;
        components_ = components;
        expl_var_   = explained_variance;
        expl_ratio_ = explained_ratio;
        total_var_ = total_variance;
        fitted_    = true;
    }

    // K ratios (each in [0, 1]); sum gives fraction of variance retained.
    const std::vector<T>& explained_variance_ratio()  const { check_fitted(); return expl_ratio_; }
    // K raw eigenvalues.
    const std::vector<T>& explained_variance_values() const { check_fitted(); return expl_var_; }

    bool is_fitted()    const noexcept { return fitted_; }
    int  n_components() const noexcept { return n_components_; }
    int  n_features()   const noexcept { return d_; }

private:
    void check_fitted() const {
        if (!fitted_)
            throw std::runtime_error("PCA must be fit (or loaded) before use.");
    }

    // Core single-sample projection; caller must ensure data has d_ elements.
    std::vector<T> transform_one_impl(const T* x) const {
        std::vector<T> out(k_, T{0});
        for (int j = 0; j < k_; ++j)
            for (int d = 0; d < d_; ++d)
                out[j] += x[d] * components_[d * k_ + j];
        return out;
    }

    int            n_components_;
    int            d_, k_;
    std::vector<T> components_;   // D*K row-major: index d*K + k
    std::vector<T> expl_var_;     // K raw eigenvalues
    std::vector<T> expl_ratio_;   // K variance ratios
    T              total_var_;
    bool           fitted_;
};

} // namespace pca
