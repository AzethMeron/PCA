// Full C++ audit: loaded-params transforms, Jacobi fit, whitening variance,
// transform_one / transform_one_raw consistency, error handling.
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pca.hpp"
#include "lcg_full_pca_params.hpp"   // full_pca_params::make_pca()

// ---------------------------------------------------------------------------
// LCG + data generation (must match test_full_audit.py exactly)
// ---------------------------------------------------------------------------
static constexpr uint32_t LCG_SEED = 143;
static constexpr int      GEN_N    = 3000;
static constexpr int      GEN_D    = 100;
static constexpr int      REF_NTX  = 50;
static constexpr int      REF_K    = 20;

static inline uint32_t lcg(uint32_t seed) {
    return 1664525u * seed + 1013904223u;
}

static std::vector<std::vector<double>> gen_data() {
    std::vector<std::vector<double>> X(GEN_N, std::vector<double>(GEN_D));
    uint32_t seed = LCG_SEED;
    for (int i = 0; i < GEN_N; ++i)
        for (int j = 0; j < GEN_D; ++j) {
            seed = lcg(seed);
            X[i][j] = seed / 4294967296.0;
        }
    return X;
}

static void center_inplace(std::vector<std::vector<double>>& X) {
    const int N = static_cast<int>(X.size());
    const int D = static_cast<int>(X[0].size());
    std::vector<double> mean(D, 0.0);
    for (const auto& row : X)
        for (int d = 0; d < D; ++d)
            mean[d] += row[d];
    for (auto& m : mean) m /= N;
    for (auto& row : X)
        for (int d = 0; d < D; ++d)
            row[d] -= mean[d];
}

// ---------------------------------------------------------------------------
// Reference file reader
// ---------------------------------------------------------------------------
struct Ref {
    std::vector<std::vector<double>> tx;    // (N_TX, K)
    std::vector<std::vector<double>> wtx;   // (N_TX, K)
    std::vector<double>              evals; // (K,)
};

static Ref read_ref(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path);
    Ref r;
    auto read_block2d = [&](std::vector<std::vector<double>>& blk) {
        int rows, cols;
        f >> rows >> cols;
        blk.resize(rows, std::vector<double>(cols));
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                f >> blk[i][j];
    };
    read_block2d(r.tx);
    read_block2d(r.wtx);
    int k; f >> k;
    r.evals.resize(k);
    for (auto& v : r.evals) f >> v;
    return r;
}

// ---------------------------------------------------------------------------
// Reporting helpers
// ---------------------------------------------------------------------------
static int g_pass = 0, g_fail = 0;

static bool chk(const std::string& name, bool ok, const std::string& detail = "") {
    std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << name;
    if (!detail.empty()) std::cout << "  (" << detail << ")";
    std::cout << "\n";
    ok ? ++g_pass : ++g_fail;
    return ok;
}

static void hdr(const std::string& title) {
    std::cout << "\n" << std::string(64, '=') << "\n"
              << "  " << title << "\n"
              << std::string(64, '=') << "\n";
}

static double max_abs_diff(const std::vector<std::vector<double>>& a,
                           const std::vector<std::vector<double>>& b) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < a[i].size(); ++j)
            m = std::max(m, std::abs(a[i][j] - b[i][j]));
    return m;
}

// ---------------------------------------------------------------------------
int main() {
    // ---- load reference data -----------------------------------------------
    Ref ref;
    try {
        ref = read_ref("lcg_full_ref.txt");
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    // ---- generate and center data ------------------------------------------
    auto X = gen_data();
    center_inplace(X);

    // ---- 1. Loaded-params transform -----------------------------------------
    hdr("1. Loaded-params transform (vs Python reference)");
    {
        auto p = full_pca_params::make_pca();

        std::vector<std::vector<double>> tx_cpp(REF_NTX);
        for (int i = 0; i < REF_NTX; ++i)
            tx_cpp[i] = p.transform_one(X[i]);

        double diff = max_abs_diff(tx_cpp, ref.tx);
        chk("Normal transform  abs_err < 1e-10", diff < 1e-10,
            "max_abs=" + std::to_string(diff));

        // whitened
        std::vector<std::vector<double>> wtx_cpp(REF_NTX);
        for (int i = 0; i < REF_NTX; ++i)
            wtx_cpp[i] = p.transform_one(X[i], /*whiten=*/true);

        double wdiff = max_abs_diff(wtx_cpp, ref.wtx);
        chk("Whitened transform  abs_err < 1e-10", wdiff < 1e-10,
            "max_abs=" + std::to_string(wdiff));
    }

    // ---- 2. transform_one vs transform_one_raw consistency -----------------
    hdr("2. transform_one vs transform_one_raw consistency");
    {
        auto p = full_pca_params::make_pca();
        auto v1 = p.transform_one(X[0]);
        auto v2 = p.transform_one_raw(X[0].data(), X[0].size());

        bool same = (v1.size() == v2.size());
        for (std::size_t j = 0; same && j < v1.size(); ++j)
            same = (v1[j] == v2[j]);
        chk("transform_one == transform_one_raw (bit-exact)", same);

        // whitened
        auto w1 = p.transform_one(X[0], true);
        auto w2 = p.transform_one_raw(X[0].data(), X[0].size(), true);
        bool wsame = (w1.size() == w2.size());
        for (std::size_t j = 0; wsame && j < w1.size(); ++j)
            wsame = (w1[j] == w2[j]);
        chk("whitened: transform_one == transform_one_raw (bit-exact)", wsame);
    }

    // ---- 3. transform() delegates to transform_one() ----------------------
    hdr("3. transform() == per-row transform_one()");
    {
        auto p = full_pca_params::make_pca();
        auto batch = p.transform(
            std::vector<std::vector<double>>(X.begin(), X.begin() + REF_NTX));

        double diff = 0.0;
        for (int i = 0; i < REF_NTX; ++i) {
            auto row = p.transform_one(X[i]);
            for (int j = 0; j < REF_K; ++j)
                diff = std::max(diff, std::abs(batch[i][j] - row[j]));
        }
        chk("transform() matches per-row transform_one()  diff==0", diff == 0.0,
            "max_abs=" + std::to_string(diff));
    }

    // ---- 4. C++ Jacobi fit eigenvalues vs Python reference -----------------
    hdr("4. C++ Jacobi fit eigenvalues vs Python eigh");
    {
        pca::PCA<double> p(REF_K);
        p.fit(X);
        const auto& ev_cpp = p.explained_variance_values();

        double max_rel = 0.0;
        for (int k = 0; k < REF_K; ++k) {
            double denom = std::abs(ref.evals[k]) + 1e-300;
            max_rel = std::max(max_rel, std::abs(ev_cpp[k] - ref.evals[k]) / denom);
        }
        chk("Jacobi eigenvalues  rel_err < 1e-6", max_rel < 1e-6,
            "max_rel=" + std::to_string(max_rel));
    }

    // ---- 5. Whitening output variance --------------------------------------
    hdr("5. Whitening output variance (full dataset)");
    {
        auto p = full_pca_params::make_pca();

        // Compute whitened transform of all N samples
        const int N = GEN_N;
        const int K = REF_K;
        std::vector<std::vector<double>> wtx(N);
        for (int i = 0; i < N; ++i)
            wtx[i] = p.transform_one(X[i], /*whiten=*/true);

        // Sample variance (ddof=1) of each output dimension should be ≈ 1.0
        double max_var_err = 0.0;
        for (int k = 0; k < K; ++k) {
            double sum = 0.0, sum2 = 0.0;
            for (int i = 0; i < N; ++i) {
                sum  += wtx[i][k];
                sum2 += wtx[i][k] * wtx[i][k];
            }
            double mean = sum / N;
            double var  = (sum2 - N * mean * mean) / (N - 1);   // ddof=1
            max_var_err = std::max(max_var_err, std::abs(var - 1.0));
        }
        chk("Whitened sample-var ≈ 1.0  abs_err < 1e-10", max_var_err < 1e-10,
            "max_abs=" + std::to_string(max_var_err));
    }

    // ---- 6. n_components clamping ------------------------------------------
    hdr("6. n_components clamping (K > D)");
    {
        pca::PCA<double> p(GEN_D + 999);  // ask for more than D features
        p.fit(X);
        bool clamped = (p.n_features() == GEN_D);
        chk("k_ clamped to D when n_components > D", clamped,
            "n_features=" + std::to_string(p.n_features()));
    }

    // ---- 7. Error handling --------------------------------------------------
    hdr("7. Error handling");
    {
        // transform before fit
        try {
            pca::PCA<double> p(5);
            p.transform(std::vector<std::vector<double>>(1, std::vector<double>(GEN_D)));
            chk("transform before fit throws", false);
        } catch (const std::runtime_error&) {
            chk("transform before fit throws", true);
        }

        // transform_one wrong size
        try {
            auto p = full_pca_params::make_pca();
            p.transform_one(std::vector<double>(GEN_D + 1, 0.0));
            chk("transform_one wrong size throws", false);
        } catch (const std::invalid_argument&) {
            chk("transform_one wrong size throws", true);
        }

        // transform_one_raw wrong size
        try {
            auto p = full_pca_params::make_pca();
            std::vector<double> buf(GEN_D + 1, 0.0);
            p.transform_one_raw(buf.data(), buf.size());
            chk("transform_one_raw wrong size throws", false);
        } catch (const std::invalid_argument&) {
            chk("transform_one_raw wrong size throws", true);
        }

        // load with bad components size
        try {
            pca::PCA<double> p(5);
            p.load(10, 5,
                   std::vector<double>(99),   // should be 50
                   std::vector<double>(5),
                   std::vector<double>(5),
                   1.0);
            chk("load bad components size throws", false);
        } catch (const std::invalid_argument&) {
            chk("load bad components size throws", true);
        }

        // load with K <= 0
        try {
            pca::PCA<double> p(5);
            p.load(0, 5,
                   std::vector<double>(0),
                   std::vector<double>(5),
                   std::vector<double>(5),
                   1.0);
            chk("load D=0 throws", false);
        } catch (const std::invalid_argument&) {
            chk("load D=0 throws", true);
        }

        // PCA with n_components <= 0
        try {
            pca::PCA<double> p(0);
            chk("PCA(0) throws", false);
        } catch (const std::invalid_argument&) {
            chk("PCA(0) throws", true);
        }
    }

    // ---- 8. float type (smoke test) ----------------------------------------
    hdr("8. float type smoke test");
    {
        pca::PCA<float> fp(REF_K);
        std::vector<std::vector<float>> Xf(GEN_N, std::vector<float>(GEN_D));
        for (int i = 0; i < GEN_N; ++i)
            for (int j = 0; j < GEN_D; ++j)
                Xf[i][j] = static_cast<float>(X[i][j]);
        fp.fit(Xf);

        const auto& ev_f = fp.explained_variance_values();
        double max_rel = 0.0;
        for (int k = 0; k < REF_K; ++k) {
            double denom = std::abs(ref.evals[k]) + 1e-300;
            max_rel = std::max(max_rel, std::abs(static_cast<double>(ev_f[k]) - ref.evals[k]) / denom);
        }
        // float precision: expect rel_err < 1e-5
        chk("float PCA eigenvalues  rel_err < 1e-5", max_rel < 1e-5,
            "max_rel=" + std::to_string(max_rel));

        auto tx_f = fp.transform_one(Xf[0]);
        chk("float transform_one returns K values",
            static_cast<int>(tx_f.size()) == REF_K);
    }

    // ---- summary ------------------------------------------------------------
    hdr("Summary");
    std::cout << "  PASS: " << g_pass << "   FAIL: " << g_fail << "\n";
    std::cout << "  Result: " << (g_fail == 0 ? "ALL PASS" : "SOME FAILURES") << "\n";
    return g_fail > 0 ? 1 : 0;
}
