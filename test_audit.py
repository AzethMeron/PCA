#!/usr/bin/env python3
"""
Deep audit — Python side.

Generates 3000×100 data with a 32-bit LCG, fits PCA (K=10),
exports parameters for C++ comparison, and writes lcg_expected.txt.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pca import PCA

# ──────────────────────────────────────── constants
SEED, N, D, K = 143, 3000, 100, 10

# ──────────────────────────────────────── LCG
def lcg(s: int) -> int:
    """One step of a 32-bit LCG (identical to C uint32 arithmetic)."""
    return (1664525 * s + 1013904223) & 0xFFFFFFFF

def generate(seed: int, n: int, d: int) -> np.ndarray:
    """Fill (n, d) float64 matrix; each cell = lcg() / 2^32."""
    data = np.empty(n * d, dtype=np.float64)
    s = seed
    for i in range(n * d):
        s = lcg(s)
        data[i] = s / 4294967296.0
    return data.reshape(n, d)

# ──────────────────────────────────────── helpers
def hdr(title: str) -> None:
    w = 62
    print(f"\n{'─'*w}")
    print(f"  {title}")
    print(f"{'─'*w}")

# ══════════════════════════════════════════════════════════════════
hdr(f"LCG sequence  (seed={SEED}, first 10 raw uint32 states)")
s = SEED
lcg_seeds: list[int] = []
for i in range(10):
    s = lcg(s)
    lcg_seeds.append(s)
    print(f"  step {i:2d}: {s:>12d}")

# ──────────────────────────────────────── generate
hdr(f"Data generation  ({N} samples × {D} features)")
raw = generate(SEED, N, D)
print(f"  shape       : {raw.shape}")
print(f"  global mean : {raw.mean():.8f}  (expected ≈ 0.5)")
print(f"  global std  : {raw.std():.8f}  (expected ≈ 0.2887 for Uniform[0,1])")
print(f"  raw[0, :5]  : {raw[0, :5]}")
print(f"  raw[1, :5]  : {raw[1, :5]}")

# ──────────────────────────────────────── center
hdr("Centering")
col_means = raw.mean(axis=0)
X = raw - col_means
residuals = np.abs(X.mean(axis=0))
print(f"  col means  min={col_means.min():.6f}  max={col_means.max():.6f}")
print(f"  max residual mean after centering: {residuals.max():.2e}  (should be ≈ 0)")

# ──────────────────────────────────────── PCA fit
hdr(f"PCA fit  (K={K} components)")
pca = PCA(K)
pca.fit(X, batch_size=64)
ev = pca.explained_variance_values()
er = pca.explained_variance_ratio()
print(f"  eigenvalues : {np.array2string(ev, precision=6, floatmode='fixed')}")
print(f"  ratios      : {np.array2string(er, precision=6, floatmode='fixed')}")
print(f"  ratio sum   : {er.sum():.8f}")

# ──────────────────────────────────────── transform
hdr("Transform  (first 5 rows × K components)")
Xt = pca.transform(X[:5])
for i, row in enumerate(Xt):
    print(f"  row {i}: {np.array2string(row, precision=6, floatmode='fixed')}")

# ──────────────────────────────────────── covariance diagonal
hdr("Covariance matrix diagonal  (should be ≈ 1/12 ≈ 0.0833 for Uniform[0,1])")
cov_diag = np.array([(X[:, j] ** 2).sum() / (N - 1) for j in range(5)])
print(f"  cov[0..4, 0..4] diag: {cov_diag}")

# ──────────────────────────────────────── n_components clamping
hdr("Edge case: n_components > D")
pca_big = PCA(D + 50)
pca_big.fit(X)
print(f"  requested K={D+50}, fitted K={pca_big.components_.shape[1]}  (should equal D={D})")
assert pca_big.components_.shape[1] == D

# ──────────────────────────────────────── export
base = os.path.dirname(os.path.abspath(__file__))
hpp_path = os.path.join(base, "lcg_pca_params.hpp")
exp_path = os.path.join(base, "lcg_expected.txt")

pca.export_to_hpp(hpp_path, dtype="double", namespace="lcg_pca")

with open(exp_path, "w") as f:
    for v in lcg_seeds:          f.write(f"{v}\n")           # 10 uint32 LCG states
    for v in raw[0, :5]:         f.write(f"{v:.17g}\n")      # 5 raw[0] floats
    for v in raw[1, :5]:         f.write(f"{v:.17g}\n")      # 5 raw[1] floats
    for v in ev:                 f.write(f"{v:.17g}\n")      # K eigenvalues
    for row in Xt:
        for v in row:            f.write(f"{v:.17g}\n")      # 5*K transform values

hdr("Exports")
print(f"  {hpp_path}")
print(f"  {exp_path}")

print(f"\n{'═'*62}")
print("  Python audit COMPLETE")
print(f"{'═'*62}")
