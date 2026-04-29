#!/usr/bin/env python3
"""
Dimensionality reduction: 3000 samples × 100 features → 50 components.
Prints explained variance for every component, transforms first 100 samples,
exports reference data for C++ cross-check.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pca import PCA

SEED, N, D, K, N_TX = 143, 3000, 100, 50, 100

def lcg(s: int) -> int:
    return (1664525 * s + 1013904223) & 0xFFFFFFFF

def generate(seed: int, n: int, d: int) -> np.ndarray:
    data = np.empty(n * d, dtype=np.float64)
    s = seed
    for i in range(n * d):
        s = lcg(s)
        data[i] = s / 4294967296.0
    return data.reshape(n, d)

# ── generate + center ─────────────────────────────────────────────
raw = generate(SEED, N, D)
X   = raw - raw.mean(axis=0)

# ── fit PCA(K=50) ─────────────────────────────────────────────────
pca = PCA(K)
pca.fit(X, batch_size=64)

ev      = pca.explained_variance_values()   # (50,) raw eigenvalues
er      = pca.explained_variance_ratio()    # (50,) ratios
cum_er  = np.cumsum(er)

print(f"=== Python PCA  (K={K}, fitted on {N}×{D})  total_variance={pca.total_variance_:.6f} ===")
print()
print(f"  {'dim':>4}  {'eigenvalue':>12}  {'ratio':>10}  {'cumulative':>10}")
print(f"  {'----':>4}  {'----------':>12}  {'-----':>10}  {'----------':>10}")
for k in range(K):
    print(f"  {k:4d}  {ev[k]:12.8f}  {er[k]:10.8f}  {cum_er[k]:10.8f}")
print()
print(f"  Variance retained by {K} components: {cum_er[-1]*100:.4f}%")
print()

# ── transform first 100 samples ───────────────────────────────────
Xt = pca.transform(X[:N_TX])   # (100, 50)

print(f"=== Python transform  ({N_TX} samples × {K} components) ===")
for i, row in enumerate(Xt):
    vals = " ".join(f"{v:+.8f}" for v in row)
    print(f"[{i:3d}] {vals}")

# ── export for C++ ────────────────────────────────────────────────
base     = os.path.dirname(os.path.abspath(__file__))
hpp_path = os.path.join(base, "lcg_pca50_params.hpp")
ref_path = os.path.join(base, "lcg_tx100_ref.txt")

pca.export_to_hpp(hpp_path, dtype="double", namespace="lcg_pca50")

with open(ref_path, "w") as f:
    for row in Xt:
        f.write(" ".join(f"{v:.17g}" for v in row) + "\n")

print()
print(f"[exported] {hpp_path}")
print(f"[exported] {ref_path}")
