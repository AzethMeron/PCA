#!/usr/bin/env bash
# Dimensionality reduction cross-check: Python PCA(K=50) vs C++ (loaded params)
set -euo pipefail
cd "$(dirname "$0")"

SEP="══════════════════════════════════════════════════════════════"
echo "$SEP"
echo "  Dimensionality Reduction: 100 features → 50 components"
echo "  LCG dataset  N=3000 D=100  |  transform first 100 samples"
echo "$SEP"

echo ""
echo "▶ Step 1 — Python: fit PCA(K=50), print explained variance + transform"
echo ""
python3 compare_transform.py

echo ""
echo "▶ Step 2 — Compile C++ compare binary"
echo ""
g++ -std=c++17 -O2 -I. -o /tmp/compare_transform compare_transform.cpp
echo "  Compiled OK → /tmp/compare_transform"

echo ""
echo "▶ Step 3 — C++: print explained variance + transform, then compare"
echo ""
/tmp/compare_transform
