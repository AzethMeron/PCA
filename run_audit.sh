#!/usr/bin/env bash
# Deep audit: LCG data generator + PCA — Python vs C++ cross-check
set -euo pipefail
cd "$(dirname "$0")"

SEP="══════════════════════════════════════════════════════════════"

echo "$SEP"
echo "  PCA Audit: Python + C++ cross-check"
echo "$SEP"

# ── step 1: Python audit + reference export ──────────────────────
echo ""
echo "▶ Step 1 — Python audit (generates lcg_expected.txt + lcg_pca_params.hpp)"
echo ""
python3 test_audit.py

# ── step 2: compile C++ ──────────────────────────────────────────
echo ""
echo "▶ Step 2 — Compile C++ audit binary"
echo ""
g++ -std=c++17 -O2 -I. -o /tmp/pca_audit test_audit.cpp
echo "  Compiled OK → /tmp/pca_audit"

# ── step 3: run C++ audit ────────────────────────────────────────
echo ""
echo "▶ Step 3 — C++ audit"
echo ""
/tmp/pca_audit
