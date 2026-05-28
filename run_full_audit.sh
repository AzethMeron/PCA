#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "========================================"
echo "  Full PCA Audit"
echo "========================================"

echo ""
echo "--- Python audit (vs sklearn) ---"
python3 test_full_audit.py

echo ""
echo "--- Compiling C++ audit ---"
g++ -std=c++17 -O2 -Wall -Wextra -o test_full_audit test_full_audit.cpp
echo "  Compiled OK"

echo ""
echo "--- Running C++ audit ---"
./test_full_audit

echo ""
echo "========================================"
echo "  Full audit complete"
echo "========================================"
