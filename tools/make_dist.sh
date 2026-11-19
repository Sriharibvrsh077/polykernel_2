#!/usr/bin/env bash
# Package PolyKernel sources (without built runtime libraries or caches).
set -euo pipefail
PK=$(cd "$(dirname "$0")/.." && pwd)
VER=$(grep -m1 "^VERSION" "$PK/bin/pkcc" | sed "s/.*'\(.*\)'.*/\1/")
OUT=${1:-$PK/../polykernel-$VER.tar.gz}
tar -C "$(dirname "$PK")" --exclude='polykernel/lib' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='polykernel/.git' -czf "$OUT" "$(basename "$PK")"
echo "$OUT"
