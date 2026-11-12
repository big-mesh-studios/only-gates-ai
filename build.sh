#!/usr/bin/env bash
# Thin wrapper kept for existing Unix habits and docs. The real implementation
# is cross-platform: scripts/build-wasm.mjs.
set -euo pipefail

cd "$(dirname "$0")"

exec node scripts/build-wasm.mjs \
  --build-dir "${BUILD_DIR:-dist}" \
  --build-type "${BUILD_TYPE:-Debug}"
