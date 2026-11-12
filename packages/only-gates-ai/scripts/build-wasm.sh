#!/usr/bin/env bash
# Thin wrapper kept for existing Unix habits and docs. The real implementation
# is cross-platform: ../../scripts/build-wasm.mjs.
set -euo pipefail

package_dir="$(cd "$(dirname "$0")/.." && pwd)"

exec node "$package_dir/../../scripts/build-wasm.mjs" \
  --build-dir dist-lib \
  --build-type "${BUILD_TYPE:-Release}" \
  --copy-wasm packages/only-gates-ai/wasm
