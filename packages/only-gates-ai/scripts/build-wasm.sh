#!/usr/bin/env bash
set -euo pipefail

package_dir="$(cd "$(dirname "$0")/.." && pwd)"
repo_root="$(cd "$package_dir/../.." && pwd)"
build_dir="$repo_root/dist-lib"
wasm_dir="$package_dir/wasm"

BUILD_DIR="$build_dir" BUILD_TYPE="${BUILD_TYPE:-Release}" bash "$repo_root/build.sh"

mkdir -p "$wasm_dir"
cp "$build_dir/only-gates-ai.js" "$wasm_dir/only-gates-ai.js"
cp "$build_dir/only-gates-ai.wasm" "$wasm_dir/only-gates-ai.wasm"
