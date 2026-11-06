#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Debug by default so DWARF is usable in Chrome. Use -O1/-Og for a faster build
# if needed, but avoid -O3 while debugging: heavy optimization degrades DWARF.
BUILD_TYPE="${BUILD_TYPE:-Debug}"

emcmake cmake -S . -B dist -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
cmake --build dist --parallel
