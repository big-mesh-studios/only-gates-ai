#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

emcmake cmake -S . -B dist -DCMAKE_BUILD_TYPE=Release
cmake --build dist --parallel
