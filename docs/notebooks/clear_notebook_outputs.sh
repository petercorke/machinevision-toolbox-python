#!/usr/bin/env bash
set -euo pipefail

ROOT="$1"
find "$ROOT" -type f -name "*.ipynb" \
  ! -path "*/old/*" \
  -exec jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {} +
