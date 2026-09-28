#!/usr/bin/env bash
# Run pydisort's focused CUDA check, full Python suite, and CTest suite.
#
# Usage:
#   ./run_pytest.sh          # execute all validation phases
#   ./run_pytest.sh --list   # print the collected tests without running them
#
# The script never activates an environment. It invokes this checkout's
# dedicated development interpreter directly.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$ROOT/.venv}"
PYTHON="$VENV/bin/python"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"

if [[ ! -x "$PYTHON" ]]; then
  printf 'Missing pydisort interpreter: %s\n' "$PYTHON" >&2
  printf 'Create it with: python -m venv %s\n' "$VENV" >&2
  exit 1
fi

if [[ ! -d "$BUILD_DIR" ]]; then
  printf 'Missing CMake build directory: %s\n' "$BUILD_DIR" >&2
  printf 'Configure it before running CTest.\n' >&2
  exit 1
fi

focused=(tests/cuda/test_cuda_backend_selection.py -v -rs)
full=(tests/ -v -rs)

if [[ "${1:-}" == "--list" ]]; then
  if [[ "$#" -ne 1 ]]; then
    printf 'Usage: %s [--list]\n' "$0" >&2
    exit 2
  fi

  printf 'Focused CUDA pytest command:\n  %q' "$PYTHON"
  printf ' -m pytest'
  printf ' %q' "${focused[@]}"
  printf '\n\nFocused CUDA tests:\n'
  "$PYTHON" -m pytest --collect-only -q "${focused[0]}"

  printf '\nFull pytest command:\n  %q' "$PYTHON"
  printf ' -m pytest'
  printf ' %q' "${full[@]}"
  printf '\n\nFull pytest tests:\n'
  "$PYTHON" -m pytest --collect-only -q "${full[0]}"

  printf '\nCTest command:\n  ctest --test-dir %q --output-on-failure\n' "$BUILD_DIR"
  printf '\nCTest tests:\n'
  ctest --test-dir "$BUILD_DIR" -N
  exit 0
fi

if [[ "$#" -ne 0 ]]; then
  printf 'Usage: %s [--list]\n' "$0" >&2
  exit 2
fi

printf '== Focused CUDA backend selection ==\n'
"$PYTHON" -m pytest "${focused[@]}"

printf '\n== Full Python suite ==\n'
"$PYTHON" -m pytest "${full[@]}"

printf '\n== CTest suite ==\n'
ctest --test-dir "$BUILD_DIR" --output-on-failure
