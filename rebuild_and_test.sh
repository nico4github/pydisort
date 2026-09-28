#!/usr/bin/env bash
# Rebuild the H100 pydisort extension and run every configured solver test.
# Detailed output from individual tools is written to temporary logs. The
# script announces every phase and its elapsed time; on failure it also prints
# the failing command log.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$ROOT/.venv}"
PYTHON="$VENV/bin/python"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
LOG_DIR="${TMPDIR:-/tmp}/pydisort-rebuild-$$"

if [[ ! -x "$PYTHON" ]]; then
  printf 'Missing pydisort interpreter: %s\n' "$PYTHON" >&2
  exit 1
fi
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  printf 'Missing CUDA compiler: %s/bin/nvcc\n' "$CUDA_HOME" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
cleanup() {
  rm -rf "$LOG_DIR"
}
trap cleanup EXIT
run() {
  local name="$1"
  shift
  local log="$LOG_DIR/$name.log"
  local started=$SECONDS
  printf '[%(%F %T)T] START %s\n' -1 "$name"
  if ! "$@" >"$log" 2>&1; then
    printf '[%(%F %T)T] FAILED %s after %ss\n\n' -1 "$name" "$((SECONDS - started))" >&2
    cat "$log" >&2
    exit 1
  fi
  printf '[%(%F %T)T] DONE  %s (%ss)\n' -1 "$name" "$((SECONDS - started))"
}

run configure "$VENV/bin/cmake" -S "$ROOT" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=ON \
  -DCUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DPython3_EXECUTABLE="$PYTHON"
run build "$VENV/bin/cmake" --build "$BUILD_DIR" --parallel
run install env PATH="$VENV/bin:$CUDA_HOME/bin:$PATH" \
  WORKSPACE="$ROOT" TORCH_CUDA_ARCH_LIST=9.0 \
  "$PYTHON" -m pip install --force-reinstall --no-build-isolation --no-deps -v "$ROOT"
run validate env PATH="$VENV/bin:$CUDA_HOME/bin:$PATH" \
  VENV="$VENV" BUILD_DIR="$BUILD_DIR" "$ROOT/run_pytest.sh"
run precommit "$VENV/bin/pre-commit" run --all-files
run dependencies "$PYTHON" -m pip check

printf 'PASS: rebuilt pydisort; focused CUDA, pytest, CTest, pre-commit, and pip checks passed.\n'
