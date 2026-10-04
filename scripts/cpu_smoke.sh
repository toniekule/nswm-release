#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

if [ -n "${NSWM_PYTHON:-}" ]; then
    PYTHON="$NSWM_PYTHON"
elif [ -x "$PWD/.venv/bin/python" ]; then
    PYTHON="$PWD/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "no Python interpreter found; set NSWM_PYTHON" >&2
    exit 1
fi

if ! "$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "Python 3.10 or newer is required; $PYTHON is $("$PYTHON" -V 2>&1)." >&2
    echo "Point NSWM_PYTHON at a suitable interpreter or create .venv first." >&2
    exit 1
fi

"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" -m nswm.cli plan-demo --policy certificate --budget 10
