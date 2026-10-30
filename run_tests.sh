#!/bin/bash
set -e
cd "$(dirname "$0")"
PY="$(command -v python3 || command -v python)"
if [ -z "$PY" ]; then echo "python3/python not found" >&2; exit 1; fi
if "$PY" -c "import pytest" 2>/dev/null; then
  "$PY" -m pytest tests/ -v --tb=short
else
  echo "pytest not installed (pip install -r requirements.txt); falling back to unittest" >&2
  "$PY" -m unittest discover -s tests -v
fi
