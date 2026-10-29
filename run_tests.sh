#!/bin/bash
set -e
cd "$(dirname "$0")"
python -m pytest tests/ -v --tb=short 2>/dev/null || python -m unittest discover -s tests -v
