#!/usr/bin/env bash
# run_pipeline.sh — run full pipeline after scrape_discovery.py has completed
set -e

PYTHON=".venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    PYTHON="python3"
fi

echo "=== Chess Predictor Pipeline ==="
echo "Python: $(${PYTHON} --version)"
echo ""

echo "--- 1/6  scrape_tournaments.py ---"
${PYTHON} src/scrape_tournaments.py

echo ""
echo "--- 2/6  parse.py ---"
${PYTHON} src/parse.py

echo ""
echo "--- 3/6  validate.py ---"
${PYTHON} src/validate.py

echo ""
echo "--- 4/6  build_features.py ---"
${PYTHON} src/build_features.py

echo ""
echo "--- 5/6  label.py ---"
${PYTHON} src/label.py

echo ""
echo "--- 6/6  train.py ---"
${PYTHON} src/train.py

echo ""
echo "=== Pipeline complete ==="
