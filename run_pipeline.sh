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

echo "--- 2/7  scrape_tournaments.py ---"
${PYTHON} src/scrape_tournaments.py

echo ""
echo "--- 3/7  parse.py ---"
${PYTHON} src/parse.py

echo ""
echo "--- 4/7  validate.py ---"
${PYTHON} src/validate.py

echo ""
echo "--- 5/7  build_features.py ---"
${PYTHON} src/build_features.py

echo ""
echo "--- 6/7  label.py ---"
${PYTHON} src/label.py

echo ""
echo "--- 7/7  train.py ---"
${PYTHON} src/train.py

echo ""
echo "=== Pipeline complete ==="
