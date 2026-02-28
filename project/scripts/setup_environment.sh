#!/usr/bin/env bash
set -euo pipefail

if command -v conda >/dev/null 2>&1; then
  conda env create -f environment.yml || conda env update -f environment.yml
  echo "Activate with: conda activate chicago-crime-simple"
else
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install pyspark==3.5.1 pandas==2.2.2 scikit-learn==1.5.1 pyarrow==17.0.0 pyyaml==6.0.2 pytest==8.3.3
  echo "Activate with: source .venv/bin/activate"
fi
