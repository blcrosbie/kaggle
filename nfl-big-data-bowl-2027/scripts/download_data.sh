#!/usr/bin/env bash
# Download NFL Big Data Bowl 2027 data into ../data (gitignored).
# Prereqs: `pip install kaggle`, credentials in ~/.kaggle/kaggle.json (or KAGGLE_API_TOKEN env var),
# and the competition rules ACCEPTED on the website (otherwise the API returns 403).
set -euo pipefail
cd "$(dirname "$0")/.."
kaggle competitions download -c nfl-big-data-bowl-2027 -p data/
cd data && unzip -o nfl-big-data-bowl-2027.zip && ls -lh
