#!/usr/bin/env bash
# Fetch the CatBoost reference project (Railway_hep) at the commit this benchmark was built against.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=https://github.com/JungWonSeok97/Railway_hep.git
COMMIT=f864cce
if [ ! -d vendor/Railway_hep/.git ]; then
  git clone -q "$REPO" vendor/Railway_hep
fi
git -C vendor/Railway_hep fetch -q origin
git -C vendor/Railway_hep checkout -q "$COMMIT"
echo "vendor/Railway_hep at $(git -C vendor/Railway_hep rev-parse --short HEAD)"
