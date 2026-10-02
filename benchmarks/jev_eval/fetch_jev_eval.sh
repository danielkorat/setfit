#!/usr/bin/env sh
# Fetch onlyoneaman/jev-eval (test cases + Jev/LLM answers) at the commit these results were produced with.
set -eu
COMMIT=89d097c8b61b21cc4bab5ce2610c59ca8f5f45b3
cd "$(dirname "$0")"
[ -d jev-eval/.git ] || git clone --quiet https://github.com/onlyoneaman/jev-eval.git jev-eval
git -C jev-eval fetch --quiet origin "$COMMIT" 2>/dev/null || true
git -C jev-eval checkout --quiet "$COMMIT"
echo "jev-eval at $(git -C jev-eval rev-parse HEAD)"
