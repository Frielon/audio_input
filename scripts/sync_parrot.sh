#!/usr/bin/env bash
# Copy Parrot Shadowing's sound analysis into audio_input/parrot_backend again, after a change there:
#   scripts/sync_parrot.sh [path/to/parrot_shadowing]      (default: ../parrot_shadowing)
# The bundled modules are copied, plus any module of its backend they now import; then run the tests.
set -euo pipefail
cd "$(dirname "$0")/.."

src="${1:-../parrot_shadowing}/backend"
dst=audio_input/parrot_backend
[[ -f "$src/analysis.py" ]] || { echo "Parrot Shadowing's backend not found at $src" >&2; exit 1; }

bundled() {  # the modules in $dst (its __init__.py is audio_input's own) and the sibling modules they import
  ls "$dst" | sed -n 's/\.py$//p' | grep -vx __init__
  sed -nE 's/^[[:space:]]*from \.([a-z_]+) import .*/\1/p; s/^[[:space:]]*from \. import ([a-z_, ]+).*/\1/p' "$dst"/*.py |
    tr ',' '\n' | sed -E 's/ as .*//; s/[[:space:]]//g'
}

changed=1
while (( changed )); do  # until a copied module imports nothing new
  changed=0
  for mod in $(bundled | sort -u); do
    if ! cmp -s "$src/$mod.py" "$dst/$mod.py"; then
      cp "$src/$mod.py" "$dst/$mod.py"
      echo "updated $dst/$mod.py"
      changed=1
    fi
  done
done
echo "$dst is up to date with $src."
