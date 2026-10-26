#!/usr/bin/env bash
# Build the formalization and check that every audited theorem depends only on
# Lean's standard axioms (propext, Classical.choice, Quot.sound): no sorry, no new axioms.
set -euo pipefail
cd "$(dirname "$0")"

lean --version
lake exe cache get          # prebuilt Mathlib for the pinned version
lake build                  # fails on any error in the proofs
lake env lean Audit.lean > audit.out 2>&1

total=$(grep -c "depends on axioms" audit.out || true)
bad=$(awk '/depends on axioms/ {
          s = $0; sub(/.*\[/, "", s); sub(/\].*/, "", s); n = split(s, a, /, */)
          for (i = 1; i <= n; i++)
            if (a[i] != "propext" && a[i] != "Classical.choice" && a[i] != "Quot.sound") { print; break }
        }' audit.out)

if [ "$total" -eq 0 ] || grep -qi "error" audit.out || [ -n "$bad" ]; then
  echo "VERIFICATION FAILED"
  [ -n "$bad" ] && echo "$bad"
  grep -i "error" audit.out || true
  exit 1
fi
echo "OK: build succeeded; $total theorems audited, all use only propext, Classical.choice, Quot.sound."
