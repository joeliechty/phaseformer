#!/bin/bash
# The point-to-point dual-path side experiments (EXTRA_SCENARIOS): data, then the four MLP
# arms at n=1024 over 10 seeds, one after another (the 4090 is compute-bound), then the
# analysis. Runs already done are skipped, so it resumes.
#   scripts/skate/run_deterministic_grid.sh > runs/skate_deterministic/grid.log 2>&1
#   scripts/skate/run_deterministic_grid.sh dual_path_fixed_via   # one scenario only
set -e
cd "$(dirname "$0")/../.."
P=${PYTHON:-python}  # the fm_bc env's interpreter
D=data/skate_deterministic
SCENS=${@:-dual_path_fixed dual_path_fixed_via dual_path_noisy_start}
for S in $SCENS; do
  if [ ! -f $D/$S.npz ]; then
    $P scripts/skate/generate_datasets.py --scenario $S --seed 0 --out $D
    $P scripts/skate/generate_datasets.py --scenario $S --seed 1 --out $D/test
    $P scripts/skate/compute_divergence.py --data $D --scenario $S
    $P scripts/skate/compute_divergence.py --data $D/test --scenario $S
  fi
done
C="scripts/skate/run_experiments.py --scenarios $SCENS --n-train 1024 --seeds 0 1 2 3 4 5 6 7 8 9 --data $D --test $D/test --out runs/skate_deterministic --device cuda"
$P $C --cdm-weights 0
$P $C --cdm-weights 0 --no-phase
$P $C --cdm-weights 0.5 --cdm-terms score
$P $C --cdm-weights 0.5 --cdm-terms score --no-phase
$P scripts/skate/cross_start_eval.py
$P scripts/skate/plot_p2p_bars.py
$P scripts/skate/evaluate_fixed_start.py
echo ALL_DONE
