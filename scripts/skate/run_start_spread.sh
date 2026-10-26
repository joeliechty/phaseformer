#!/bin/bash
# Start-spread experiment: the four MLP arms over 10 seeds on every start-spread set (random and
# sorted sides, and the mixtures' noisy demos alone), one run after another, then the
# evaluation, which also re-evaluates the existing point-to-point sets and C. The data comes
# from build_start_spread.py. It waits for any Push-T training on the GPU to finish first
# (the 4090 is compute-bound). Runs already done are skipped, so it resumes.
#   scripts/skate/run_start_spread.sh > runs/skate_deterministic/start_spread.log 2>&1
set -e
cd "$(dirname "$0")/../.."
P=${PYTHON:-python}  # the fm_bc env's interpreter
D=data/skate_deterministic
[ -f $D/test/dual_path_mixonly15.npz ] || $P scripts/skate/build_start_spread.py
while pgrep -f train_pusht_policy.py > /dev/null; do
  echo "waiting for Push-T training to finish ($(date +%H:%M))"
  sleep 300
done
SCENS="dual_path_spread02 dual_path_spread04 dual_path_spread08 dual_path_spread12 dual_path_sorted02 dual_path_sorted08 dual_path_mixonly05 dual_path_mixonly10 dual_path_mixonly15"
C="scripts/skate/run_experiments.py --scenarios $SCENS --n-train 1024 --seeds 0 1 2 3 4 5 6 7 8 9 --data $D --test $D/test --out runs/skate_deterministic --device cuda"
$P $C --cdm-weights 0
$P $C --cdm-weights 0 --no-phase
$P $C --cdm-weights 0.5 --cdm-terms score
$P $C --cdm-weights 0.5 --cdm-terms score --no-phase
$P scripts/skate/start_spread_eval.py
echo ALL_DONE
