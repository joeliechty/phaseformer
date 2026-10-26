#!/bin/bash
# Start-noise sweep: the four MLP arms at n=1024 over 10 seeds on each start-noise mixture
# (dual_path_mix05, mix10, mix15), one run after another, then the sweep's evaluation. The mixtures
# are built from dual_path_fixed and dual_path_noisy_start (run_deterministic_grid.sh makes
# those). Runs already done are skipped, so it resumes.
#   scripts/skate/run_start_mix_sweep.sh > runs/skate_deterministic/mix_sweep.log 2>&1
set -e
cd "$(dirname "$0")/../.."
P=${PYTHON:-python}  # the fm_bc env's interpreter
D=data/skate_deterministic
[ -f $D/dual_path_mix15.npz ] || $P scripts/skate/build_start_mixes.py
SCENS="dual_path_mix05 dual_path_mix10 dual_path_mix15"
C="scripts/skate/run_experiments.py --scenarios $SCENS --n-train 1024 --seeds 0 1 2 3 4 5 6 7 8 9 --data $D --test $D/test --out runs/skate_deterministic --device cuda"
$P $C --cdm-weights 0
$P $C --cdm-weights 0 --no-phase
$P $C --cdm-weights 0.5 --cdm-terms score
$P $C --cdm-weights 0.5 --cdm-terms score --no-phase
$P scripts/skate/start_mix_eval.py
$P scripts/skate/plot_p2p_combined.py
echo ALL_DONE
