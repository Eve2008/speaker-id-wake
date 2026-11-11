#!/bin/bash
# Revision evaluation lane A (replaces lane 1 of run_eval_rev.sh, which was stopped during its
# campplus_zhen_adv_d1 step to evaluate the runtime-decomposition configurations first).
# campplus_zhen_adv_cip0 is evaluated on the main set first and re-run with --ablations once
# all of its ablation embeddings exist (same main results; eval_abl.done marks the second run).
set -e
cd /home/ali/Desktop/Articles/artifacts/speaker-id-wake
source experiments/run_eval_rev_lib.sh
run campplus_zhen_adv_cip0 main
run campplus_zhen_adv_np128 main
run campplus_zhen_adv_d1 main
run titanet_small main
run campplus_zhen_adv_cip0_d1 main
run titanet_small_d1 main
run campplus_zhen_adv__cw main
run titanet_small__cw main
run wespeaker_campplus_vox_wsfe__cw main
run wespeaker_campplus_vox main
if [ ! -f results/campplus_zhen_adv_cip0/eval_abl.done ]; then
  until ready campplus_zhen_adv_cip0 ablation 2>/dev/null; do sleep 60; done
  python experiments/03_evaluate.py --model campplus_zhen_adv_cip0 --ablations > logs/eval_rev_campplus_zhen_adv_cip0_abl.log 2>&1
  python experiments/19_paired.py --dump campplus_zhen_adv_cip0 --force > logs/paired_rev_campplus_zhen_adv_cip0_abl.log 2>&1
  date -Is > results/campplus_zhen_adv_cip0/eval_abl.done
  echo "evaluated campplus_zhen_adv_cip0 with ablations $(date -Is)"
fi
echo "lane A all done $(date -Is)" > logs/eval_rev_laneA.done
