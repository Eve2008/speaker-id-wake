#!/bin/bash
# Revision evaluation lanes F1/F2 (11 Nov, after the reboot at 08:10 stopped lanes A, E and 3).
# Same run() as run_eval_rev_lib.sh (waits for complete embeddings; results/<cfg>/eval_rev.done marks
# a finished configuration), but one BLAS thread per lane, because the matched-condition extraction
# (run_extract_rev_c.sh, 6 workers) runs at the same time. Usage: run_eval_rev_f.sh 1|2
set -e
cd /home/ali/Desktop/Articles/artifacts/speaker-id-wake
source experiments/run_eval_rev_lib.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ "$1" = 1 ]; then
  run campplus_zhen_adv_np128 main
  run eres2net_base_zh main
  run campplus_zhen_adv_d1 main
  run wespeaker_resnet34_vox_wsfe__cw main
  run campplus_zhen_adv__cw main
  run campplus_zhen_adv_cip0_d1 main
  run titanet_small_d1 main
else
  run titanet_small main
  run wespeaker_campplus_vox main
  run wespeaker_resnet34_cnc_wsfe__cw main
  run titanet_small__cw main
  run wespeaker_campplus_vox_wsfe__cw main
  run eres2net_base_zh_d1 main
fi
echo "lane F$1 all done $(date -Is)" > "logs/eval_rev_laneF$1.done"
