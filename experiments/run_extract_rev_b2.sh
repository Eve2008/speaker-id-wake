#!/bin/bash
# Revision queue B, second version (3 workers). Replaces run_extract_rev_b.sh after its first
# step had started, to add two 3D-Speaker VoxCeleb models (matched condition) and to run the
# trimming specs of campplus_zhen_adv_cip0 before its other ablation specs. Waits for the
# still-running np128 extraction of the first queue, then re-runs that step (nothing left to do).
set -e
source /home/ali/Desktop/Articles/.venv/bin/activate
cd /home/ali/Desktop/Articles/artifacts/speaker-id-wake
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
while pgrep -f "02_extract.py --set main --models campplus_zhen_adv_np128" > /dev/null; do sleep 30; done
step() {
  local n="$1"; shift
  if [ -f "logs/extract_rev_$n.done" ]; then echo "skip $n"; return; fi
  python experiments/02_extract.py "$@" > "logs/extract_rev_$n.log.part" 2>&1
  cat "logs/extract_rev_$n.log.part" >> "logs/extract_rev_$n.log"; rm -f "logs/extract_rev_$n.log.part"
  date -Is > "logs/extract_rev_$n.done"
}
step np128 --set main --models campplus_zhen_adv_np128 --workers 3 --chunk 300
step cip0_d1 --set main --models campplus_zhen_adv_cip0_d1 --workers 3 --chunk 300
step titanet_d1 --set main --models titanet_small_d1 --workers 3 --chunk 300
step campplus_vox_d1 --set main --models campplus_vox_cip0_d1 --workers 3 --chunk 300
until [ -f logs/extract_rev_cip0.done ]; do sleep 60; done
step cip0_trim --set ablation --trim-only --models campplus_zhen_adv_cip0 --workers 3 --chunk 300
step eres2net_vox_d1 --set main --models eres2net_vox_d1 --workers 3 --chunk 300
step cip0_abl --set ablation --models campplus_zhen_adv_cip0 --workers 3 --chunk 300
echo done > logs/extract_rev_b.alldone
