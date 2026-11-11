#!/bin/bash
# Revision queue B (3 workers): NumPy front end + onnxruntime 1.28.2 (ctypes) for NexaRoom's
# CAM++ (campplus_zhen_adv_np128), the matched-condition runs of CAM++ (patched, noise floor)
# and TitaNet-small (noise floor), then all ablation specs for campplus_zhen_adv_cip0
# (waits until queue A has finished that model's main set: same embedding file).
set -e
source /home/ali/Desktop/Articles/.venv/bin/activate
cd /home/ali/Desktop/Articles/artifacts/speaker-id-wake
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
step() {
  local n="$1"; shift
  if [ -f "logs/extract_rev_$n.done" ]; then echo "skip $n"; return; fi
  python experiments/02_extract.py "$@" > "logs/extract_rev_$n.log" 2>&1
  date -Is > "logs/extract_rev_$n.done"
}
step np128 --set main --models campplus_zhen_adv_np128 --workers 3 --chunk 300
step cip0_d1 --set main --models campplus_zhen_adv_cip0_d1 --workers 3 --chunk 300
step titanet_d1 --set main --models titanet_small_d1 --workers 3 --chunk 300
until [ -f logs/extract_rev_cip0.done ]; do sleep 60; done
step cip0_abl --set ablation --models campplus_zhen_adv_cip0 --workers 3 --chunk 300
echo done > logs/extract_rev_b.alldone
