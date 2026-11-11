#!/bin/bash
# Revision queue A (4 workers): NexaRoom's CAM++ file with the AveragePool patch through
# sherpa-onnx (campplus_zhen_adv_cip0), then ERes2Net-base with the 1-LSB noise floor
# (matched condition), then the trimming ablation for ERes2Net-base.
# Each step is resumable (02_extract.py skips embedded rows); a .done file marks each step.
set -e
source /home/ali/Desktop/Articles/.venv/bin/activate
cd /home/ali/Desktop/Articles/artifacts/speaker-id-wake
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
step() {  # name, then 02_extract.py arguments
  local n="$1"; shift
  if [ -f "logs/extract_rev_$n.done" ]; then echo "skip $n"; return; fi
  python experiments/02_extract.py "$@" > "logs/extract_rev_$n.log" 2>&1
  date -Is > "logs/extract_rev_$n.done"
}
step cip0 --set main --models campplus_zhen_adv_cip0 --workers 4 --chunk 300
step eres2net_d1 --set main --models eres2net_base_zh_d1 --workers 4 --chunk 300
step eres2net_trim --set ablation --trim-only --models eres2net_base_zh --workers 4 --chunk 300
echo done > logs/extract_rev_a.alldone
