#!/bin/bash
# NexaRoom's default model with sherpa-onnx's front-end conventions run by
# onnxruntime 1.31.1 (key campplus_zhen_adv_ort131), after the subset check in
# results/runtime_effect_subset/ (same features: EER 7.1% with 1.31.1 vs 7.8%
# with 1.27.0). Waits for run_extract_d1.sh to finish, then uses its 2 workers.
set -e
source /home/ali/Desktop/Articles/.venv/bin/activate
cd /home/ali/Desktop/Articles/artifacts/speaker-id-wake
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
until [ -f results/campplus_zhen_adv_d1/eval.done ]; do sleep 60; done
python experiments/02_extract.py --set main --models campplus_zhen_adv_ort131 \
  --workers 2 --chunk 300 > logs/extract_main_ort131.log 2>&1
python experiments/03_evaluate.py --model campplus_zhen_adv_ort131 > logs/eval_campplus_zhen_adv_ort131.log 2>&1
date -Is > results/campplus_zhen_adv_ort131/eval.done
