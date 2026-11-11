#!/bin/bash
# Main-set extraction for NexaRoom's default model with the same 1-LSB noise
# floor that the WeSpeaker front end uses (key campplus_zhen_adv_d1), so the
# comparison with the *_wsfe models is like for like. Runs next to
# run_extract_wsfe.sh with 2 workers (6 + 2 = 8 processes), then evaluates.
set -e
source /home/ali/Desktop/Articles/.venv/bin/activate
cd /home/ali/Desktop/Articles/artifacts/speaker-id-wake
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python experiments/02_extract.py --set main --models campplus_zhen_adv_d1 \
  --workers 2 --chunk 300 > logs/extract_main_d1.log 2>&1
python experiments/03_evaluate.py --model campplus_zhen_adv_d1 > logs/eval_campplus_zhen_adv_d1.log 2>&1
date -Is > results/campplus_zhen_adv_d1/eval.done
