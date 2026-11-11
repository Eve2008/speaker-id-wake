#!/bin/bash
# Second resume of the main-set extraction (the machine rebooted during
# run_extract_rest2.sh, with eres2net_base_zh at 18,800 of 83,924 specs).
# 02_extract.py is resumable; partial rows were checked (no mixed-NaN rows,
# all norms 1.0) before resuming. Fastest models first, so their evaluation
# can start while ERes2Net is still running.
set -e
source /home/ali/Desktop/Articles/.venv/bin/activate
cd /home/ali/Desktop/Articles/artifacts/speaker-id-wake
export OMP_NUM_THREADS=1
python experiments/02_extract.py --set main \
  --models wespeaker_resnet34_vox,wespeaker_resnet34_cnc,eres2net_base_zh \
  --workers 7 --chunk 300 > logs/extract_main_rest3.log 2>&1
echo done > logs/extract_main_rest3.done
