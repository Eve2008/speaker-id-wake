#!/bin/bash
# Third resume of the main-set extraction (the session that ran
# run_extract_rest3.sh ended at 18:19 and its workers were killed). At restart:
# wespeaker_resnet34_vox 5,000 of 83,924 main specs done, eres2net_base_zh
# 18,800, wespeaker_resnet34_cnc 800; no mixed-NaN rows, all norms 1.0.
# 02_extract.py is resumable, so this only computes what is missing.
set -e
source /home/ali/Desktop/Articles/.venv/bin/activate
cd /home/ali/Desktop/Articles/artifacts/speaker-id-wake
export OMP_NUM_THREADS=1
python experiments/02_extract.py --set main \
  --models wespeaker_resnet34_vox,wespeaker_resnet34_cnc,eres2net_base_zh \
  --workers 7 --chunk 300 > logs/extract_main_rest4.log 2>&1
echo done > logs/extract_main_rest4.done
