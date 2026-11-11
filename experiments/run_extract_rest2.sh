#!/bin/bash
# Resume of the main-set extraction for the last three models (the first queue
# run was interrupted after wespeaker_campplus_vox). 02_extract.py is resumable.
set -e
source /home/ali/Desktop/Articles/.venv/bin/activate
cd /home/ali/Desktop/Articles/artifacts/speaker-id-wake
export OMP_NUM_THREADS=1
python experiments/02_extract.py --set main \
  --models eres2net_base_zh,wespeaker_resnet34_vox,wespeaker_resnet34_cnc \
  --workers 8 --chunk 300 > logs/extract_main_rest2.log 2>&1
