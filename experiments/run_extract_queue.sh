#!/bin/bash
# Extraction queue: ablation set for the NexaRoom default model, then the main
# set for the other five models. 02_extract.py is resumable, so re-running this
# script only computes what is missing.
set -e
source /home/ali/Desktop/Articles/.venv/bin/activate
cd /home/ali/Desktop/Articles/artifacts/speaker-id-wake
export OMP_NUM_THREADS=1
python experiments/02_extract.py --set ablation --models campplus_zhen_adv --workers 8 --chunk 300 \
  > logs/extract_ablation_campplus.log 2>&1
python experiments/02_extract.py --set main \
  --models titanet_small,wespeaker_campplus_vox,eres2net_base_zh,wespeaker_resnet34_vox,wespeaker_resnet34_cnc \
  --workers 8 --chunk 300 > logs/extract_main_rest.log 2>&1
