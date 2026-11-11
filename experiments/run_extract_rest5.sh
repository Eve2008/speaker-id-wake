#!/bin/bash
# Fourth resume of the main-set extraction (the machine rebooted at about
# 19:08 while run_extract_rest4.sh was running). At restart:
# wespeaker_resnet34_vox 13,400 of 83,924 main specs done, eres2net_base_zh
# 18,800, wespeaker_resnet34_cnc 800; no mixed-NaN rows, no zero rows, all
# norms 1.0. 02_extract.py now flushes after every chunk. It is resumable, so
# this only computes what is missing. Started detached (setsid) so that it
# survives the end of the agent session.
set -e
source /home/ali/Desktop/Articles/.venv/bin/activate
cd /home/ali/Desktop/Articles/artifacts/speaker-id-wake
export OMP_NUM_THREADS=1
python experiments/02_extract.py --set main \
  --models wespeaker_resnet34_vox,wespeaker_resnet34_cnc,eres2net_base_zh \
  --workers 7 --chunk 300 > logs/extract_main_rest5.log 2>&1
echo done > logs/extract_main_rest5.done
