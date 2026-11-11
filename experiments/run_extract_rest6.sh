#!/bin/bash
# Main-set extraction after the WeSpeaker CMN finding (19:30). The as-shipped
# WeSpeaker ResNet34 runs were stopped (vox: about 20,000 of 83,924 specs,
# cnc: 800; kept in cache/emb but not evaluated). Queue: ERes2Net (as shipped;
# its metadata already asks sherpa-onnx for CMN), then the three CMN-flagged
# WeSpeaker copies made by 08_make_cmn_models.py. 02_extract.py is resumable
# and flushes after every chunk.
set -e
source /home/ali/Desktop/Articles/.venv/bin/activate
cd /home/ali/Desktop/Articles/artifacts/speaker-id-wake
export OMP_NUM_THREADS=1
python experiments/02_extract.py --set main --models eres2net_base_zh \
  --workers 7 --chunk 300 > logs/extract_main_rest6_eres2net.log 2>&1
python experiments/02_extract.py --set main \
  --models wespeaker_resnet34_vox_cmn,wespeaker_campplus_vox_cmn,wespeaker_resnet34_cnc_cmn \
  --workers 7 --chunk 300 > logs/extract_main_rest6_cmn.log 2>&1
echo done > logs/extract_main_rest6.done
