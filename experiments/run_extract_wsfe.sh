#!/bin/bash
# Main-set extraction for the three WeSpeaker models with WeSpeaker's own
# front end + onnxruntime (backend "wespeaker_ort", swk/frontend.py), after the
# subset checks in results/wespeaker_ort_check/ (as-shipped sherpa-onnx path:
# near-chance EER; WeSpeaker front end: EER 8-10% at 1 s on the subset).
# 02_extract.py is resumable and flushes after every chunk.
set -e
source /home/ali/Desktop/Articles/.venv/bin/activate
cd /home/ali/Desktop/Articles/artifacts/speaker-id-wake
export OMP_NUM_THREADS=1
python experiments/02_extract.py --set main \
  --models wespeaker_resnet34_vox_wsfe,wespeaker_campplus_vox_wsfe,wespeaker_resnet34_cnc_wsfe \
  --workers 6 --chunk 300 > logs/extract_main_wsfe.log 2>&1
echo done > logs/extract_main_wsfe.done
