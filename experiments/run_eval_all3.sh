#!/bin/bash
# Evaluation queue for the WeSpeaker-front-end models (run_extract_wsfe.sh) (same
# logic as run_eval_all.sh: waits until a model's main-set embeddings are
# complete, skips a model whose results/<model>/eval.done exists).
# The as-shipped WeSpeaker ResNet34 models are not evaluated (extraction
# stopped after the CMN finding); wespeaker_campplus_vox (as shipped) was
# evaluated by run_eval_all.sh and documents the failure.
set -e
source /home/ali/Desktop/Articles/.venv/bin/activate
cd /home/ali/Desktop/Articles/artifacts/speaker-id-wake
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ready() {
  python - "$1" <<'PY'
import sys, numpy as np, pandas as pd
m = sys.argv[1]
s = pd.read_csv("cache/specs.csv")
a = np.load(f"cache/emb/{m}.npy", mmap_mode="r")
sys.exit(0 if not np.isnan(a[s.spec_id[s.set == "main"].values, 0]).any() else 1)
PY
}
run() {
  local m="$1"; shift
  if [ -f "results/$m/eval.done" ]; then echo "skip $m"; return; fi
  until ready "$m" 2>/dev/null; do sleep 60; done
  python experiments/03_evaluate.py --model "$m" "$@" > "logs/eval_$m.log" 2>&1
  date -Is > "results/$m/eval.done"
  echo "evaluated $m"
}

run wespeaker_resnet34_vox_wsfe
run wespeaker_campplus_vox_wsfe
run wespeaker_resnet34_cnc_wsfe
echo all-done > logs/eval_all3.done
