#!/bin/bash
# Evaluate every model with the current 03_evaluate.py, one results/<model>/
# folder per model. The NexaRoom default model also runs the ablations (it is
# the only model with ablation embeddings). A model whose results/<model>/
# eval.done marker exists is skipped, so the queue can be re-run after an
# interruption. Waits for each model's main-set embeddings to be complete.
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
  until ready "$m"; do sleep 60; done
  rm -f "results/$m/eval.done"
  python experiments/03_evaluate.py --model "$m" "$@" > "logs/eval_$m.log" 2>&1
  date -Is > "results/$m/eval.done"
}
run wespeaker_campplus_vox
run titanet_small
run campplus_zhen_adv --ablations
run wespeaker_resnet34_vox
run wespeaker_resnet34_cnc
run eres2net_base_zh
echo all-done > logs/eval_all.done
