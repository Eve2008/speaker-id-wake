source /home/ali/Desktop/Articles/.venv/bin/activate
cd /home/ali/Desktop/Articles/artifacts/speaker-id-wake
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
ready() {  # model, set (main | trim | ablation)
  python - "$1" "$2" <<'PY'
import sys, numpy as np, pandas as pd
m, which = sys.argv[1].split("__")[0], sys.argv[2]
s = pd.read_csv("cache/specs.csv")
sel = s[s.set == "main"] if which == "main" else s[s.set == "ablation"]
if which == "trim":
    sel = sel[sel.trim.astype(bool)]
a = np.load(f"cache/emb/{m}.npy", mmap_mode="r")
sys.exit(0 if not np.isnan(a[sel.spec_id.values, 0]).any() else 1)
PY
}
run() {  # cfg, needed-set, extra args
  local c="$1" need="$2"; shift 2
  if [ -f "results/$c/eval_rev.done" ]; then echo "skip $c"; return; fi
  until ready "$c" main 2>/dev/null && { [ "$need" = main ] || ready "$c" "$need" 2>/dev/null; }; do sleep 60; done
  local m="${c%%__*}" b=""
  [ "$c" != "$m" ] && b="--backend ${c#*__}"
  python experiments/03_evaluate.py --model "$m" $b "$@" > "logs/eval_rev_$c.log" 2>&1
  python experiments/19_paired.py --dump "$c" --force > "logs/paired_rev_$c.log" 2>&1
  date -Is > "results/$c/eval_rev.done"
  echo "evaluated $c $(date -Is)"
}
