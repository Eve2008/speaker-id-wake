"""Sanity checks on the per-model result folders and the embedding cache.

For every model: are the main-set embeddings complete (no NaN rows, unit
norm), does results/<model>/ hold every expected file, were the files written
after the current 03_evaluate.py, do the tables have the expected number of
rows and no missing estimates, and do they cover all 5 seeds.

Output: results/check_results.json (and a short printout).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swk import data, splits  # noqa: E402

FILES = ["main.csv", "eer.csv", "gender.csv", "gender_eer.csv", "curves.csv", "tuned_thresholds.csv",
         "per_speaker_main.csv.gz", "counts.json"]
ABL_FILES = ["fixed_word.csv", "fixed_word_eer.csv", "ablations.csv", "ablations_eer.csv"]


def main() -> None:
    specs = pd.read_csv(data.CACHE / "specs.csv")
    main_ids = specs.spec_id[specs.set == "main"].to_numpy()
    abl_ids = specs.spec_id[specs.set == "ablation"].to_numpy()
    eval_mtime = (data.ROOT / "experiments" / "03_evaluate.py").stat().st_mtime
    report = {}
    for m in data.MODELS:
        r: dict = {}
        p = data.CACHE / "emb" / f"{m}.npy"
        if p.exists():
            a = np.load(p, mmap_mode="r")
            done = ~np.isnan(a).any(axis=1)
            mixed = int((np.isnan(a).any(axis=1) & ~np.isnan(a).all(axis=1)).sum())
            nrm = np.linalg.norm(np.asarray(a[main_ids[done[main_ids]]], dtype=np.float64), axis=1)
            r["emb_main_done"] = int(done[main_ids].sum())
            r["emb_main_total"] = int(main_ids.size)
            r["emb_ablation_done"] = int(done[abl_ids].sum())
            r["emb_mixed_nan_rows"] = mixed
            r["emb_norm_min_max"] = [float(nrm.min()), float(nrm.max())] if nrm.size else None
        out = data.RESULTS / m
        have = [f for f in FILES + ABL_FILES if (out / f).exists()]
        r["files_present"] = have
        r["missing_core_files"] = [f for f in FILES if f not in have]
        r["eval_done_marker"] = (out / "eval.done").read_text().strip() if (out / "eval.done").exists() else None
        r["written_after_current_evaluate_py"] = bool(have) and all(
            (out / f).stat().st_mtime >= eval_mtime for f in have)
        if (out / "main.csv").exists():
            mm = pd.read_csv(out / "main.csv")
            r["main_rows"] = int(len(mm))
            r["main_nan_est"] = int(mm.est.isna().sum())
            r["main_seed_cols_complete"] = bool(mm[[f"seed{s}" for s in splits.SEEDS]].notna().all().all())
            r["main_n_spk_range"] = [int(mm.n_spk.min()), int(mm.n_spk.max())]
        if (out / "eer.csv").exists():
            e = pd.read_csv(out / "eer.csv")
            r["eer_rows"] = int(len(e))
            r["eer_nan"] = int(e.eer.isna().sum())
        report[m] = r
        print(m, {k: v for k, v in r.items() if k != "files_present"})
    with open(data.RESULTS / "check_results.json", "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
