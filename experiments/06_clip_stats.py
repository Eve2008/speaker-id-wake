"""Audio statistics of the clips used as test utterances.

Active speech per clip = span of 25 ms frames (10 ms hop) within 30 dB of the
clip's loudest frame (the same rule as the trimming ablation). Reported for the
random-word 1 s test clips of seed 0 and for the fixed-word clips.

Output: results/clip_stats.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swk import data  # noqa: E402


def active_seconds(x: np.ndarray) -> float:
    return data.trim_int16(x, margin_s=0.0).size / data.SR


def main() -> None:
    meta = data.load_meta().set_index("clip")
    with open(data.RESULTS / "splits" / "seed0.json") as f:
        sp = json.load(f)
    groups: dict[str, list[int]] = {"random_word_test_clips": [c for v in sp.values() for c in v["test"]]}
    for v in sp.values():
        for w, c in v.get("fw", {}).items():
            groups.setdefault(f"fixed_word:{w}", []).append(c)
    rows = []
    for name, clips in groups.items():
        act, rms = [], []
        for c in clips:
            x = data.load_clip_int16(meta.at[c, "file"])
            act.append(active_seconds(x))
            v = x.astype(np.float32)
            rms.append(float(np.sqrt(np.mean(v * v))))
        act = np.array(act)
        rows.append({"group": name, "n_clips": len(clips), "active_s_median": float(np.median(act)),
                     "active_s_p25": float(np.percentile(act, 25)), "active_s_p75": float(np.percentile(act, 75)),
                     "rms_median": float(np.median(rms)), "frac_rms_below_80": float(np.mean(np.array(rms) < 80))})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(data.RESULTS / "clip_stats.csv", index=False)


if __name__ == "__main__":
    main()
