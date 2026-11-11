"""How much exact digital silence do the MSWC clips contain?

For the 1 s test clips of seed 0 (all pool speakers, 6 clips each): share of
samples that are exactly 0, longest run of exact zeros, and, for the log-mel
front end used by the WeSpeaker models (int16 scale, no dither), the gap
between the median log-mel value of all-zero frames (log of float32 eps) and
the median log-mel value of the clip's loudest 10% of frames.

Output: results/zero_padding.json, results/zero_padding_per_clip.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swk import data, frontend  # noqa: E402


def main() -> None:
    meta = data.load_meta().set_index("clip")
    with open(data.RESULTS / "splits" / "seed0.json") as f:
        sp = json.load(f)
    clips = sorted({c for v in sp.values() for c in v["test"]})
    rows = []
    for c in clips:
        x = data.load_clip_int16(meta.at[c, "file"])
        z = x == 0
        d = np.diff(np.concatenate([[0], z.astype(np.int8), [0]]))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        run = int((ends - starts).max()) if starts.size else 0
        fb = frontend.fbank(x.astype(np.float64), window="hamming")
        frame_mean = fb.mean(axis=1)
        floor = float(np.log(np.finfo(np.float32).eps))
        n_floor = int(np.sum(np.all(fb <= floor + 1e-6, axis=1)))
        loud = np.sort(frame_mean)[-max(1, frame_mean.size // 10):].mean()
        rows.append({"clip": c, "samples": x.size, "zero_share": float(z.mean()), "longest_zero_run_s": run / data.SR,
                     "frames": fb.shape[0], "frames_at_log_floor": n_floor, "loudest10pct_frame_logmel": float(loud)})
    df = pd.DataFrame(rows)
    df.to_csv(data.RESULTS / "zero_padding_per_clip.csv", index=False)
    floor = float(np.log(np.finfo(np.float32).eps))
    out = {"n_clips": int(len(df)), "zero_share_median": float(df.zero_share.median()),
           "zero_share_p25": float(df.zero_share.quantile(0.25)), "zero_share_p75": float(df.zero_share.quantile(0.75)),
           "share_clips_zero_run_ge_0.1s": float((df.longest_zero_run_s >= 0.1).mean()),
           "longest_zero_run_s_median": float(df.longest_zero_run_s.median()),
           "frames_at_log_floor_share_median": float((df.frames_at_log_floor / df.frames).median()),
           "log_floor": floor,
           "loudest10pct_frame_logmel_median_int16": float(df.loudest10pct_frame_logmel.median()),
           "gap_nats_median": float(df.loudest10pct_frame_logmel.median() - floor)}
    with open(data.RESULTS / "zero_padding.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
