"""How much speech do the benchmark's "1 s", "2 s" and "3 s" inputs contain? (revision)

Active speech per clip is the span of 25 ms frames within 30 dB of the clip's
loudest frame (06_clip_stats.py, the rule of the trimming ablation); an
utterance made of several clips gets the sum over its clips. Reported for
seed 0 (all 1,366 pool speakers): the 1/2/3-clip test utterances and the
4-clip enrollment sentences. Also from cache/spec_audio.npy (all seeds): the
length of the 2-clip utterances relative to NexaRoom's 2.0 s short-audio
boundary, and the length of the trimmed 1 s test clips relative to its 0.8 s
minimum.

Output: results/speech_content.json
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


def q(v) -> dict:
    v = np.asarray(v, dtype=float)
    return {"n": int(v.size), "median": float(np.median(v)), "p25": float(np.percentile(v, 25)),
            "p75": float(np.percentile(v, 75)), "p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))}


def main() -> None:
    meta = data.load_meta().set_index("clip")
    specs = pd.read_csv(data.CACHE / "specs.csv")
    roles = pd.read_csv(data.CACHE / "roles.csv")
    audio = np.load(data.CACHE / "spec_audio.npy", mmap_mode="r")
    r0 = roles[(roles.seed == 0) & roles.role.isin(["enroll_3x4", "test1", "test2", "test3"])]
    sp = specs.set_index("spec_id").loc[r0.spec_id.unique()]
    clips = sorted({int(c) for s in sp.clips for c in s.split()})
    act = {}
    for i, c in enumerate(clips):
        act[c] = active_seconds(data.load_clip_int16(meta.at[c, "file"]))
        if i % 5000 == 0:
            print(i, len(clips), flush=True)
    out = {"definition": "frames within 30 dB of the clip's loudest frame, summed over the clips of an utterance",
           "seed": 0}
    for role in ("test1", "test2", "test3", "enroll_3x4"):
        ids = r0[r0.role == role].spec_id.to_numpy()
        a = [sum(act[int(c)] for c in sp.at[i, "clips"].split()) for i in ids]
        dur = np.asarray(audio[ids, 0], dtype=float)
        out[role] = {"active_speech_s": q(a), "length_s": q(dur)}
    m = specs[(specs.set == "main") & (specs.kind == "test")]
    d2 = np.asarray(audio[m[m.n_clips == 2].spec_id.to_numpy(), 0], dtype=float)
    out["test2_all_seeds"] = {"n": int(d2.size), "min_s": float(d2.min()), "max_s": float(d2.max()),
                              "n_below_2.0_s": int(np.sum(d2 < 2.0))}
    t = specs[(specs.set == "ablation") & (specs.kind == "test") & specs.trim.astype(bool)]
    dt = np.asarray(audio[t.spec_id.to_numpy(), 0], dtype=float)
    out["test1_trimmed_all_seeds"] = {"length_s": q(dt), "n_below_0.8_s": int(np.sum(dt < 0.8)), "n": int(dt.size)}
    with open(data.RESULTS / "speech_content.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
