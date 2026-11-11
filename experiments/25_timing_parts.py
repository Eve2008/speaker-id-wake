"""CPU time of the front end and of the ONNX graph, measured separately (revision).

Review point: the timings in results/timing.csv mix implementations (sherpa-onnx's
C++ front end for some models, our NumPy front end for the WeSpeaker and
onnxruntime-1.31.1 configurations). Here every model's graph runs on the same
runtime (onnxruntime 1.31.1, Python, one intra-op thread) with precomputed
features of the right shape, and the NumPy front end and sherpa-onnx's
end-to-end path are timed in the same pinned process for reference. 60 timed
runs after 3 warm-up runs per length, cycling through 10 utterances; CPU time
(time.process_time). The machine was busier than during 04_timing.py, so only
within-run comparisons are meaningful; the end-to-end sherpa-onnx row of each
model links this run to results/timing.csv.

Output: results/timing_parts_rev.csv
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from swk import data, frontend  # noqa: E402

MODELS = ["campplus_zhen_adv", "eres2net_base_zh", "titanet_small", "wespeaker_campplus_vox", "campplus_vox_cip0_d1", "eres2net_vox_d1",
          "wespeaker_resnet34_vox", "wespeaker_resnet34_cnc"]
LENGTHS = {"1 s": 1, "2 s": 2, "3 s": 3}
REPEATS, WARMUP = 60, 3


def timeit(fn, items) -> tuple[float, float]:
    """(median, minimum) CPU ms over REPEATS runs; the minimum is less sensitive to contention."""
    ts = []
    for r in range(WARMUP + REPEATS):
        x = items[r % len(items)]
        c0 = time.process_time()
        fn(x)
        if r >= WARMUP:
            ts.append(time.process_time() - c0)
    return float(np.median(ts) * 1000), float(np.min(ts) * 1000)


def child(model: str, out: str) -> None:
    import onnxruntime as ort
    import sherpa_onnx

    meta = data.load_meta()
    rng = np.random.default_rng(0)
    files = meta.file.to_numpy()[rng.choice(len(meta), 200, replace=False)]
    clips = [data.load_clip_int16(f) for f in files]
    path = str(data.model_path(model))
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    ins = [i.name for i in sess.get_inputs()]
    nemo = model == "titanet_small"
    outs = ["embs"] if nemo else None
    ex = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=path, num_threads=1, provider="cpu"))
    rows = []
    for label, k in LENGTHS.items():
        pcm = [np.concatenate(clips[j * 8:j * 8 + k]) for j in range(10)]
        wav = [p.astype(np.float32) / 32768.0 for p in pcm]
        feats = [frontend.fbank(p.astype(np.float64), window="hamming").astype(np.float32) for p in pcm]
        if nemo:
            feeds = [{ins[0]: f.T[None].copy(), ins[1]: np.array([f.shape[0]], np.int64)} for f in feats]
        else:
            feeds = [{ins[0]: f[None]} for f in feats]

        def e2e(w):
            s = ex.create_stream()
            s.accept_waveform(sample_rate=16000, waveform=w)
            s.input_finished()
            ex.compute(s)

        gm, gmin = timeit(lambda fd: sess.run(outs, fd), feeds)
        fm, fmin = timeit(lambda p: frontend.fbank(p.astype(np.float64), window="hamming"), pcm)
        sm, smin = timeit(e2e, wav)
        rows.append({"model": model, "utterance": label, "graph_only_ort131_ms": gm, "graph_only_ort131_min_ms": gmin,
                     "numpy_frontend_ms": fm, "numpy_frontend_min_ms": fmin,
                     "sherpa_onnx_end_to_end_ms": sm, "sherpa_onnx_end_to_end_min_ms": smin})
    pd.DataFrame(rows).to_csv(out, index=False)


def main() -> None:
    tmp = data.RESULTS / "timing_parts_rev"
    tmp.mkdir(parents=True, exist_ok=True)
    frames = []
    load0 = os.getloadavg()
    for m in MODELS:
        out = tmp / f"{m}.csv"
        if not out.exists():
            subprocess.run(["taskset", "-c", "3", sys.executable, __file__, "--child", m, str(out)], check=True,
                           env={**os.environ, "OMP_NUM_THREADS": "1"})
        frames.append(pd.read_csv(out))
        print(frames[-1].to_string(index=False), flush=True)
    df = pd.concat(frames)
    df.to_csv(data.RESULTS / "timing_parts_rev.csv", index=False)
    with open(data.RESULTS / "timing_parts_rev_env.json", "w") as f:
        json.dump({"loadavg_before": load0, "loadavg_after": os.getloadavg(), "core": 3, "repeats": REPEATS,
                   "warmup": WARMUP, "clock": "time.process_time"}, f, indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3])
    else:
        main()
