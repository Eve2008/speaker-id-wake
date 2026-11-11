"""Single-thread CPU time per utterance for each speaker model (this x86 machine).

Each model runs in its own subprocess pinned to one core (taskset), with
sherpa-onnx num_threads=1 and OMP_NUM_THREADS=1. For every test duration we
time create_stream + accept_waveform + compute on real MSWC audio (30 timed
runs after 3 warm-up runs, cycling through 10 different utterances), plus the
NexaRoom scoring step for a 5-person household (NexaRoom's own Python code).
Raspberry Pi timing is NOT measured here.

Output: results/timing.csv, results/timing_env.json
A repeat run (``python experiments/04_timing.py --rep 2``) writes
results/timing_rep2.csv and results/timing_env_rep2.json instead, to measure
run-to-run spread.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from swk import data  # noqa: E402

DURATIONS = {"1 s (1 clip)": (1, 0.0), "2 s (2 clips)": (2, 0.0), "3 s (3 clips)": (3, 0.0),
             "4.3 s enrollment sentence (4 clips)": (4, 0.1), "8.7 s enrollment sentence (8 clips)": (8, 0.1)}
# benchmark configurations (the *_cmn and *_d1 keys use the same graphs/front ends as timed ones)
TIMED_MODELS = ["campplus_zhen_adv", "eres2net_base_zh", "titanet_small", "wespeaker_campplus_vox",
                "wespeaker_resnet34_vox", "wespeaker_resnet34_cnc", "wespeaker_campplus_vox_wsfe",
                "wespeaker_resnet34_vox_wsfe", "wespeaker_resnet34_cnc_wsfe", "campplus_zhen_adv_ort131"]
REPEATS = 30
WARMUP = 3


def child(model: str, out: str) -> None:
    import resource

    import sherpa_onnx

    sys.path.insert(0, "/home/ali/Desktop/Articles/work/NexaRoom")
    from nexaroom.voice.speakers import SpeakerEngine

    meta = data.load_meta()
    rng = np.random.default_rng(0)
    files = meta.file.to_numpy()[rng.choice(len(meta), 200, replace=False)]
    clips = [data.load_clip_int16(f) for f in files]
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t = time.perf_counter()
    if data.MODELS[model].backend == "wespeaker_ort":
        from swk.frontend import WeSpeakerOrtExtractor

        ex = WeSpeakerOrtExtractor(str(data.model_path(model)), dither=data.MODELS[model].dither)
    elif data.MODELS[model].backend == "sherpalike_ort":
        from swk.frontend import SherpaLikeOrtExtractor

        ex = SherpaLikeOrtExtractor(str(data.model_path(model)), dither=data.MODELS[model].dither)
    else:
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(data.model_path(model)), num_threads=1,
                                                          provider="cpu")
        ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
    load_s = time.perf_counter() - t
    rows = []
    for label, (k, gap) in DURATIONS.items():
        utts = []
        for j in range(10):
            pcm = data.join(clips[j * 8:j * 8 + k], gap)
            utts.append(pcm.astype(np.float32) / 32768.0)
        times = []
        for r in range(WARMUP + REPEATS):
            w = utts[r % len(utts)]
            c0 = time.process_time()
            t0 = time.perf_counter()
            s = ex.create_stream()
            s.accept_waveform(sample_rate=16000, waveform=w)
            s.input_finished()
            v = ex.compute(s)
            wall = time.perf_counter() - t0
            cpu = time.process_time() - c0
            if r >= WARMUP:
                times.append((wall, cpu))
        a = np.array(times) * 1000
        dur_s = utts[0].size / 16000
        rows.append({"model": model, "utterance": label, "audio_s": round(dur_s, 2),
                     "wall_ms_median": float(np.median(a[:, 0])), "wall_ms_p10": float(np.percentile(a[:, 0], 10)),
                     "wall_ms_p90": float(np.percentile(a[:, 0], 90)), "cpu_ms_median": float(np.median(a[:, 1])),
                     "rtf_median": float(np.median(a[:, 0]) / 1000 / dur_s)})
    # NexaRoom scoring for a 5-person household with 3 enrollment sentences each (pure Python + numpy)
    d = ex.dim
    profs = [list(data_unit(rng, 3, d)) for _ in range(5)]
    q = data_unit(rng, 1, d)[0]
    t0 = time.perf_counter()
    for _ in range(200):
        sorted((SpeakerEngine.profile_score(q, p) for p in profs), reverse=True)
    score_ms = (time.perf_counter() - t0) / 200 * 1000
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for r in rows:
        r.update({"embedding_dim": d, "load_s": load_s, "scoring_ms_5_profiles": score_ms,
                  "rss_increase_mb": (rss1 - rss0) / 1024.0,
                  "file_mb": data.model_path(model).stat().st_size / 1e6})
    pd.DataFrame(rows).to_csv(out, index=False)


def data_unit(rng, n, d):
    x = rng.standard_normal((n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main(rep: int = 1) -> None:
    sfx = "" if rep == 1 else f"_rep{rep}"
    tmp = data.RESULTS / f"timing_parts{sfx}"
    tmp.mkdir(parents=True, exist_ok=True)
    env = {"cpu": subprocess.run(["bash", "-c", "lscpu | grep 'Model name' | head -1"], capture_output=True,
                                 text=True).stdout.strip(),
           "python": platform.python_version(), "machine": platform.machine(),
           "loadavg_before": os.getloadavg(), "core": 3, "repeats": REPEATS, "warmup": WARMUP,
           "note": "Other jobs were running on the machine; numbers are single-thread wall/CPU time on one pinned core."}
    try:
        import sherpa_onnx
        env["sherpa_onnx"] = sherpa_onnx.__version__
    except Exception:  # noqa: BLE001
        pass
    frames = []
    for m in TIMED_MODELS:
        out = tmp / f"{m}.csv"
        if out.exists():  # resumable: one file per model
            frames.append(pd.read_csv(out))
            continue
        cmd = ["taskset", "-c", str(env["core"]), sys.executable, __file__, "--child", m, str(out)]
        subprocess.run(cmd, check=True, env={**os.environ, "OMP_NUM_THREADS": "1"})
        frames.append(pd.read_csv(out))
        print(frames[-1][["model", "utterance", "wall_ms_median", "cpu_ms_median"]].to_string(index=False), flush=True)
    env["loadavg_after"] = os.getloadavg()
    pd.concat(frames).to_csv(data.RESULTS / f"timing{sfx}.csv", index=False)
    with open(data.RESULTS / f"timing_env{sfx}.json", "w") as f:
        json.dump(env, f, indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 2 and sys.argv[1] == "--rep":
        main(int(sys.argv[2]))
    else:
        main()
