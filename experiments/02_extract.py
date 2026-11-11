"""Compute speaker embeddings for every utterance spec with every model.

Usage: python experiments/02_extract.py --set main [--models a,b] [--workers 8]

Each worker process loads the sherpa-onnx extractors once (num_threads=1) and
handles a chunk of speakers. Embeddings are L2-normalised, as in NexaRoom, and
written to cache/emb/<model>.npy (float32 memmap, NaN = not computed yet), so
the script can be stopped and resumed. Per-utterance audio statistics used by
NexaRoom's audio_problem() check go to cache/spec_audio.npy.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swk import data  # noqa: E402

EMB_DIR = data.CACHE / "emb"
_EXTRACTORS: dict = {}
_META: pd.DataFrame | None = None


def _extractor(key: str):
    if key not in _EXTRACTORS:
        if data.MODELS[key].backend == "wespeaker_ort":
            from swk.frontend import WeSpeakerOrtExtractor

            _EXTRACTORS[key] = WeSpeakerOrtExtractor(str(data.model_path(key)), dither=data.MODELS[key].dither,
                                                       cmn=True)
        elif data.MODELS[key].backend == "sherpalike_ort128":
            from swk.frontend import SherpaLikeOrtcExtractor

            _EXTRACTORS[key] = SherpaLikeOrtcExtractor(str(data.model_path(key)), dither=data.MODELS[key].dither)
        elif data.MODELS[key].backend == "sherpalike_ort":
            from swk.frontend import SherpaLikeOrtExtractor

            _EXTRACTORS[key] = SherpaLikeOrtExtractor(str(data.model_path(key)), dither=data.MODELS[key].dither)
        else:
            import sherpa_onnx

            cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(data.model_path(key)), num_threads=1,
                                                              provider="cpu")
            assert cfg.validate(), key
            _EXTRACTORS[key] = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
    return _EXTRACTORS[key]


def _init() -> None:
    global _META
    _META = data.load_meta().set_index("clip")


def _work(task):
    rows, models = task  # rows: list of (spec_id, clips str, gap, trim)
    assert _META is not None
    cache: dict[int, np.ndarray] = {}
    out_ids, out_audio = [], []
    out = {m: [] for m in models}
    for spec_id, clips, gap, trim in rows:
        ids = [int(c) for c in clips.split()]
        parts = []
        for c in ids:
            if c not in cache:
                cache[c] = data.load_clip_int16(_META.at[c, "file"])
            x = cache[c]
            parts.append(data.trim_int16(x) if trim else x)
        pcm = data.join(parts, gap)
        out_audio.append(data.audio_problem_flags(pcm))
        wav = pcm.astype(np.float32) / 32768.0
        for m in models:
            ex = _extractor(m)
            # the WeSpeaker/onnxruntime path dithers with a fixed seed per utterance spec
            st = ex.create_stream(seed=[int(spec_id), 17]) if data.MODELS[m].backend != "sherpa" \
                else ex.create_stream()
            w = wav
            if data.MODELS[m].backend == "sherpa" and data.MODELS[m].dither > 0:
                # same fixed-seed 1-LSB Gaussian noise floor as the WeSpeaker/onnxruntime path
                noise = np.random.default_rng([int(spec_id), 17]).standard_normal(pcm.size)
                w = ((pcm.astype(np.float64) + data.MODELS[m].dither * noise) / 32768.0).astype(np.float32)
            st.accept_waveform(sample_rate=data.SR, waveform=w)
            st.input_finished()
            if not ex.is_ready(st):
                v = np.full(ex.dim, np.nan, dtype=np.float32)
            else:
                v = np.asarray(ex.compute(st), dtype=np.float32)
                n = float(np.linalg.norm(v))
                v = v / n if np.isfinite(n) and n > 1e-9 else np.full_like(v, np.nan)
            out[m].append(v)
        out_ids.append(spec_id)
    return np.array(out_ids), np.array(out_audio, dtype=np.float32), {m: np.vstack(v) for m, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="main", choices=["main", "ablation", "all"])
    ap.add_argument("--models", default="campplus_zhen_adv,titanet_small,wespeaker_campplus_vox,eres2net_base_zh,wespeaker_resnet34_vox,wespeaker_resnet34_cnc")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N pending specs")
    ap.add_argument("--trim-only", action="store_true", help="with --set ablation: only the trimmed specs")
    args = ap.parse_args()
    models = args.models.split(",")
    specs = pd.read_csv(data.CACHE / "specs.csv")
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    n = len(specs)
    mm = {}
    for m in models:
        p = EMB_DIR / f"{m}.npy"
        dim = _extractor(m).dim
        if not p.exists():
            arr = np.lib.format.open_memmap(p, mode="w+", dtype=np.float32, shape=(n, dim))
            arr[:] = np.nan
            arr.flush()
            del arr
        mm[m] = np.load(p, mmap_mode="r+")
    _EXTRACTORS.clear()
    ap_path = data.CACHE / "spec_audio.npy"
    if not ap_path.exists():
        a = np.lib.format.open_memmap(ap_path, mode="w+", dtype=np.float32, shape=(n, 3))
        a[:] = np.nan
        a.flush()
        del a
    audio = np.load(ap_path, mmap_mode="r+")

    sel = specs if args.set == "all" else specs[specs.set == args.set]
    if args.trim_only:  # revision: trimming ablation only (trimmed enrollment + trimmed 1 s test specs)
        sel = sel[sel.trim.astype(bool)]
    todo_mask = np.zeros(n, dtype=bool)
    todo_mask[sel.spec_id.values] = True
    ctx = get_context("spawn")
    # one model at a time, so evaluation of the first models can start early
    for m in models:
        done = ~np.isnan(mm[m][:, 0])
        todo = specs[todo_mask & ~done]
        print(f"[{m}] {len(todo)} of {len(sel)} specs to embed", flush=True)
        if todo.empty:
            continue
        todo = todo.sort_values(["spk", "spec_id"])  # audio-cache locality
        rows = list(zip(todo.spec_id, todo.clips, todo.gap, todo.trim))
        if args.limit:
            rows = rows[: args.limit]
        tasks = [(rows[i:i + args.chunk], [m]) for i in range(0, len(rows), args.chunk)]
        t0 = time.time()
        finished = 0
        with ctx.Pool(args.workers, initializer=_init) as pool:
            for k, (ids, au, embs) in enumerate(pool.imap_unordered(_work, tasks)):
                audio[ids] = au
                mm[m][ids] = embs[m]
                finished += len(ids)
                # flush after every chunk, so an interruption loses at most the chunks in flight
                mm[m].flush()
                audio.flush()
                if k % 10 == 0 or finished == len(rows):
                    el = time.time() - t0
                    print(f"[{m}] {finished}/{len(rows)} specs, {el/60:.1f} min, "
                          f"eta {el/finished*(len(rows)-finished)/60:.1f} min", flush=True)
        mm[m].flush()
        audio.flush()
        bad = int(np.isnan(mm[m][sel.spec_id.values, 0]).sum())
        print(f"[{m}] done in {(time.time()-t0)/60:.1f} min; NaN rows in set: {bad}", flush=True)


if __name__ == "__main__":
    main()
