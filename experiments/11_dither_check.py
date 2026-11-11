"""Effect of a 1-LSB noise floor (dither) on every model, small subset.

MSWC clips are zero-padded to 1 s: in a random sample of 300 clips the median
share of samples that are exactly 0 is 0.61, and 97% of clips contain a run
of exact zeros of at least 0.1 s. A microphone never delivers exact digital
silence. With dither = 0 (sherpa-onnx) the log-mel energy of those frames hits
the float epsilon floor, and for models that take int16-scaled input
(WeSpeaker, normalize_samples = 0) this is far outside what they saw in
training (WeSpeaker trains with Kaldi dither 1.0 on int16-scaled audio).

Here each utterance gets Gaussian noise with a standard deviation of 1 int16
LSB (about -90 dBFS RMS) before embedding, with a fixed seed per utterance, and
the cosine EER on the same 100-speaker subset as 09/10 is compared with the
undithered input, for every model.

Usage: python experiments/11_dither_check.py MODEL [MODEL ...]
Output: results/dither_check/<model>.json (one file per model)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swk import data, metrics, rules  # noqa: E402

N_SPK = 100


def main(models: list[str]) -> None:
    import sherpa_onnx

    meta = data.load_meta().set_index("clip")
    with open(data.RESULTS / "splits" / "seed0.json") as f:
        sp = {int(k): v for k, v in json.load(f).items()}
    ids = [k for k in sorted(sp) if not sp[k]["dev"]][:N_SPK]
    enroll, test, owner = [], [], []
    for i, k in enumerate(ids):
        clips = {c: data.load_clip_int16(meta.at[c, "file"]) for c in sp[k]["enroll"][:12] + sp[k]["test"]}
        e = sp[k]["enroll"]
        enroll.append([data.join([clips[c] for c in e[j * 4:(j + 1) * 4]], 0.1) for j in range(3)])
        for c in sp[k]["test"]:
            test.append(clips[c])
            owner.append(i)
    owner = np.array(owner)
    edges = metrics.hist_edges(-1.0, 1.0, 4000)
    out_dir = data.RESULTS / "dither_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in models:
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(data.model_path(m)), num_threads=1,
                                                          provider="cpu")
        ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        res = {"model": m, "n_speakers": len(ids), "n_test": int(owner.size), "seed": 0}
        for cond, lsb in (("raw", 0.0), ("dither_1lsb", 1.0)):
            def emb(pcm, tag):
                x = pcm.astype(np.float64)
                if lsb:
                    x = x + lsb * np.random.default_rng(tag).standard_normal(x.size)
                st = ex.create_stream()
                st.accept_waveform(sample_rate=16000, waveform=(x / 32768.0).astype(np.float32))
                st.input_finished()
                v = np.asarray(ex.compute(st), dtype=np.float64)
                return v / np.linalg.norm(v)

            enr = np.array([[emb(p, [i, j, 1]) for j, p in enumerate(s)] for i, s in enumerate(enroll)])
            q = np.array([emb(p, [u, 0, 2]) for u, p in enumerate(test)])
            _, cos = rules.profile_score_matrix(q, enr)
            tgt = cos[np.arange(owner.size), owner]
            mask = np.ones_like(cos, bool)
            mask[np.arange(owner.size), owner] = False
            eer, mindcf = metrics.eer_from_hist(np.histogram(tgt, edges)[0].astype(float),
                                                np.histogram(cos[mask], edges)[0].astype(float))
            rng = np.random.default_rng(0)
            acc = []
            for u in range(owner.size):
                others = rng.choice(np.delete(np.arange(len(ids)), owner[u]), (50, 4))
                cand = np.concatenate([np.full((50, 1), owner[u]), others], 1)
                acc.append(np.mean(np.argmax(cos[u][cand], 1) == 0))
            res[cond] = {"eer_cos_1s": float(eer), "mindcf_1s": float(mindcf),
                         "closed_set_acc_N5_1s": float(np.mean(acc)),
                         "norm_of_mean_test_embedding": float(np.linalg.norm(q.mean(0)))}
            print(m, cond, res[cond], flush=True)
        with open(out_dir / f"{m}.json", "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
