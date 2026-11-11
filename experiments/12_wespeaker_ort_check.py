"""Subset check of the WeSpeaker front end + onnxruntime path (swk/frontend.py).

Same 100-speaker subset as 09-11 (seed 0, first 100 eval speakers, 3x4
enrollment, six 1 s test clips). Conditions per WeSpeaker model:
  wespeaker_frontend   int16 scale, Hamming, dither 1 LSB, CMN (the configuration used in the benchmark)
  no_dither            same, dither 0
  no_cmn               same, no CMN
Usage: python experiments/12_wespeaker_ort_check.py MODEL [MODEL ...]   (keys of the as-shipped files)
Output: results/wespeaker_ort_check/<model>.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swk import data, frontend, metrics, rules  # noqa: E402

N_SPK = 100
CONDS = {"wespeaker_frontend": (1.0, True), "no_dither": (0.0, True), "no_cmn": (1.0, False)}


def main(models: list[str]) -> None:
    import onnxruntime as ort

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
    out_dir = data.RESULTS / "wespeaker_ort_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in models:
        res = {"model": m, "onnxruntime": ort.__version__, "n_speakers": len(ids), "n_test": int(owner.size)}
        for cond, (dither, cmn) in CONDS.items():
            ex = frontend.WeSpeakerOrtExtractor(str(data.model_path(m)), dither=dither, cmn=cmn)

            def emb(pcm, tag):
                st = ex.create_stream(seed=tag)
                st.accept_waveform(16000, pcm.astype(np.float32) / 32768.0)
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
            res[cond] = {"dither_lsb": dither, "cmn": cmn, "eer_cos_1s": float(eer), "mindcf_1s": float(mindcf),
                         "closed_set_acc_N5_1s": float(np.mean(acc)),
                         "norm_of_mean_test_embedding": float(np.linalg.norm(q.mean(0)))}
            print(m, cond, res[cond], flush=True)
            with open(out_dir / f"{m}.json", "w") as f:
                json.dump(res, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
