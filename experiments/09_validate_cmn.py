"""Quick check that the CMN flag fixes the WeSpeaker models under sherpa-onnx.

On seed 0, for the first 100 eval speakers (by id): embed the 3x4 enrollment
sentences and the six 1 s test clips with the as-shipped WeSpeaker models and
with their CMN-flagged copies (and with the NexaRoom default model as a
reference). Report the cosine EER (centroid scoring), the norm of the mean
embedding (a large common component means the embeddings carry little speaker
information), and gain invariance: cosine between the embeddings of a 1 s
clip and the same clip at half amplitude (CMN removes a constant log-mel
offset, so it should be close to 1 with CMN).

This is a small sanity check run before the full extraction; the full
benchmark numbers come from 02_extract.py + 03_evaluate.py.
Output: results/cmn_validation.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swk import data, metrics, rules, splits  # noqa: E402

MODELS = ["wespeaker_campplus_vox", "wespeaker_campplus_vox_cmn", "wespeaker_resnet34_vox",
          "wespeaker_resnet34_vox_cmn", "campplus_zhen_adv"]
N_SPK = 100


def main() -> None:
    import sherpa_onnx

    meta = data.load_meta().set_index("clip")
    with open(data.RESULTS / "splits" / "seed0.json") as f:
        sp = {int(k): v for k, v in json.load(f).items()}
    ids = [k for k in sorted(sp) if not sp[k]["dev"]][:N_SPK]
    wav = {}

    def clip(c):
        if c not in wav:
            wav[c] = data.load_clip_int16(meta.at[c, "file"])
        return wav[c]

    enroll_pcm, test_pcm, owner = [], [], []
    for i, k in enumerate(ids):
        e = sp[k]["enroll"]
        enroll_pcm.append([data.join([clip(c) for c in e[j * 4:(j + 1) * 4]], 0.1) for j in range(3)])
        for c in sp[k]["test"]:
            test_pcm.append(clip(c))
            owner.append(i)
    owner = np.array(owner)
    out = {"n_speakers": len(ids), "n_test": int(owner.size), "seed": 0, "models": {}}
    for m in MODELS:
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(data.model_path(m)), num_threads=1,
                                                          provider="cpu")
        ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

        def emb(pcm):
            st = ex.create_stream()
            st.accept_waveform(sample_rate=data.SR, waveform=pcm.astype(np.float32) / 32768.0)
            st.input_finished()
            v = np.asarray(ex.compute(st), dtype=np.float64)
            return v / np.linalg.norm(v)

        enr = np.array([[emb(p) for p in s] for s in enroll_pcm])
        q = np.array([emb(p) for p in test_pcm])
        half = np.array([emb((p.astype(np.int32) // 2).astype(np.int16)) for p in test_pcm[:200]])
        _, cos = rules.profile_score_matrix(q, enr)
        tgt = cos[np.arange(owner.size), owner]
        mask = np.ones_like(cos, bool)
        mask[np.arange(owner.size), owner] = False
        edges = metrics.hist_edges(-1.0, 1.0, 4000)
        ht = np.histogram(tgt, edges)[0].astype(float)
        hn = np.histogram(cos[mask], edges)[0].astype(float)
        eer, _ = metrics.eer_from_hist(ht, hn)
        acc5 = []
        rng = np.random.default_rng(0)
        for u in range(owner.size):
            others = rng.choice(np.delete(np.arange(len(ids)), owner[u]), (50, 4))
            cand = np.concatenate([np.full((50, 1), owner[u]), others], 1)
            acc5.append(np.mean(np.argmax(cos[u][cand], 1) == 0))
        out["models"][m] = {"eer_cos_1s": float(eer), "closed_set_acc_N5_1s": float(np.mean(acc5)),
                            "mean_target_cos": float(tgt.mean()), "mean_nontarget_cos": float(cos[mask].mean()),
                            "norm_of_mean_test_embedding": float(np.linalg.norm(q.mean(0))),
                            "gain_half_vs_full_cos_median": float(np.median(np.sum(half * q[:200], 1)))}
        print(m, out["models"][m], flush=True)
        with open(data.RESULTS / "cmn_validation.json", "w") as f:  # rewritten after every model
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
