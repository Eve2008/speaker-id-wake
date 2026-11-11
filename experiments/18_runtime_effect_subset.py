"""Effect of the onnxruntime version on accuracy, same features, small subset.

results/ort_version_check.json shows that onnxruntime <= 1.28 (sherpa-onnx
2.0.1 bundles 1.28.2) and >= 1.29 return different embeddings for the two
CAM++ graphs, and results/avgpool_check.json shows one concrete change:
AveragePool with ceil_mode=1 and count_include_pad=1 (used 52 times by the
3D-Speaker CAM++ graph) divides the last, partial window by the full kernel in
1.27.0 and by the number of valid frames in 1.29.0 / 1.31.1 (the PyTorch
behaviour).

Here the same numpy features (sherpa-like: Povey, 20-7600 Hz, sample scale and
CMN from the model metadata, so sherpa-onnx's own conventions) are fed to the
graph by onnxruntime-node 1.27.0 and by onnxruntime 1.31.1 (Python), and the
cosine EER / closed-set accuracy (N = 5) at 1 s are computed on the
100-speaker subset used by 09-12 (seed 0). sherpa-onnx's own result is
included for reference.

Usage: python experiments/18_runtime_effect_subset.py NODE_PREFIX MODEL [MODEL ...]
Output: results/runtime_effect_subset/<model>.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from swk import data, frontend, metrics, rules  # noqa: E402

N_SPK = 100


def scores(enr: np.ndarray, q: np.ndarray, owner: np.ndarray, n_spk: int) -> dict:
    edges = metrics.hist_edges(-1.0, 1.0, 4000)
    _, cos = rules.profile_score_matrix(q, enr)
    tgt = cos[np.arange(owner.size), owner]
    mask = np.ones_like(cos, bool)
    mask[np.arange(owner.size), owner] = False
    eer, mindcf = metrics.eer_from_hist(np.histogram(tgt, edges)[0].astype(float),
                                        np.histogram(cos[mask], edges)[0].astype(float))
    rng = np.random.default_rng(0)
    acc = []
    for u in range(owner.size):
        others = rng.choice(np.delete(np.arange(n_spk), owner[u]), (50, 4))
        cand = np.concatenate([np.full((50, 1), owner[u]), others], 1)
        acc.append(np.mean(np.argmax(cos[u][cand], 1) == 0))
    return {"eer_cos_1s": float(eer), "mindcf_1s": float(mindcf), "closed_set_acc_N5_1s": float(np.mean(acc))}


def main(node_prefix: str, models: list[str]) -> None:
    import onnxruntime as ort
    import sherpa_onnx

    meta = data.load_meta().set_index("clip")
    with open(data.RESULTS / "splits" / "seed0.json") as f:
        sp = {int(k): v for k, v in json.load(f).items()}
    ids = [k for k in sorted(sp) if not sp[k]["dev"]][:N_SPK]
    utts, owner_t = [], []  # enrollment first (3 per speaker), then tests
    for i, k in enumerate(ids):
        clips = {c: data.load_clip_int16(meta.at[c, "file"]) for c in sp[k]["enroll"][:12] + sp[k]["test"]}
        e = sp[k]["enroll"]
        utts += [data.join([clips[c] for c in e[j * 4:(j + 1) * 4]], 0.1) for j in range(3)]
    for i, k in enumerate(ids):
        for c in sp[k]["test"]:
            utts.append(data.load_clip_int16(meta.at[c, "file"]))
            owner_t.append(i)
    owner = np.array(owner_t)
    n_enr = 3 * len(ids)
    out_dir = data.RESULTS / "runtime_effect_subset"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    n = lambda v: np.asarray(v, dtype=np.float64) / np.linalg.norm(v)  # noqa: E731
    for m in models:
        path = str(data.model_path(m))
        sess = ort.InferenceSession(path, ort.SessionOptions(), providers=["CPUExecutionProvider"])
        md = sess.get_modelmeta().custom_metadata_map
        scale = 1.0 / 32768.0 if md.get("normalize_samples", "1") == "1" else 1.0
        cmn = md.get("feature_normalize_type", "") == "global-mean"
        lines, py = [], []
        for u, pcm in enumerate(utts):
            f = frontend.fbank(pcm.astype(np.float64) * scale, window="povey", high=-400.0)
            if cmn:
                f = f - f.mean(axis=0, keepdims=True)
            f = f.astype(np.float32)
            py.append(n(sess.run(None, {sess.get_inputs()[0].name: f[None]})[0][0]))
            f.tofile(tmp / f"f{u}.f32")
            lines.append(f"{tmp / f'f{u}.f32'} {f.shape[0]} {tmp / f'o{u}.f32'}")
        (tmp / "list.txt").write_text("\n".join(lines))
        subprocess.run(["node", str(ROOT / "experiments" / "ort_node_batch.js"),
                        str(Path(node_prefix) / "v127" / "node_modules" / "onnxruntime-node"), path,
                        str(tmp / "list.txt")], check=True)
        nd = [n(np.fromfile(tmp / f"o{u}.f32", dtype=np.float32)) for u in range(len(utts))]
        ex = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=path, num_threads=1, provider="cpu"))
        sh = []
        for pcm in utts:
            st = ex.create_stream()
            st.accept_waveform(sample_rate=16000, waveform=pcm.astype(np.float32) / 32768.0)
            st.input_finished()
            sh.append(n(ex.compute(st)))
        res = {"model": m, "n_speakers": len(ids), "n_test": int(owner.size), "seed": 0,
               "features": {"scale": "float" if scale != 1.0 else "int16", "cmn": cmn}}
        for name, embs in (("onnxruntime_1.31.1_python", py), ("onnxruntime_node_1.27.0", nd),
                           ("sherpa_onnx_2.0.1", sh)):
            e = np.array(embs)
            res[name] = scores(e[:n_enr].reshape(len(ids), 3, -1), e[n_enr:], owner, len(ids))
            print(m, name, res[name], flush=True)
        with open(out_dir / f"{m}.json", "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
