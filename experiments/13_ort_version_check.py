"""Does the onnxruntime version change the speaker embeddings?

For 20 random MSWC clips, compute sherpa-like features in numpy
(swk.frontend.fbank, Povey window, 20-7600 Hz; sample scale and CMN as the
model metadata tells sherpa-onnx: int16 scale and no CMN for WeSpeaker, float
scale and CMN for 3D-Speaker; for WeSpeaker ResNet34 this matches sherpa-onnx's
own output with median cosine 0.98) and run each ONNX graph with
  - the onnxruntime Python package (1.31.1 here),
  - onnxruntime-node 1.27.0, 1.29.0 and 1.31.1 (experiments/ort_node_run.js),
and run sherpa-onnx 2.0.1 (bundled onnxruntime 1.28.2) on the same audio.
Report median cosine between the embeddings of each pair of runtimes.

Usage: python experiments/13_ort_version_check.py NODE_PREFIX
  where NODE_PREFIX/v127, NODE_PREFIX/v129, NODE_PREFIX/v131 each hold
  node_modules/onnxruntime-node (npm install --ignore-scripts onnxruntime-node@X).
Output: results/ort_version_check.json
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
from swk import data, frontend  # noqa: E402

NODE_VERSIONS = {"node_ort_1.27.0": "v127", "node_ort_1.29.0": "v129", "node_ort_1.31.1": "v131"}
MODELS = ["wespeaker_campplus_vox", "wespeaker_resnet34_vox", "wespeaker_resnet34_cnc", "campplus_zhen_adv",
          "eres2net_base_zh"]


def main(node_prefix: str) -> None:
    import onnxruntime as ort
    import sherpa_onnx

    meta = data.load_meta()
    rng = np.random.default_rng(3)
    files = meta.file.to_numpy()[rng.choice(len(meta), 20, replace=False)]
    clips = [data.load_clip_int16(f) for f in files]
    n = lambda v: np.asarray(v, dtype=np.float64) / np.linalg.norm(v)  # noqa: E731
    out = {"n_clips": len(clips), "onnxruntime_python": ort.__version__, "sherpa_onnx": sherpa_onnx.__version__,
           "models": {}}
    tmp = Path(tempfile.mkdtemp())
    for key in MODELS:
        path = str(data.model_path(key))
        emb: dict[str, list] = {k: [] for k in ["python_ort", "sherpa_onnx", *NODE_VERSIONS]}
        sess = ort.InferenceSession(path, ort.SessionOptions(), providers=["CPUExecutionProvider"])
        md = sess.get_modelmeta().custom_metadata_map
        # sherpa-onnx's own conventions, read from the metadata: sample scale and optional CMN
        scale = 1.0 / 32768.0 if md.get("normalize_samples", "1") == "1" else 1.0
        cmn = md.get("feature_normalize_type", "") == "global-mean"
        feats = []
        for c in clips:
            f = frontend.fbank(c.astype(np.float64) * scale, window="povey", high=-400.0)
            feats.append((f - f.mean(axis=0, keepdims=True) if cmn else f).astype(np.float32))
        ex = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=path, num_threads=1, provider="cpu"))
        for i, (c, f) in enumerate(zip(clips, feats)):
            emb["python_ort"].append(n(sess.run(None, {sess.get_inputs()[0].name: f[None]})[0][0]))
            st = ex.create_stream()
            st.accept_waveform(sample_rate=16000, waveform=c.astype(np.float32) / 32768.0)
            st.input_finished()
            emb["sherpa_onnx"].append(n(ex.compute(st)))
            fp = tmp / f"f{i}.f32"
            f.tofile(fp)
            for name, sub in NODE_VERSIONS.items():
                mod = Path(node_prefix) / sub / "node_modules" / "onnxruntime-node"
                op = tmp / f"o{i}.f32"
                subprocess.run(["node", str(ROOT / "experiments" / "ort_node_run.js"), str(mod), path, str(fp),
                                str(f.shape[0]), str(op)], check=True, env={**os.environ, "OMP_NUM_THREADS": "1"})
                emb[name].append(n(np.fromfile(op, dtype=np.float32)))
        names = list(emb)
        pair = {}
        for a in names:
            for b in names:
                if a < b:
                    pair[f"{a} vs {b}"] = float(np.median([x @ y for x, y in zip(emb[a], emb[b])]))
        out["models"][key] = pair
        print(key, json.dumps(pair, indent=1), flush=True)
    with open(data.RESULTS / "ort_version_check.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
