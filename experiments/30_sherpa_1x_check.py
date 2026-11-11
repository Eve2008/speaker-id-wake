"""Does sherpa-onnx 1.13.8 (what NexaRoom's pin installs) behave like sherpa-onnx 2.0.1? (revision)

NexaRoom's requirements-voice.txt pins sherpa-onnx>=1.13,<2, whose newest release is 1.13.8,
while the benchmark ran sherpa-onnx 2.0.1. PyPI cannot be reached from this machine, so the
1.13.8 Python wheel was not tested; the Node.js package of the same release
(npm sherpa-onnx-node@1.13.8 + sherpa-onnx-linux-x64@1.13.8) was installed in a scratch folder.
This script
1. reads the ONNX Runtime version of the library bundled with 1.13.8 (OrtGetApiBase version string),
2. runs the one-node AveragePool test of 17_avgpool_check.py (kernel 100, ceil mode,
   count_include_pad = 1, all-ones input) on that library,
3. embeds the same 20 one-second clips and 20 three-second inputs as 20_runtime_isolation.py with
   sherpa-onnx-node 1.13.8 and with sherpa-onnx 2.0.1 (Python) and reports the median cosine
   between the two, per graph.

Usage: python experiments/30_sherpa_1x_check.py <folder with node_modules/sherpa-onnx-node>
Output: results/sherpa_1x_check.json
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from swk import data  # noqa: E402
from swk.ortc import OrtC  # noqa: E402

GRAPHS = ["campplus_zhen_adv", "eres2net_base_zh", "titanet_small"]


def main(prefix: str) -> None:
    import sherpa_onnx

    mod = Path(prefix) / "node_modules" / "sherpa-onnx-node"
    lib = Path(prefix) / "node_modules" / "sherpa-onnx-linux-x64" / "libonnxruntime.so"
    pkg = json.loads((mod / "package.json").read_text())
    res: dict = {"sherpa_onnx_node_version": pkg["version"], "sherpa_onnx_python_version": sherpa_onnx.__version__}
    spec = importlib.util.spec_from_file_location("ap", ROOT / "experiments" / "17_avgpool_check.py")
    ap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ap)
    tmp = data.CACHE / "tmp_sherpa1x"
    tmp.mkdir(exist_ok=True)
    mp = tmp / "pool_cip1.onnx"
    mp.write_bytes(ap.model_bytes(1))
    toy = {}
    version = None
    for L in ap.LENGTHS:
        k = -(-L // 100)
        r = OrtC(lib, mp, "x", "y", k, out_shape=(1, 1, k))
        version = r.version
        toy[f"L={L}"] = [round(float(v), 4) for v in r.run(np.ones((1, 1, L), np.float32))]
    res["bundled_onnxruntime_version"] = version
    res["avgpool_toy_count_include_pad_1"] = toy

    meta = data.load_meta()
    rng = np.random.default_rng(3)  # same clips as 13_ort_version_check.py and 20_runtime_isolation.py
    files = meta.file.to_numpy()[rng.choice(len(meta), 20, replace=False)]
    utts = {"1s": [data.load_clip_int16(f) for f in files]}
    rng3 = np.random.default_rng(4)
    f3 = meta.file.to_numpy()[rng3.choice(len(meta), 60, replace=False)].reshape(20, 3)
    utts["3s"] = [np.concatenate([data.load_clip_int16(f) for f in row]) for row in f3]
    lists = {}
    for ln, us in utts.items():
        names = []
        for i, u in enumerate(us):
            p = tmp / f"{ln}_{i:02d}.f32"
            (u.astype(np.float32) / 32768.0).tofile(p)
            names.append(str(p))
        lp = tmp / f"{ln}.txt"
        lp.write_text("\n".join(names) + "\n")
        lists[ln] = lp
    nrm = lambda v: np.asarray(v, dtype=np.float64) / np.linalg.norm(v)  # noqa: E731
    agree = {}
    for g in GRAPHS:
        path = data.model_path(g)
        ex = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(path), num_threads=1, provider="cpu"))
        d = {}
        for ln, us in utts.items():
            out = tmp / f"{g}_{ln}.f32"
            subprocess.run(["node", str(ROOT / "experiments" / "sherpa_node_embed.js"), str(mod), str(path),
                            str(lists[ln]), str(out)], check=True, capture_output=True)
            e13 = np.fromfile(out, dtype=np.float32).reshape(len(us), ex.dim)
            cos = []
            for u, v in zip(us, e13):
                st = ex.create_stream()
                st.accept_waveform(sample_rate=16000, waveform=u.astype(np.float32) / 32768.0)
                st.input_finished()
                cos.append(float(nrm(ex.compute(st)) @ nrm(v)))
            d[f"{ln}: sherpa-onnx 1.13.8 (node) vs 2.0.1 (python), median cosine"] = float(np.median(cos))
            d[f"{ln}: minimum cosine"] = float(np.min(cos))
        agree[g] = d
        print(g, d, flush=True)
    res["n_utterances"] = {k: len(v) for k, v in utts.items()}
    res["agreement"] = agree
    res["note"] = ("NexaRoom pins sherpa-onnx>=1.13,<2; the Python wheel of 1.13.8 could not be downloaded here, "
                   "so its Node.js package (same release) was tested")
    (data.RESULTS / "sherpa_1x_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
