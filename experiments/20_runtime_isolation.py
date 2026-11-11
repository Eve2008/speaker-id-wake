"""Isolate the runtime-version effect on NexaRoom's CAM++ graph (revision).

Review point: campplus_zhen_adv_ort131 changes the front end (NumPy instead of
sherpa-onnx's C++) and the runtime (1.31.1 instead of the bundled 1.28.2) at
once, and the AveragePool explanation rested on a one-node toy graph run on
1.27.0 / 1.29.0 / 1.31.1, never on 1.28.2. This script

1. writes a patched copy of the CAM++ file in which the 52 AveragePool nodes
   have count_include_pad = 0 instead of 1 (one protobuf byte per node, the
   file length is unchanged; data/models/patched/);
2. validates the ctypes C-API wrapper (swk/ortc.py): with the 1.31.1 library of
   the onnxruntime wheel it must reproduce the Python package;
3. runs the one-node AveragePool graph of 17_avgpool_check.py on the 1.28.2
   library bundled with sherpa-onnx 2.0.1;
4. for 20 random 1 s clips and 20 random 3 s utterances (three clips joined),
   feeds IDENTICAL NumPy features to every graph on 1.28.2 (ctypes) and 1.31.1
   (Python), with the original and the patched CAM++ graph, and also runs
   sherpa-onnx 2.0.1 itself (its own C++ front end, bundled 1.28.2) on the
   original and the patched file. Reports the median cosine for each pair.

Output: results/runtime_isolation.json
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from swk import data, frontend  # noqa: E402
from swk.ortc import PY_ORT, SHERPA_ORT, OrtC  # noqa: E402

PATCH_DIR = data.MODELS_DIR / "patched"
CAMPP = data.MODELS_DIR / data.MODELS["campplus_zhen_adv"].filename
CAMPP_CIP0 = data.MODELS_DIR / data.MODELS["campplus_zhen_adv_cip0"].filename
PAT_OLD = b"\x0a\x11count_include_pad\x18\x01"
PAT_NEW = b"\x0a\x11count_include_pad\x18\x00"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make_patched() -> dict:
    raw = CAMPP.read_bytes()
    n = raw.count(PAT_OLD)
    assert n == 52, n
    PATCH_DIR.mkdir(parents=True, exist_ok=True)
    new = raw.replace(PAT_OLD, PAT_NEW)
    assert len(new) == len(raw) and new.count(PAT_NEW) == 52
    if not CAMPP_CIP0.exists() or CAMPP_CIP0.read_bytes() != new:
        CAMPP_CIP0.write_bytes(new)
    return {"source": CAMPP.name, "source_sha256": sha(CAMPP), "patched": str(CAMPP_CIP0.relative_to(data.MODELS_DIR)),
            "patched_sha256": sha(CAMPP_CIP0), "nodes_patched": n,
            "change": "AveragePool attribute count_include_pad 1 -> 0 (protobuf varint 0x01 -> 0x00)"}


def toy_avgpool() -> dict:
    spec = importlib.util.spec_from_file_location("ap", ROOT / "experiments" / "17_avgpool_check.py")
    ap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ap)
    out = {}
    tmp = data.CACHE / "tmp_avgpool"
    tmp.mkdir(exist_ok=True)
    for cip in (None, 1):
        mp = tmp / f"pool_{cip}.onnx"
        mp.write_bytes(ap.model_bytes(cip))
        name = "count_include_pad=absent(0)" if cip is None else "count_include_pad=1"
        res = {}
        for L in ap.LENGTHS:
            k = -(-L // 100)
            row = {}
            for label, lib in (("ort_1.28.2_sherpa_bundled", SHERPA_ORT), ("ort_1.31.1_ctypes", PY_ORT)):
                r = OrtC(lib, mp, "x", "y", k, out_shape=(1, 1, k))
                row[label] = [round(float(v), 4) for v in r.run(np.ones((1, 1, L), np.float32))]
            res[f"L={L}"] = row
        out[name] = res
    return out


def main() -> None:
    import onnxruntime as ort
    import sherpa_onnx

    res: dict = {"onnxruntime_python": ort.__version__, "sherpa_onnx": sherpa_onnx.__version__}
    res["patch"] = make_patched()
    print(res["patch"], flush=True)
    res["avgpool_toy"] = toy_avgpool()
    print(json.dumps(res["avgpool_toy"]), flush=True)

    meta = data.load_meta()
    rng = np.random.default_rng(3)  # same 20 clips as 13_ort_version_check.py
    files = meta.file.to_numpy()[rng.choice(len(meta), 20, replace=False)]
    utts = {"1s": [data.load_clip_int16(f) for f in files]}
    rng3 = np.random.default_rng(4)
    f3 = meta.file.to_numpy()[rng3.choice(len(meta), 60, replace=False)].reshape(20, 3)
    utts["3s"] = [np.concatenate([data.load_clip_int16(f) for f in row]) for row in f3]
    nrm = lambda v: np.asarray(v, dtype=np.float64) / np.linalg.norm(v)  # noqa: E731

    graphs = {"campplus_zhen_adv": CAMPP, "campplus_zhen_adv_cip0": CAMPP_CIP0,
              "eres2net_base_zh": data.model_path("eres2net_base_zh"), "titanet_small": data.model_path("titanet_small"),
              "wespeaker_campplus_vox": data.model_path("wespeaker_campplus_vox"),
              "wespeaker_resnet34_vox": data.model_path("wespeaker_resnet34_vox"),
              "wespeaker_resnet34_cnc": data.model_path("wespeaker_resnet34_cnc")}
    emb: dict = {}
    valid = []
    for key, path in graphs.items():
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        md = sess.get_modelmeta().custom_metadata_map
        ins = [i.name for i in sess.get_inputs()]
        out_name = "embs" if key == "titanet_small" else sess.get_outputs()[0].name
        dim = int(md.get("output_dim", sess.get_outputs()[0].shape[-1]))
        nemo = md.get("framework", "") == "nemo"
        scale = 1.0 / 32768.0 if md.get("normalize_samples", "1") == "1" else 1.0
        cmn = md.get("feature_normalize_type", "") == "global-mean"
        c128 = OrtC(SHERPA_ORT, path, ins, out_name, dim)
        c131 = OrtC(PY_ORT, path, ins, out_name, dim)
        ex = None if nemo else sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(path), num_threads=1, provider="cpu"))
        for ln, us in utts.items():
            e = {"py_1.31.1": [], "ctypes_1.28.2": [], "ctypes_1.31.1": [], "sherpa_onnx_2.0.1": []}
            for u in us:
                f = frontend.fbank(u.astype(np.float64) * scale, window="povey", high=-400.0)
                f = (f - f.mean(axis=0, keepdims=True) if cmn else f).astype(np.float32)
                if nemo:  # identical (not NeMo-style) features to both runtimes: (1, 80, T) + length
                    feeds = [f.T[None].copy(), np.array([f.shape[0]], np.int64)]
                    pyo = sess.run([out_name], dict(zip(ins, feeds)))[0][0]
                else:
                    feeds = [f[None]]
                    pyo = sess.run(None, {ins[0]: feeds[0]})[0][0]
                e["py_1.31.1"].append(nrm(pyo))
                e["ctypes_1.28.2"].append(nrm(c128.run(feeds)))
                y131 = c131.run(feeds)
                valid.append(float(np.max(np.abs(y131 - pyo)) / max(np.max(np.abs(pyo)), 1e-9)))
                e["ctypes_1.31.1"].append(nrm(y131))
                if ex is not None:
                    st = ex.create_stream()
                    st.accept_waveform(sample_rate=16000, waveform=u.astype(np.float32) / 32768.0)
                    st.input_finished()
                    e["sherpa_onnx_2.0.1"].append(nrm(ex.compute(st)))
            emb[(key, ln)] = {k: v for k, v in e.items() if v}
        print(key, "done", flush=True)
    res["ctypes_validation"] = {"max_rel_abs_diff_ctypes_1.31.1_vs_python_1.31.1": max(valid), "n_runs": len(valid)}

    med = lambda a, b: float(np.median([x @ y for x, y in zip(a, b)]))  # noqa: E731
    agree: dict = {}
    for (key, ln), e in emb.items():
        d = agree.setdefault(key, {})
        d[f"{ln}: ort 1.28.2 vs 1.31.1 (same features)"] = med(e["ctypes_1.28.2"], e["py_1.31.1"])
        if "sherpa_onnx_2.0.1" in e:
            d[f"{ln}: sherpa-onnx vs ort 1.28.2 (front end only)"] = med(e["sherpa_onnx_2.0.1"], e["ctypes_1.28.2"])
            d[f"{ln}: sherpa-onnx vs ort 1.31.1 (front end + runtime)"] = med(e["sherpa_onnx_2.0.1"], e["py_1.31.1"])
    for ln in utts:
        o, p = emb[("campplus_zhen_adv", ln)], emb[("campplus_zhen_adv_cip0", ln)]
        agree["campplus_zhen_adv_cip0"].update({
            f"{ln}: patched on 1.28.2 vs original on 1.31.1 (same features)": med(p["ctypes_1.28.2"], o["py_1.31.1"]),
            f"{ln}: patched vs original, both on 1.31.1": med(p["py_1.31.1"], o["py_1.31.1"]),
            f"{ln}: patched vs original, both on 1.28.2": med(p["ctypes_1.28.2"], o["ctypes_1.28.2"]),
            f"{ln}: sherpa-onnx patched vs sherpa-onnx original": med(p["sherpa_onnx_2.0.1"], o["sherpa_onnx_2.0.1"]),
        })
    res["n_utterances"] = {k: len(v) for k, v in utts.items()}
    res["features"] = ("NumPy fbank with sherpa-onnx conventions (Povey window, 20-7600 Hz, sample scale and CMN "
                       "from the model metadata); TitaNet gets the same features transposed, not NeMo features, "
                       "so only its runtime comparison is meaningful")
    res["median_cosine"] = agree
    with open(data.RESULTS / "runtime_isolation.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(agree, indent=1))


if __name__ == "__main__":
    main()
