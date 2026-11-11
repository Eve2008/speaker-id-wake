"""Make CMN-flagged copies of the three WeSpeaker ONNX models.

Why: sherpa-onnx (speaker-embedding-extractor-general-impl.h) subtracts the
per-utterance mean of the fbank features only when the model metadata has
feature_normalize_type == "global-mean". The 3D-Speaker files carry that
entry; the WeSpeaker files in the sherpa-onnx release do not, so sherpa-onnx
feeds them un-normalised log-mel features. WeSpeaker's own inference code
(wespeaker/cli/speaker.py, compute_features) subtracts the feature mean
(CMN), and the ONNX graphs do not do it internally (checked in
check_cmn_models(): the output changes when a constant is added to all frames).

What: the copy is the original file byte for byte plus one extra
ModelProto.metadata_props entry (field 14) appended at the end:
feature_normalize_type = global-mean. Protobuf merges repeated fields that
appear later in the stream, so the weights and graph are unchanged.

Output: /home/ali/Desktop/Articles/data/models/cmn/<name>_cmn.onnx and
results/cmn_models.json (source and output SHA-256, checks).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swk import data  # noqa: E402

SOURCES = {
    "wespeaker_campplus_vox_cmn": "wespeaker_en_voxceleb_CAM++.onnx",
    "wespeaker_resnet34_vox_cmn": "wespeaker_en_voxceleb_resnet34.onnx",
    "wespeaker_resnet34_cnc_cmn": "wespeaker_zh_cnceleb_resnet34.onnx",
}


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _ld(field: int, payload: bytes) -> bytes:
    """Length-delimited protobuf field."""

    return _varint((field << 3) | 2) + _varint(len(payload)) + payload


def metadata_entry(key: str, value: str) -> bytes:
    entry = _ld(1, key.encode()) + _ld(2, value.encode())  # StringStringEntryProto
    return _ld(14, entry)  # ModelProto.metadata_props


def check_cmn_models(src: Path, dst: Path) -> dict:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    a = ort.InferenceSession(str(src), so, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(dst), so, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((1, 150, 80)) * 2 + 10).astype(np.float32)
    c = (rng.standard_normal((1, 1, 80)) * 3).astype(np.float32)
    name = a.get_inputs()[0].name
    ya, yb = a.run(None, {name: x})[0][0], b.run(None, {name: x})[0][0]
    yc = a.run(None, {name: x + c})[0][0]

    def cos(u, v):
        return float(u @ v / np.linalg.norm(u) / np.linalg.norm(v))

    return {"metadata": dict(b.get_modelmeta().custom_metadata_map),
            "same_output_as_source_on_random_features": bool(np.allclose(ya, yb, atol=1e-6)),
            "graph_is_offset_invariant_cos": cos(ya, yc)}


def main() -> None:
    out_dir = data.MODELS_DIR / "cmn"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for key, fname in SOURCES.items():
        src = data.MODELS_DIR / fname
        raw = src.read_bytes()
        dst = out_dir / fname.replace(".onnx", "_cmn.onnx")
        dst.write_bytes(raw + metadata_entry("feature_normalize_type", "global-mean"))
        chk = check_cmn_models(src, dst)
        report[key] = {"source": fname, "source_sha256": hashlib.sha256(raw).hexdigest(),
                       "output": str(dst.relative_to(data.MODELS_DIR)),
                       "output_sha256": hashlib.sha256(dst.read_bytes()).hexdigest(), **chk}
        print(key, json.dumps(report[key], indent=1))
    with open(data.RESULTS / "cmn_models.json", "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
