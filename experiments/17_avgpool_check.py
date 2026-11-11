"""Does onnxruntime change AveragePool(ceil_mode=1, count_include_pad=0) between versions?

The WeSpeaker CAM++ graph has 52 AveragePool nodes with kernel 100, stride 100,
ceil_mode=1 and no count_include_pad (default 0); the 3D-Speaker CAM++ graph
has the same nodes with count_include_pad=1 (results/onnx_ops.json). This
script writes a one-node ONNX model with that AveragePool (protobuf written by
hand, no onnx package) for both count_include_pad values, runs it on inputs of
ones of length L (shape 1 x 1 x L) with onnxruntime Python and onnxruntime-node
1.27.0 / 1.29.0 / 1.31.1 and with PyTorch avg_pool1d, and records the outputs. By the ONNX definition,
with count_include_pad=0 every output of an all-ones input is 1.

Usage: python experiments/17_avgpool_check.py NODE_PREFIX
Output: results/avgpool_check.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
from swk import data  # noqa: E402

NODE = {"node_ort_1.27.0": "v127", "node_ort_1.29.0": "v129", "node_ort_1.31.1": "v131"}
LENGTHS = (49, 100, 150, 250)

JS = r"""
const [modDir, model, L, out] = process.argv.slice(2);
const ort = require(modDir); const fs = require('fs');
(async () => {
  const n = parseInt(L); const x = new Float32Array(n).fill(1);
  const s = await ort.InferenceSession.create(model, { intraOpNumThreads: 1 });
  const r = await s.run({ x: new ort.Tensor('float32', x, [1, 1, n]) });
  fs.writeFileSync(out, JSON.stringify(Array.from(r.y.data)));
})();
"""


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _ld(f: int, p: bytes) -> bytes:
    return _varint((f << 3) | 2) + _varint(len(p)) + p


def _vi(f: int, v: int) -> bytes:
    return _varint(f << 3) + _varint(v)


def _attr_int(name: str, v: int) -> bytes:
    return _ld(5, _ld(1, name.encode()) + _vi(3, v) + _vi(20, 2))


def _attr_ints(name: str, vs) -> bytes:
    return _ld(5, _ld(1, name.encode()) + b"".join(_vi(8, v) for v in vs) + _vi(20, 7))


def _value_info(name: str) -> bytes:
    dims = b"".join(_ld(1, _ld(2, d.encode())) for d in ("n", "c", "l"))
    tensor = _vi(1, 1) + _ld(2, dims)  # elem_type FLOAT, shape
    return _ld(1, name.encode()) + _ld(2, _ld(1, tensor))


def model_bytes(count_include_pad: int | None) -> bytes:
    attrs = _attr_ints("kernel_shape", [100]) + _attr_ints("strides", [100]) + _attr_int("ceil_mode", 1) \
        + _attr_ints("pads", [0, 0])
    if count_include_pad is not None:
        attrs += _attr_int("count_include_pad", count_include_pad)
    node = _ld(1, b"x") + _ld(2, b"y") + _ld(4, b"AveragePool") + attrs
    graph = _ld(1, node) + _ld(2, b"g") + _ld(11, _value_info("x")) + _ld(12, _value_info("y"))
    return _vi(1, 7) + _ld(8, _ld(1, b"") + _vi(2, 14)) + _ld(7, graph)


def main(node_prefix: str) -> None:
    import onnxruntime as ort

    tmp = Path(tempfile.mkdtemp())
    js = tmp / "pool.js"
    js.write_text(JS)
    out = {"onnxruntime_python": ort.__version__, "kernel": 100, "stride": 100, "ceil_mode": 1, "input": "ones",
           "cases": {}}
    for cip in (None, 1):
        name = "count_include_pad=absent(0)" if cip is None else "count_include_pad=1"
        mp = tmp / f"pool_{cip}.onnx"
        mp.write_bytes(model_bytes(cip))
        res = {}
        sess = ort.InferenceSession(str(mp), providers=["CPUExecutionProvider"])
        for L in LENGTHS:
            row = {"python_ort": [round(float(v), 4) for v in
                                  sess.run(None, {"x": np.ones((1, 1, L), np.float32)})[0].ravel()]}
            for label, sub in NODE.items():
                o = tmp / "o.json"
                subprocess.run(["node", str(js), str(Path(node_prefix) / sub / "node_modules" / "onnxruntime-node"),
                                str(mp), str(L), str(o)], check=True)
                row[label] = [round(float(v), 4) for v in json.loads(o.read_text())]
            try:  # reference: PyTorch, in which the model was trained
                import torch
                import torch.nn.functional as F

                row[f"torch_{torch.__version__}"] = [round(float(v), 4) for v in F.avg_pool1d(
                    torch.ones(1, 1, L), 100, 100, ceil_mode=True,
                    count_include_pad=bool(cip if cip is not None else 0)).flatten()]
            except ImportError:
                pass
            res[f"L={L}"] = row
        out["cases"][name] = res
        print(name, json.dumps(res, indent=1))
    with open(data.RESULTS / "avgpool_check.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
