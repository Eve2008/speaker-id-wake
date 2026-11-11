"""List the operator types (and opset) of the speaker ONNX graphs.

A minimal protobuf wire-format reader (the onnx package is not installed):
ModelProto.opset_import (8) -> OperatorSetIdProto.version (2);
ModelProto.graph (7) -> GraphProto.node (1) -> NodeProto.op_type (4),
NodeProto.attribute (5) -> AttributeProto.name (1), .i (3), .ints (8).
Used to list the operators that the WeSpeaker CAM++ graph uses and the
ResNet34 graphs do not (candidates for the onnxruntime-version difference in
results/ort_version_check.json).

Output: results/onnx_ops.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swk import data  # noqa: E402


def varint(b: bytes, i: int) -> tuple[int, int]:
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return r, i


def fields(b: bytes):
    i = 0
    while i < len(b):
        key, i = varint(b, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = varint(b, i)
        elif wt == 1:
            v, i = b[i:i + 8], i + 8
        elif wt == 2:
            n, i = varint(b, i)
            v, i = b[i:i + n], i + n
        elif wt == 5:
            v, i = b[i:i + 4], i + 4
        else:
            raise ValueError(f"wire type {wt}")
        yield f, wt, v


def describe(path: Path) -> dict:
    raw = path.read_bytes()
    ops: Counter = Counter()
    opset = {}
    pools = []
    for f, wt, v in fields(raw):
        if f == 8 and wt == 2:
            dom, ver = "", None
            for g, _, x in fields(v):
                if g == 1:
                    dom = x.decode()
                elif g == 2:
                    ver = x
            opset[dom or "ai.onnx"] = ver
        if f == 7 and wt == 2:
            for g, gwt, node in fields(v):
                if g != 1 or gwt != 2:
                    continue
                op, attrs = None, {}
                for h, hwt, x in fields(node):
                    if h == 4:
                        op = x.decode()
                    elif h == 5 and hwt == 2:
                        name, val = None, None
                        for a, awt, y in fields(x):
                            if a == 1:
                                name = y.decode()
                            elif a == 3 and awt == 0:
                                val = y
                            elif a == 8:
                                val = (val or []) + ([y] if awt == 0 else [])
                        attrs[name] = val
                ops[op] += 1
                if op and "Pool" in op:
                    pools.append({"op": op, **{k: v for k, v in attrs.items() if k in
                                               ("ceil_mode", "count_include_pad", "kernel_shape", "strides", "pads")}})
    return {"opset": opset, "ops": dict(sorted(ops.items())), "pool_nodes": pools}


def main() -> None:
    out = {}
    for key in ("wespeaker_campplus_vox", "wespeaker_resnet34_vox", "wespeaker_resnet34_cnc", "campplus_zhen_adv"):
        out[key] = describe(data.model_path(key))
    only = sorted(set(out["wespeaker_campplus_vox"]["ops"]) - set(out["wespeaker_resnet34_vox"]["ops"])
                  - set(out["wespeaker_resnet34_cnc"]["ops"]))
    out["ops_in_wespeaker_campplus_not_in_resnet34"] = only
    with open(data.RESULTS / "onnx_ops.json", "w") as f:
        json.dump(out, f, indent=1)
    for k in ("wespeaker_campplus_vox", "campplus_zhen_adv"):
        print(k, out[k]["opset"], out[k]["ops"])
        print("  pools:", out[k]["pool_nodes"][:6], "...", len(out[k]["pool_nodes"]))
    print("only in WeSpeaker CAM++:", only)


if __name__ == "__main__":
    main()
