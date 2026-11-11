"""Pooling nodes of the graphs not covered by 16_onnx_ops.py (revision): ERes2Net-base, TitaNet-small
and the two 3D-Speaker VoxCeleb models. Same protobuf reader as 16_onnx_ops.py.
Output: results/onnx_ops_rev.json"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swk import data  # noqa: E402

spec = importlib.util.spec_from_file_location("ops", Path(__file__).resolve().parent / "16_onnx_ops.py")
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)
out = {}
for k in ("eres2net_base_zh", "titanet_small", "campplus_vox_cip0_d1", "eres2net_vox_d1", "campplus_zhen_adv_cip0"):
    d = ops.describe(data.model_path(k))
    pools = d["pool_nodes"]
    out[k] = {"opset": d["opset"], "n_pool_nodes": len(pools),
              "pool_types": sorted({p["op"] for p in pools}),
              "averagepool_count_include_pad": sorted({p.get("count_include_pad", 0) or 0 for p in pools
                                                      if p["op"] == "AveragePool"}),
              "ops": d["ops"]}
with open(data.RESULTS / "onnx_ops_rev.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: {a: v[a] for a in ("n_pool_nodes", "pool_types", "averagepool_count_include_pad")} for k, v in out.items()}))
