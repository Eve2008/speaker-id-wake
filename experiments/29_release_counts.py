"""Count the ONNX models in the sherpa-onnx speaker-recognition release (revision, 11 Nov).

Reads the release asset list saved from api.github.com (results/release_assets_rev.json) and
the release checksum file (data/models/speaker_models_checksum.txt if present) and writes
results/release_counts.json: number of .onnx models, how many are <= 40.3 MB (the largest
evaluated file) and the sizes of the larger ones.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swk import data  # noqa: E402

LIMIT_MB = 40.3


def main() -> None:
    rel = json.loads((data.RESULTS / "release_assets_rev.json").read_text())
    onnx = sorted(((a["name"], a["size"] / 1e6) for a in rel["assets"] if a["name"].endswith(".onnx")),
                  key=lambda t: t[1])
    small = [(n, round(s, 1)) for n, s in onnx if round(s, 1) <= LIMIT_MB]
    large = [(n, round(s, 1)) for n, s in onnx if round(s, 1) > LIMIT_MB]
    out = {"release_tag": rel["tag_name"], "n_assets": len(rel["assets"]), "n_onnx": len(onnx),
           "n_onnx_le_40.3MB": len(small), "n_onnx_gt_40.3MB": len(large),
           "small": small, "large": large}
    ck = Path("/home/ali/Desktop/Articles/data/models/speaker_models_checksum.txt")
    if ck.exists():
        out["n_onnx_in_checksum_file"] = sum(1 for ln in ck.read_text().splitlines() if ln.split("\t")[0].strip().endswith(".onnx"))
    (data.RESULTS / "release_counts.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, list)}, indent=1))


if __name__ == "__main__":
    main()
