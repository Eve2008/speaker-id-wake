"""File size, embedding dimension and checksum of the model files added in the revision.

The two 3D-Speaker VoxCeleb models come from the same sherpa-onnx release as the
others; their SHA-256 sums are compared with data/models/speaker_models_checksum.txt.
The patched CAM++ files (count_include_pad 1 -> 0, experiments/20_runtime_isolation.py)
are listed with their own sums.

Output: results/models_rev.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swk import data  # noqa: E402

KEYS = ["campplus_vox_cip0_d1", "eres2net_vox_d1", "campplus_zhen_adv_cip0"]
SOURCES = {"campplus_vox_cip0_d1": "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
           "eres2net_vox_d1": "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx",
           "campplus_zhen_adv_cip0": "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"}


def main() -> None:
    import onnxruntime as ort

    ref = dict(l.split() for l in (data.MODELS_DIR / "speaker_models_checksum.txt").read_text().splitlines() if l.strip())
    out = {}
    for k in KEYS:
        p = data.model_path(k)
        src = data.MODELS_DIR / SOURCES[k]
        s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        md = s.get_modelmeta().custom_metadata_map
        out[k] = {"file": str(p.relative_to(data.MODELS_DIR)), "file_mb": p.stat().st_size / 1e6,
                  "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "dim": int(md.get("output_dim", 0)),
                  "release_file": src.name, "release_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                  "release_sha256_matches_checksum_file": hashlib.sha256(src.read_bytes()).hexdigest() == ref.get(src.name),
                  "metadata": md}
    with open(data.RESULTS / "models_rev.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: {a: v[a] for a in ("file_mb", "dim", "release_sha256_matches_checksum_file")} for k, v in out.items()}))


if __name__ == "__main__":
    main()
