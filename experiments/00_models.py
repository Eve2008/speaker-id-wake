"""Record the speaker models and model configurations used: file, size, SHA-256
(checked against the release checksum file), embedding dimension, inference
backend and dither. Output: results/models.json

Model keys:
  as-shipped sherpa-onnx path: campplus_zhen_adv, eres2net_base_zh, titanet_small, wespeaker_campplus_vox,
      wespeaker_resnet34_vox, wespeaker_resnet34_cnc
  *_cmn: WeSpeaker files with feature_normalize_type=global-mean appended (08_make_cmn_models.py), sherpa-onnx;
      diagnostic only (09, 11)
  *_wsfe: as-shipped WeSpeaker files run with WeSpeaker's own front end + onnxruntime (swk/frontend.py)
  campplus_zhen_adv_d1: NexaRoom's default model with a 1-LSB noise floor, sherpa-onnx
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swk import data  # noqa: E402

RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models"


def bundled_ort_version(sherpa_onnx) -> str | None:
    lib = Path(sherpa_onnx.__file__).parent / "lib" / "libonnxruntime.so"
    if not lib.exists():
        return None
    m = re.findall(rb"VERS_(1\.\d+\.\d+)", lib.read_bytes())
    return m[-1].decode() if m else None


def main() -> None:
    import onnxruntime as ort
    import sherpa_onnx

    sums = {}
    for line in (data.MODELS_DIR / "speaker_models_checksum.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            sums[parts[0]] = parts[1]
    runtime = {"sherpa_onnx": sherpa_onnx.__version__, "sherpa_onnx_bundled_onnxruntime": bundled_ort_version(sherpa_onnx),
               "onnxruntime_python": ort.__version__}
    out = []
    for key, info in data.MODELS.items():
        p = data.model_path(key)
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if info.backend == "wespeaker_ort":
            from swk.frontend import WeSpeakerOrtExtractor

            dim = WeSpeakerOrtExtractor(str(p)).dim
        else:
            cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(p), num_threads=1, provider="cpu")
            dim = sherpa_onnx.SpeakerEmbeddingExtractor(cfg).dim
        out.append({"key": key, "file": info.filename, "label": info.label, "family": info.family,
                    "training_data_as_named": info.training_data, "bytes": p.stat().st_size, "sha256": h,
                    "sha256_matches_release_checksum": sums.get(Path(info.filename).name) == h,
                    "embedding_dim": dim, "backend": info.backend, "dither_lsb": info.dither,
                    "runtime": runtime["sherpa_onnx"] if info.backend == "sherpa" else
                    f"onnxruntime {runtime['onnxruntime_python']} + swk/frontend.py",
                    "source": RELEASE})
    with open(data.RESULTS / "models.json", "w") as f:
        json.dump({"runtime": runtime, "models": out}, f, indent=1)
    print(json.dumps(runtime))
    for o in out:
        print(o["key"], o["embedding_dim"], o["bytes"], o["sha256_matches_release_checksum"], o["backend"], o["dither_lsb"])


if __name__ == "__main__":
    main()
