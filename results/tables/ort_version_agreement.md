Cosine between embeddings of the same 20 clips from identical input features (sherpa-onnx column: sherpa-onnx's own front end).

| model | ort 1.27.0 vs 1.29.0 | ort 1.29.0 vs 1.31.1 | ort 1.27.0 vs sherpa-onnx 2.0.1 | ort 1.31.1 vs sherpa-onnx 2.0.1 |
|---|---|---|---|---|
| wespeaker_campplus_vox | -0.045 | 1.0 | 0.985 | -0.046 |
| wespeaker_resnet34_vox | 1.0 | 1.0 | 0.977 | 0.977 |
| wespeaker_resnet34_cnc | 1.0 | 1.0 | 0.975 | 0.975 |
| campplus_zhen_adv | 0.887 | 1.0 | 0.971 | 0.864 |
| eres2net_base_zh | 1.0 | 1.0 | 0.974 | 0.974 |
