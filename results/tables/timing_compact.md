The 1-LSB noise-floor configuration uses the default model's graph and front end; its timing row is the default model's (adding the noise was not timed separately).

| model | timing key | CPU ms, 1 s | CPU ms, 2 s | CPU ms, 3 s | CPU ms, 4.3 s sentence | CPU ms, 8.7 s sentence | file MB | dim |
|---|---|---|---|---|---|---|---|---|
| 3D-Speaker CAM++ (NexaRoom default) | campplus_zhen_adv | 22.5 | 37.7 | 54.3 | 77.4 | 155.0 | 28.3 | 192 |
| 3D-Speaker ERes2Net-base | eres2net_base_zh | 52.2 | 104.8 | 159.5 | 235.5 | 499.9 | 39.6 | 512 |
| NeMo TitaNet-small | titanet_small | 18.0 | 31.8 | 45.8 | 65.2 | 133.9 | 40.3 | 192 |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | wespeaker_campplus_vox_wsfe | 22.0 | 37.2 | 54.9 | 75.5 | 159.2 | 29.3 | 512 |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | wespeaker_resnet34_vox_wsfe | 50.4 | 98.3 | 147.9 | 209.8 | 430.5 | 26.5 | 256 |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | wespeaker_resnet34_cnc_wsfe | 51.0 | 98.3 | 147.9 | 209.9 | 438.1 | 26.5 | 256 |
| 3D-Speaker CAM++, 1-LSB noise floor | campplus_zhen_adv | 22.5 | 37.7 | 54.3 | 77.4 | 155.0 | 28.3 | 192 |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | campplus_zhen_adv_ort131 | 22.3 | 38.5 | 55.0 | 75.2 | 162.1 | 28.3 | 192 |
| WeSpeaker CAM++ VoxCeleb, sherpa-onnx as shipped | wespeaker_campplus_vox | 21.7 | 37.3 | 54.2 | 76.6 | 152.2 | 29.3 | 512 |
