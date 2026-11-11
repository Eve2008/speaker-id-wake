Bin upper edges of 0.0005-wide histograms, pooled over seeds and eval speakers.

| model | test | target cos q05 | target cos median | non-target cos median | non-target cos q95 | non-target cos q99 | non-target cos q99.9 |
|---|---|---|---|---|---|---|---|
| 3D-Speaker CAM++ (NexaRoom default) | 1 s | 0.3255 | 0.5455 | 0.215 | 0.3955 | 0.472 | 0.5525 |
| 3D-Speaker CAM++ (NexaRoom default) | 2 s | 0.496 | 0.6755 | 0.254 | 0.438 | 0.515 | 0.594 |
| 3D-Speaker CAM++ (NexaRoom default) | 3 s | 0.556 | 0.7255 | 0.267 | 0.4585 | 0.5375 | 0.618 |
| 3D-Speaker ERes2Net-base | 1 s | 0.4345 | 0.6505 | 0.266 | 0.4735 | 0.5515 | 0.6285 |
| 3D-Speaker ERes2Net-base | 2 s | 0.5835 | 0.754 | 0.3025 | 0.5145 | 0.5905 | 0.6635 |
| 3D-Speaker ERes2Net-base | 3 s | 0.658 | 0.803 | 0.3205 | 0.5355 | 0.6105 | 0.6815 |
| NeMo TitaNet-small | 1 s | 0.2405 | 0.478 | 0.154 | 0.355 | 0.4405 | 0.5345 |
| NeMo TitaNet-small | 2 s | 0.425 | 0.6265 | 0.1835 | 0.4015 | 0.4915 | 0.587 |
| NeMo TitaNet-small | 3 s | 0.5105 | 0.694 | 0.197 | 0.423 | 0.5145 | 0.61 |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | 1 s | 0.257 | 0.488 | 0.183 | 0.3745 | 0.4575 | 0.5475 |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | 2 s | 0.42 | 0.6215 | 0.219 | 0.4255 | 0.51 | 0.5965 |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | 3 s | 0.514 | 0.688 | 0.2415 | 0.4545 | 0.538 | 0.6225 |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | 1 s | 0.3215 | 0.5315 | 0.2085 | 0.4005 | 0.486 | 0.5825 |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | 2 s | 0.475 | 0.6505 | 0.2455 | 0.443 | 0.5255 | 0.6115 |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | 3 s | 0.555 | 0.7085 | 0.2635 | 0.464 | 0.546 | 0.6295 |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | 1 s | 0.4195 | 0.624 | 0.272 | 0.4855 | 0.5655 | 0.6435 |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | 2 s | 0.5675 | 0.729 | 0.309 | 0.5305 | 0.6095 | 0.6835 |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | 3 s | 0.638 | 0.7785 | 0.3285 | 0.555 | 0.6325 | 0.705 |
| 3D-Speaker CAM++, 1-LSB noise floor | 1 s | 0.3345 | 0.551 | 0.207 | 0.392 | 0.47 | 0.5515 |
| 3D-Speaker CAM++, 1-LSB noise floor | 2 s | 0.495 | 0.6745 | 0.2335 | 0.424 | 0.5035 | 0.5855 |
| 3D-Speaker CAM++, 1-LSB noise floor | 3 s | 0.558 | 0.726 | 0.2475 | 0.4465 | 0.5275 | 0.61 |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | 1 s | 0.355 | 0.5595 | 0.23 | 0.403 | 0.4765 | 0.5545 |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | 2 s | 0.495 | 0.6765 | 0.256 | 0.439 | 0.5155 | 0.5945 |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | 3 s | 0.573 | 0.736 | 0.2755 | 0.462 | 0.539 | 0.618 |
| WeSpeaker CAM++ VoxCeleb, sherpa-onnx as shipped | 1 s | 0.231 | 0.285 | 0.2845 | 0.354 | 0.3935 | 0.444 |
| WeSpeaker CAM++ VoxCeleb, sherpa-onnx as shipped | 2 s | -0.0175 | 0.0885 | 0.083 | 0.1905 | 0.233 | 0.274 |
| WeSpeaker CAM++ VoxCeleb, sherpa-onnx as shipped | 3 s | 0.4935 | 0.5535 | 0.55 | 0.604 | 0.629 | 0.675 |
