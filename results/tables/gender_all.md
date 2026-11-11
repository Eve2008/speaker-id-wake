N = 5. Same-gender household: the 4 competitors (members) or 5 enrolled speakers (strangers) share the test speaker's gender label; cross: all have the other label.

| model | test | closed-set acc. N=5, same-gender household % | closed-set acc. N=5, random household % | closed-set acc. N=5, cross-gender household % | NexaRoom rule stranger FA, same-gender % | NexaRoom rule stranger FA, cross-gender % |
|---|---|---|---|---|---|---|
| 3D-Speaker CAM++ (NexaRoom default) | 1 s | 92.6 [92.1, 93.1] | 93.9 [93.5, 94.4] | 97.3 [96.9, 97.6] | 0.45 [0.42, 0.50] | 0.03 [0.02, 0.03] |
| 3D-Speaker CAM++ (NexaRoom default) | 2 s | 98.9 [98.7, 99.0] | 99.1 [98.9, 99.3] | 99.6 [99.5, 99.8] | 5.55 [5.25, 5.87] | 0.44 [0.38, 0.50] |
| 3D-Speaker CAM++ (NexaRoom default) | 3 s | 99.4 [99.3, 99.6] | 99.5 [99.4, 99.7] | 99.8 [99.6, 99.9] | 8.86 [8.39, 9.35] | 0.65 [0.57, 0.75] |
| 3D-Speaker ERes2Net-base | 1 s | 96.0 [95.6, 96.4] | 96.8 [96.4, 97.1] | 98.6 [98.3, 98.8] | 3.68 [3.47, 3.93] | 0.24 [0.20, 0.28] |
| 3D-Speaker ERes2Net-base | 2 s | 99.1 [98.9, 99.3] | 99.3 [99.1, 99.5] | 99.7 [99.5, 99.8] | 18.23 [17.61, 18.88] | 1.72 [1.53, 1.91] |
| 3D-Speaker ERes2Net-base | 3 s | 99.6 [99.4, 99.7] | 99.6 [99.5, 99.8] | 99.9 [99.8, 100.0] | 23.26 [22.53, 24.01] | 2.41 [2.16, 2.65] |
| NeMo TitaNet-small | 1 s | 89.2 [88.6, 89.8] | 91.0 [90.4, 91.6] | 96.1 [95.6, 96.5] | 0.40 [0.36, 0.45] | 0.04 [0.03, 0.05] |
| NeMo TitaNet-small | 2 s | 97.6 [97.2, 97.9] | 98.1 [97.7, 98.4] | 99.3 [99.1, 99.5] | 4.37 [4.04, 4.72] | 0.42 [0.35, 0.52] |
| NeMo TitaNet-small | 3 s | 99.1 [98.8, 99.3] | 99.3 [99.1, 99.5] | 99.6 [99.4, 99.8] | 6.67 [6.20, 7.17] | 0.63 [0.53, 0.74] |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | 1 s | 88.4 [87.8, 89.0] | 90.4 [89.8, 90.9] | 96.2 [95.8, 96.5] | 0.64 [0.59, 0.70] | 0.07 [0.06, 0.09] |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | 2 s | 96.4 [95.9, 96.8] | 97.2 [96.8, 97.5] | 99.2 [98.9, 99.4] | 5.95 [5.67, 6.28] | 0.62 [0.56, 0.69] |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | 3 s | 98.4 [98.0, 98.7] | 98.7 [98.4, 99.0] | 99.5 [99.3, 99.7] | 9.57 [9.11, 10.07] | 0.97 [0.87, 1.07] |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | 1 s | 91.0 [90.5, 91.5] | 92.4 [91.9, 92.9] | 96.7 [96.3, 97.0] | 0.98 [0.90, 1.07] | 0.18 [0.15, 0.21] |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | 2 s | 97.5 [97.1, 97.8] | 98.0 [97.6, 98.3] | 99.3 [99.1, 99.5] | 6.67 [6.31, 7.05] | 0.76 [0.65, 0.88] |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | 3 s | 99.1 [98.9, 99.3] | 99.2 [99.0, 99.4] | 99.7 [99.5, 99.8] | 9.75 [9.27, 10.24] | 1.04 [0.90, 1.19] |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | 1 s | 94.3 [93.8, 94.7] | 95.4 [95.0, 95.8] | 98.1 [97.9, 98.4] | 4.10 [3.84, 4.37] | 0.37 [0.32, 0.42] |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | 2 s | 98.5 [98.2, 98.7] | 98.8 [98.6, 99.0] | 99.7 [99.5, 99.8] | 19.70 [18.97, 20.49] | 2.78 [2.52, 3.05] |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | 3 s | 99.4 [99.2, 99.5] | 99.5 [99.3, 99.7] | 99.9 [99.8, 99.9] | 25.14 [24.33, 25.99] | 3.93 [3.60, 4.30] |
| 3D-Speaker CAM++, 1-LSB noise floor | 1 s | 92.7 [92.2, 93.2] | 94.0 [93.6, 94.5] | 97.5 [97.1, 97.8] | 0.47 [0.43, 0.52] | 0.02 [0.01, 0.02] |
| 3D-Speaker CAM++, 1-LSB noise floor | 2 s | 98.9 [98.7, 99.1] | 99.2 [99.0, 99.3] | 99.7 [99.5, 99.8] | 4.53 [4.23, 4.83] | 0.28 [0.23, 0.33] |
| 3D-Speaker CAM++, 1-LSB noise floor | 3 s | 99.4 [99.3, 99.6] | 99.6 [99.4, 99.7] | 99.8 [99.7, 99.9] | 7.49 [7.04, 7.94] | 0.43 [0.37, 0.50] |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | 1 s | 94.6 [94.2, 95.0] | 95.6 [95.2, 95.9] | 98.1 [97.8, 98.4] | 0.47 [0.44, 0.52] | 0.03 [0.02, 0.04] |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | 2 s | 98.9 [98.7, 99.1] | 99.1 [99.0, 99.3] | 99.6 [99.5, 99.8] | 5.65 [5.34, 5.99] | 0.41 [0.36, 0.47] |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | 3 s | 99.6 [99.5, 99.7] | 99.7 [99.6, 99.8] | 99.9 [99.8, 99.9] | 9.03 [8.58, 9.51] | 0.70 [0.61, 0.80] |
| WeSpeaker CAM++ VoxCeleb, sherpa-onnx as shipped | 1 s | 21.0 [19.3, 22.7] | 21.4 [19.8, 23.1] | 16.9 [15.4, 18.6] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| WeSpeaker CAM++ VoxCeleb, sherpa-onnx as shipped | 2 s | 32.8 [31.3, 34.5] | 32.5 [30.9, 34.3] | 24.9 [22.8, 26.9] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| WeSpeaker CAM++ VoxCeleb, sherpa-onnx as shipped | 3 s | 24.7 [22.9, 26.4] | 25.7 [24.0, 27.4] | 38.4 [36.5, 40.3] | 0.06 [0.05, 0.07] | 0.07 [0.06, 0.09] |
