Open set, household of N = 5. Member wakes: true speaker enrolled; stranger wakes: not enrolled. Rows sum to 100% over correct + wrong + unknown + ambiguous + too_short for members.

| model | test | rule | member named correctly % | wrong member named % | member -> unknown % | member -> ambiguous % | member -> too_short % | stranger accepted % | stranger -> ambiguous % |
|---|---|---|---|---|---|---|---|---|---|
| 3D-Speaker CAM++ (NexaRoom default) | 1 s | NexaRoom rule (defaults) | 37.9 [36.8, 39.1] | 0.05 | 60.8 | 0.2 | 1.02 | 0.26 [0.24, 0.27] | 0.02 |
| 3D-Speaker CAM++ (NexaRoom default) | 1 s | cosine >= 0.50 | 64.3 [63.2, 65.4] | 0.57 | 34.1 | 0.0 | 1.02 | 2.33 [2.22, 2.43] | 0.00 |
| 3D-Speaker CAM++ (NexaRoom default) | 1 s | cosine, threshold matched to NexaRoom's dev FA | 39.8 [38.6, 41.0] | 0.10 |  |  |  | 0.25 [0.23, 0.26] |  |
| 3D-Speaker CAM++ (NexaRoom default) | 2 s | NexaRoom rule (defaults) | 91.2 [90.5, 92.0] | 0.06 | 7.4 | 0.9 | 0.41 | 3.40 [3.25, 3.55] | 0.65 |
| 3D-Speaker CAM++ (NexaRoom default) | 2 s | cosine >= 0.50 | 94.2 [93.5, 94.8] | 0.28 | 5.1 | 0.0 | 0.41 | 6.52 [6.23, 6.78] | 0.00 |
| 3D-Speaker CAM++ (NexaRoom default) | 2 s | cosine, threshold matched to NexaRoom's dev FA | 91.4 [90.6, 92.2] | 0.20 |  |  |  | 3.36 [3.19, 3.52] |  |
| 3D-Speaker CAM++ (NexaRoom default) | 3 s | NexaRoom rule (defaults) | 96.3 [95.8, 96.8] | 0.05 | 2.6 | 0.7 | 0.24 | 5.61 [5.37, 5.82] | 1.13 |
| 3D-Speaker CAM++ (NexaRoom default) | 3 s | cosine >= 0.50 | 97.8 [97.4, 98.2] | 0.22 | 1.7 | 0.0 | 0.24 | 10.32 [9.88, 10.73] | 0.00 |
| 3D-Speaker CAM++ (NexaRoom default) | 3 s | cosine, threshold matched to NexaRoom's dev FA | 96.6 [96.1, 97.2] | 0.16 |  |  |  | 5.60 [5.33, 5.86] |  |
| 3D-Speaker ERes2Net-base | 1 s | NexaRoom rule (defaults) | 73.8 [72.7, 75.0] | 0.13 | 23.9 | 1.1 | 1.02 | 2.45 [2.33, 2.57] | 0.41 |
| 3D-Speaker ERes2Net-base | 1 s | cosine >= 0.50 | 88.1 [87.4, 88.8] | 0.97 | 9.9 | 0.0 | 1.02 | 13.18 [12.69, 13.69] | 0.00 |
| 3D-Speaker ERes2Net-base | 1 s | cosine, threshold matched to NexaRoom's dev FA | 74.1 [72.9, 75.1] | 0.32 |  |  |  | 2.47 [2.34, 2.61] |  |
| 3D-Speaker ERes2Net-base | 2 s | NexaRoom rule (defaults) | 96.6 [96.1, 97.0] | 0.11 | 1.8 | 1.2 | 0.41 | 14.01 [13.56, 14.44] | 5.01 |
| 3D-Speaker ERes2Net-base | 2 s | cosine >= 0.50 | 97.9 [97.6, 98.3] | 0.40 | 1.2 | 0.0 | 0.41 | 25.45 [24.62, 26.27] | 0.00 |
| 3D-Speaker ERes2Net-base | 2 s | cosine, threshold matched to NexaRoom's dev FA | 97.0 [96.5, 97.4] | 0.29 |  |  |  | 13.81 [13.27, 14.36] |  |
| 3D-Speaker ERes2Net-base | 3 s | NexaRoom rule (defaults) | 98.5 [98.2, 98.9] | 0.07 | 0.5 | 0.6 | 0.24 | 18.55 [18.05, 19.05] | 7.42 |
| 3D-Speaker ERes2Net-base | 3 s | cosine >= 0.50 | 99.1 [98.8, 99.4] | 0.23 | 0.4 | 0.0 | 0.24 | 33.37 [32.42, 34.32] | 0.00 |
| 3D-Speaker ERes2Net-base | 3 s | cosine, threshold matched to NexaRoom's dev FA | 98.7 [98.4, 99.1] | 0.17 |  |  |  | 18.16 [17.50, 18.82] |  |
| NeMo TitaNet-small | 1 s | NexaRoom rule (defaults) | 19.1 [18.2, 20.0] | 0.06 | 79.6 | 0.1 | 1.02 | 0.21 [0.19, 0.23] | 0.02 |
| NeMo TitaNet-small | 1 s | cosine >= 0.50 | 42.4 [41.2, 43.5] | 0.42 | 56.1 | 0.0 | 1.02 | 1.15 [1.08, 1.22] | 0.00 |
| NeMo TitaNet-small | 1 s | cosine, threshold matched to NexaRoom's dev FA | 23.8 [22.7, 24.7] | 0.11 |  |  |  | 0.23 [0.21, 0.25] |  |
| NeMo TitaNet-small | 2 s | NexaRoom rule (defaults) | 81.5 [80.4, 82.4] | 0.12 | 17.1 | 0.9 | 0.41 | 2.50 [2.35, 2.63] | 0.41 |
| NeMo TitaNet-small | 2 s | cosine >= 0.50 | 85.9 [84.9, 86.8] | 0.38 | 13.3 | 0.0 | 0.41 | 3.88 [3.65, 4.09] | 0.00 |
| NeMo TitaNet-small | 2 s | cosine, threshold matched to NexaRoom's dev FA | 82.8 [81.8, 83.9] | 0.30 |  |  |  | 2.69 [2.51, 2.85] |  |
| NeMo TitaNet-small | 3 s | NexaRoom rule (defaults) | 93.7 [93.0, 94.4] | 0.08 | 5.3 | 0.7 | 0.24 | 3.88 [3.66, 4.08] | 0.67 |
| NeMo TitaNet-small | 3 s | cosine >= 0.50 | 95.3 [94.6, 96.0] | 0.21 | 4.2 | 0.0 | 0.24 | 6.12 [5.77, 6.44] | 0.00 |
| NeMo TitaNet-small | 3 s | cosine, threshold matched to NexaRoom's dev FA | 93.9 [93.1, 94.6] | 0.17 |  |  |  | 4.15 [3.88, 4.39] |  |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | 1 s | NexaRoom rule (defaults) | 21.5 [20.6, 22.3] | 0.12 | 77.1 | 0.3 | 1.02 | 0.37 [0.34, 0.39] | 0.05 |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | 1 s | cosine >= 0.50 | 45.2 [44.3, 46.3] | 0.68 | 53.1 | 0.0 | 1.02 | 1.72 [1.64, 1.81] | 0.00 |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | 1 s | cosine, threshold matched to NexaRoom's dev FA | 27.4 [26.4, 28.4] | 0.21 |  |  |  | 0.41 [0.38, 0.44] |  |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | 2 s | NexaRoom rule (defaults) | 80.5 [79.6, 81.5] | 0.24 | 17.2 | 1.6 | 0.41 | 3.74 [3.60, 3.89] | 0.82 |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | 2 s | cosine >= 0.50 | 85.5 [84.6, 86.5] | 0.74 | 13.3 | 0.0 | 0.41 | 5.74 [5.50, 5.99] | 0.00 |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | 2 s | cosine, threshold matched to NexaRoom's dev FA | 82.6 [81.7, 83.7] | 0.60 |  |  |  | 4.08 [3.89, 4.27] |  |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | 3 s | NexaRoom rule (defaults) | 93.2 [92.5, 93.8] | 0.15 | 5.0 | 1.5 | 0.24 | 6.25 [6.02, 6.49] | 1.52 |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | 3 s | cosine >= 0.50 | 95.5 [94.9, 96.1] | 0.49 | 3.7 | 0.0 | 0.24 | 10.06 [9.68, 10.46] | 0.00 |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | 3 s | cosine, threshold matched to NexaRoom's dev FA | 94.0 [93.3, 94.7] | 0.41 |  |  |  | 6.56 [6.27, 6.86] |  |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | 1 s | NexaRoom rule (defaults) | 31.7 [30.6, 32.8] | 0.20 | 66.8 | 0.4 | 1.02 | 0.55 [0.51, 0.59] | 0.05 |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | 1 s | cosine >= 0.50 | 60.7 [59.6, 61.9] | 1.18 | 37.1 | 0.0 | 1.02 | 3.38 [3.20, 3.55] | 0.00 |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | 1 s | cosine, threshold matched to NexaRoom's dev FA | 34.2 [33.1, 35.4] | 0.33 |  |  |  | 0.64 [0.59, 0.69] |  |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | 2 s | NexaRoom rule (defaults) | 87.4 [86.6, 88.3] | 0.19 | 10.4 | 1.6 | 0.41 | 4.05 [3.87, 4.23] | 0.80 |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | 2 s | cosine >= 0.50 | 92.2 [91.5, 93.0] | 0.73 | 6.6 | 0.0 | 0.41 | 7.82 [7.45, 8.16] | 0.00 |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | 2 s | cosine, threshold matched to NexaRoom's dev FA | 88.0 [87.1, 88.9] | 0.53 |  |  |  | 4.19 [3.96, 4.40] |  |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | 3 s | NexaRoom rule (defaults) | 95.8 [95.2, 96.3] | 0.08 | 2.8 | 1.1 | 0.24 | 6.14 [5.87, 6.40] | 1.33 |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | 3 s | cosine >= 0.50 | 97.6 [97.2, 98.1] | 0.36 | 1.7 | 0.0 | 0.24 | 11.68 [11.17, 12.17] | 0.00 |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | 3 s | cosine, threshold matched to NexaRoom's dev FA | 96.3 [95.7, 96.8] | 0.29 |  |  |  | 6.30 [5.96, 6.61] |  |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | 1 s | NexaRoom rule (defaults) | 65.9 [64.8, 67.0] | 0.23 | 31.0 | 1.8 | 1.02 | 3.09 [2.93, 3.25] | 0.63 |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | 1 s | cosine >= 0.50 | 84.9 [84.1, 85.7] | 1.68 | 12.4 | 0.0 | 1.02 | 15.82 [15.22, 16.41] | 0.00 |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | 1 s | cosine, threshold matched to NexaRoom's dev FA | 65.7 [64.6, 66.9] | 0.58 |  |  |  | 3.08 [2.90, 3.25] |  |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | 2 s | NexaRoom rule (defaults) | 94.8 [94.3, 95.4] | 0.17 | 2.3 | 2.3 | 0.41 | 16.23 [15.69, 16.72] | 6.40 |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | 2 s | cosine >= 0.50 | 97.2 [96.8, 97.7] | 0.70 | 1.6 | 0.0 | 0.41 | 29.99 [29.02, 30.95] | 0.00 |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | 2 s | cosine, threshold matched to NexaRoom's dev FA | 95.9 [95.3, 96.4] | 0.55 |  |  |  | 16.44 [15.73, 17.11] |  |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | 3 s | NexaRoom rule (defaults) | 97.7 [97.3, 98.1] | 0.09 | 0.7 | 1.3 | 0.24 | 21.31 [20.67, 21.88] | 9.44 |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | 3 s | cosine >= 0.50 | 99.0 [98.6, 99.2] | 0.34 | 0.5 | 0.0 | 0.24 | 39.07 [37.86, 40.19] | 0.00 |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | 3 s | cosine, threshold matched to NexaRoom's dev FA | 98.4 [98.0, 98.8] | 0.28 |  |  |  | 21.68 [20.79, 22.54] |  |
| 3D-Speaker CAM++, 1-LSB noise floor | 1 s | NexaRoom rule (defaults) | 40.0 [38.9, 41.3] | 0.05 | 58.7 | 0.2 | 1.02 | 0.25 [0.24, 0.27] | 0.02 |
| 3D-Speaker CAM++, 1-LSB noise floor | 1 s | cosine >= 0.50 | 65.8 [64.7, 66.9] | 0.55 | 32.6 | 0.0 | 1.02 | 2.21 [2.12, 2.30] | 0.00 |
| 3D-Speaker CAM++, 1-LSB noise floor | 1 s | cosine, threshold matched to NexaRoom's dev FA | 42.6 [41.4, 43.9] | 0.10 |  |  |  | 0.25 [0.24, 0.27] |  |
| 3D-Speaker CAM++, 1-LSB noise floor | 2 s | NexaRoom rule (defaults) | 91.1 [90.4, 91.9] | 0.06 | 7.7 | 0.7 | 0.41 | 2.74 [2.61, 2.87] | 0.46 |
| 3D-Speaker CAM++, 1-LSB noise floor | 2 s | cosine >= 0.50 | 94.0 [93.3, 94.6] | 0.25 | 5.4 | 0.0 | 0.41 | 5.02 [4.78, 5.26] | 0.00 |
| 3D-Speaker CAM++, 1-LSB noise floor | 2 s | cosine, threshold matched to NexaRoom's dev FA | 91.6 [90.8, 92.3] | 0.18 |  |  |  | 2.73 [2.59, 2.88] |  |
| 3D-Speaker CAM++, 1-LSB noise floor | 3 s | NexaRoom rule (defaults) | 96.3 [95.8, 96.9] | 0.05 | 2.8 | 0.6 | 0.24 | 4.69 [4.48, 4.89] | 0.81 |
| 3D-Speaker CAM++, 1-LSB noise floor | 3 s | cosine >= 0.50 | 97.7 [97.2, 98.2] | 0.20 | 1.9 | 0.0 | 0.24 | 8.42 [8.05, 8.76] | 0.00 |
| 3D-Speaker CAM++, 1-LSB noise floor | 3 s | cosine, threshold matched to NexaRoom's dev FA | 96.6 [96.0, 97.2] | 0.16 |  |  |  | 4.72 [4.48, 4.95] |  |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | 1 s | NexaRoom rule (defaults) | 42.7 [41.6, 43.9] | 0.04 | 56.0 | 0.3 | 1.02 | 0.27 [0.25, 0.28] | 0.03 |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | 1 s | cosine >= 0.50 | 69.9 [68.8, 70.9] | 0.49 | 28.6 | 0.0 | 1.02 | 2.53 [2.41, 2.64] | 0.00 |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | 1 s | cosine, threshold matched to NexaRoom's dev FA | 44.8 [43.6, 45.9] | 0.09 |  |  |  | 0.25 [0.24, 0.27] |  |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | 2 s | NexaRoom rule (defaults) | 91.1 [90.3, 91.8] | 0.06 | 7.6 | 0.9 | 0.41 | 3.45 [3.29, 3.60] | 0.65 |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | 2 s | cosine >= 0.50 | 94.2 [93.6, 94.8] | 0.28 | 5.1 | 0.0 | 0.41 | 6.62 [6.32, 6.90] | 0.00 |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | 2 s | cosine, threshold matched to NexaRoom's dev FA | 91.4 [90.7, 92.3] | 0.19 |  |  |  | 3.40 [3.23, 3.57] |  |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | 3 s | NexaRoom rule (defaults) | 97.3 [96.9, 97.8] | 0.03 | 1.8 | 0.6 | 0.24 | 5.73 [5.49, 5.96] | 1.25 |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | 3 s | cosine >= 0.50 | 98.4 [98.0, 98.7] | 0.14 | 1.2 | 0.0 | 0.24 | 10.82 [10.38, 11.26] | 0.00 |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | 3 s | cosine, threshold matched to NexaRoom's dev FA | 97.6 [97.1, 98.1] | 0.11 |  |  |  | 5.72 [5.44, 6.00] |  |
| WeSpeaker CAM++ VoxCeleb, sherpa-onnx as shipped | 1 s | NexaRoom rule (defaults) | 0.0 [0.0, 0.0] | 0.00 | 99.0 | 0.0 | 1.02 | 0.00 [0.00, 0.00] | 0.00 |
| WeSpeaker CAM++ VoxCeleb, sherpa-onnx as shipped | 1 s | cosine >= 0.50 | 0.0 [0.0, 0.0] | 0.01 | 99.0 | 0.0 | 1.02 | 0.02 [0.01, 0.03] | 0.00 |
| WeSpeaker CAM++ VoxCeleb, sherpa-onnx as shipped | 1 s | cosine, threshold matched to NexaRoom's dev FA | 0.0 [0.0, 0.0] | 0.01 |  |  |  | 0.01 [0.00, 0.02] |  |
| WeSpeaker CAM++ VoxCeleb, sherpa-onnx as shipped | 2 s | NexaRoom rule (defaults) | 0.0 [0.0, 0.0] | 0.00 | 99.6 | 0.0 | 0.41 | 0.00 [0.00, 0.00] | 0.00 |
| WeSpeaker CAM++ VoxCeleb, sherpa-onnx as shipped | 2 s | cosine >= 0.50 | 0.0 [0.0, 0.0] | 0.00 | 99.6 | 0.0 | 0.41 | 0.00 [0.00, 0.00] | 0.00 |
| WeSpeaker CAM++ VoxCeleb, sherpa-onnx as shipped | 2 s | cosine, threshold matched to NexaRoom's dev FA | 0.0 [0.0, 0.0] | 0.00 |  |  |  | 0.01 [0.00, 0.01] |  |
| WeSpeaker CAM++ VoxCeleb, sherpa-onnx as shipped | 3 s | NexaRoom rule (defaults) | 0.2 [0.1, 0.3] | 0.04 | 3.3 | 96.3 | 0.24 | 0.10 [0.09, 0.12] | 96.09 |
| WeSpeaker CAM++ VoxCeleb, sherpa-onnx as shipped | 3 s | cosine >= 0.50 | 24.9 [23.7, 26.1] | 71.97 | 2.9 | 0.0 | 0.24 | 96.60 [96.18, 97.01] | 0.00 |
| WeSpeaker CAM++ VoxCeleb, sherpa-onnx as shipped | 3 s | cosine, threshold matched to NexaRoom's dev FA | 0.1 [0.0, 0.1] | 0.06 |  |  |  | 0.10 [0.05, 0.15] |  |
