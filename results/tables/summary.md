### EER and closed-set identification accuracy (cosine to centroid)

| model | test | EER cosine % | EER AS-norm % | closed-set acc. N=5 % | closed-set acc. N=20 % |
|---|---|---|---|---|---|
| 3D-Speaker CAM++ (NexaRoom default) | 1 s | 8.44 [8.11, 8.72] | 7.20 [6.90, 7.58] | 94.5 [94.1, 94.8] | 87.0 [86.4, 87.6] |
| 3D-Speaker CAM++ (NexaRoom default) | 2 s | 3.10 [2.83, 3.34] | 2.28 [2.05, 2.57] | 99.1 [99.0, 99.3] | 97.6 [97.3, 97.9] |
| 3D-Speaker CAM++ (NexaRoom default) | 3 s | 2.07 [1.86, 2.29] | 1.56 [1.36, 1.80] | 99.5 [99.4, 99.6] | 98.7 [98.4, 98.9] |
| 3D-Speaker CAM++, 1-LSB noise floor | 1 s | 7.72 [7.37, 8.03] | 6.81 [6.50, 7.12] | 94.7 [94.4, 95.0] | 87.3 [86.8, 88.0] |
| 3D-Speaker CAM++, 1-LSB noise floor | 2 s | 2.75 [2.52, 2.96] | 2.18 [1.94, 2.41] | 99.2 [99.0, 99.3] | 97.7 [97.4, 98.0] |
| 3D-Speaker CAM++, 1-LSB noise floor | 3 s | 1.88 [1.67, 2.13] | 1.57 [1.33, 1.81] | 99.5 [99.4, 99.7] | 98.7 [98.4, 98.9] |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | 1 s | 7.34 [7.06, 7.63] | 6.01 [5.73, 6.31] | 96.1 [95.8, 96.3] | 90.2 [89.7, 90.7] |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | 2 s | 3.05 [2.81, 3.30] | 2.30 [2.08, 2.55] | 99.2 [99.0, 99.3] | 97.6 [97.3, 97.9] |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | 3 s | 1.70 [1.49, 1.92] | 1.25 [1.09, 1.47] | 99.6 [99.5, 99.7] | 99.1 [98.8, 99.3] |
| 3D-Speaker ERes2Net-base | 1 s | 6.58 [6.27, 6.89] | 5.03 [4.76, 5.32] | 97.2 [96.9, 97.4] | 93.0 [92.5, 93.5] |
| 3D-Speaker ERes2Net-base | 2 s | 2.80 [2.55, 3.09] | 2.01 [1.73, 2.30] | 99.3 [99.1, 99.5] | 98.2 [97.9, 98.5] |
| 3D-Speaker ERes2Net-base | 3 s | 1.78 [1.51, 2.09] | 1.22 [0.97, 1.46] | 99.6 [99.5, 99.8] | 99.1 [98.8, 99.3] |
| NeMo TitaNet-small | 1 s | 10.62 [10.29, 10.96] | 9.02 [8.72, 9.40] | 92.3 [91.9, 92.7] | 82.4 [81.6, 83.0] |
| NeMo TitaNet-small | 2 s | 4.28 [3.98, 4.57] | 3.33 [3.04, 3.64] | 98.4 [98.1, 98.6] | 95.6 [95.2, 96.1] |
| NeMo TitaNet-small | 3 s | 2.66 [2.40, 2.92] | 1.99 [1.67, 2.26] | 99.3 [99.1, 99.5] | 98.1 [97.8, 98.4] |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | 1 s | 10.84 [10.51, 11.20] | 9.55 [9.18, 9.92] | 91.6 [91.2, 92.0] | 81.0 [80.3, 81.7] |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | 2 s | 5.21 [4.90, 5.55] | 4.45 [4.13, 4.76] | 97.5 [97.2, 97.8] | 93.4 [92.9, 93.9] |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | 3 s | 3.08 [2.83, 3.35] | 2.54 [2.34, 2.86] | 98.8 [98.6, 99.0] | 96.9 [96.5, 97.3] |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | 1 s | 8.82 [8.50, 9.11] | 7.64 [7.33, 7.97] | 93.4 [93.0, 93.8] | 84.8 [84.2, 85.4] |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | 2 s | 3.81 [3.50, 4.09] | 3.21 [2.93, 3.49] | 98.3 [98.0, 98.5] | 95.4 [95.0, 95.8] |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | 3 s | 2.24 [1.96, 2.54] | 1.82 [1.53, 2.05] | 99.3 [99.1, 99.4] | 98.0 [97.7, 98.3] |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | 1 s | 7.93 [7.63, 8.28] | 6.29 [5.94, 6.64] | 96.0 [95.7, 96.3] | 89.8 [89.2, 90.3] |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | 2 s | 3.68 [3.38, 3.93] | 2.68 [2.41, 3.01] | 98.9 [98.7, 99.1] | 97.0 [96.6, 97.4] |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | 3 s | 2.38 [2.06, 2.74] | 1.55 [1.36, 1.83] | 99.5 [99.3, 99.6] | 98.6 [98.3, 98.9] |
| WeSpeaker CAM++ VoxCeleb, sherpa-onnx as shipped | 1 s | 49.73 [49.18, 50.20] | 50.00 [50.00, 50.00] | 21.6 [20.3, 22.8] | 5.7 [5.0, 6.5] |
| WeSpeaker CAM++ VoxCeleb, sherpa-onnx as shipped | 2 s | 48.31 [48.02, 48.54] | 50.00 [50.00, 50.00] | 33.6 [32.2, 34.8] | 11.9 [11.1, 12.8] |
| WeSpeaker CAM++ VoxCeleb, sherpa-onnx as shipped | 3 s | 47.82 [47.32, 48.40] | 50.00 [49.99, 50.00] | 25.7 [24.5, 26.9] | 7.7 [7.0, 8.5] |

### Open set, household of N = 5 (member wakes and stranger wakes)

Dev-tuned thresholds aim at 1% stranger acceptance on dev speakers; the realised value on eval speakers is in main.csv (`stranger_fa`).

| model | test | NexaRoom rule: correct % | NexaRoom rule: wrong member % | NexaRoom rule: stranger accepted % | cosine >= 0.50: correct % | cosine >= 0.50: wrong member % | cosine >= 0.50: stranger accepted % | 1% stranger FA (dev-tuned), cosine: correct % | 1% stranger FA (dev-tuned), NexaRoom rule: correct % | 1% stranger FA (dev-tuned), AS-norm: correct % |
|---|---|---|---|---|---|---|---|---|---|---|
| 3D-Speaker CAM++ (NexaRoom default) | 1 s | 37.9 | 0.05 | 0.26 | 64.3 | 0.57 | 2.33 | 56.1 [55.0, 57.4] | 55.6 [54.4, 56.8] | 65.0 [63.8, 66.3] |
| 3D-Speaker CAM++ (NexaRoom default) | 2 s | 91.2 | 0.06 | 3.40 | 94.2 | 0.28 | 6.52 | 85.3 [84.4, 86.3] | 83.2 [82.3, 84.2] | 90.6 [89.8, 91.4] |
| 3D-Speaker CAM++ (NexaRoom default) | 3 s | 96.3 | 0.05 | 5.61 | 97.8 | 0.22 | 10.32 | 91.0 [90.1, 91.9] | 88.1 [87.3, 89.0] | 94.4 [93.7, 95.1] |
| 3D-Speaker CAM++, 1-LSB noise floor | 1 s | 40.0 | 0.05 | 0.25 | 65.8 | 0.55 | 2.21 | 57.9 [56.7, 59.1] | 57.4 [56.2, 58.7] | 66.6 [65.4, 67.8] |
| 3D-Speaker CAM++, 1-LSB noise floor | 2 s | 91.1 | 0.06 | 2.74 | 94.0 | 0.25 | 5.02 | 87.0 [86.1, 88.0] | 85.0 [84.2, 85.9] | 91.2 [90.4, 92.0] |
| 3D-Speaker CAM++, 1-LSB noise floor | 3 s | 96.3 | 0.05 | 4.69 | 97.7 | 0.20 | 8.42 | 91.9 [91.1, 92.8] | 89.9 [89.1, 90.7] | 94.6 [93.9, 95.3] |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | 1 s | 42.7 | 0.04 | 0.27 | 69.9 | 0.49 | 2.53 | 60.7 [59.6, 61.9] | 60.4 [59.3, 61.5] | 70.5 [69.4, 71.6] |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | 2 s | 91.1 | 0.06 | 3.45 | 94.2 | 0.28 | 6.62 | 84.8 [83.9, 85.9] | 83.2 [82.3, 84.2] | 90.6 [89.8, 91.4] |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | 3 s | 97.3 | 0.03 | 5.73 | 98.4 | 0.14 | 10.82 | 93.1 [92.3, 93.9] | 91.2 [90.4, 91.9] | 95.9 [95.3, 96.5] |
| 3D-Speaker ERes2Net-base | 1 s | 73.8 | 0.13 | 2.45 | 88.1 | 0.97 | 13.18 | 64.7 [63.5, 66.0] | 64.3 [63.0, 65.5] | 76.4 [75.4, 77.5] |
| 3D-Speaker ERes2Net-base | 2 s | 96.6 | 0.11 | 14.01 | 97.9 | 0.40 | 25.45 | 87.7 [86.7, 88.7] | 87.1 [86.2, 88.0] | 92.9 [92.1, 93.6] |
| 3D-Speaker ERes2Net-base | 3 s | 98.5 | 0.07 | 18.55 | 99.1 | 0.23 | 33.37 | 94.3 [93.5, 95.0] | 94.0 [93.3, 94.7] | 96.8 [96.2, 97.3] |
| NeMo TitaNet-small | 1 s | 19.1 | 0.06 | 0.21 | 42.4 | 0.42 | 1.15 | 41.7 [40.5, 42.8] | 40.3 [39.1, 41.4] | 55.0 [53.6, 56.3] |
| NeMo TitaNet-small | 2 s | 81.5 | 0.12 | 2.50 | 85.9 | 0.38 | 3.88 | 74.0 [72.7, 75.2] | 69.8 [68.6, 70.9] | 85.0 [84.0, 86.0] |
| NeMo TitaNet-small | 3 s | 93.7 | 0.08 | 3.88 | 95.3 | 0.21 | 6.12 | 87.1 [85.9, 88.1] | 84.9 [83.8, 85.8] | 93.7 [93.0, 94.4] |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | 1 s | 21.5 | 0.12 | 0.37 | 45.2 | 0.68 | 1.72 | 37.8 [36.8, 38.9] | 35.4 [34.4, 36.5] | 49.7 [48.4, 50.9] |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | 2 s | 80.5 | 0.24 | 3.74 | 85.5 | 0.74 | 5.74 | 67.6 [66.4, 68.9] | 57.6 [56.6, 58.7] | 76.2 [75.0, 77.5] |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | 3 s | 93.2 | 0.15 | 6.25 | 95.5 | 0.49 | 10.06 | 81.8 [80.8, 83.0] | 77.2 [76.1, 78.3] | 87.5 [86.4, 88.6] |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | 1 s | 31.7 | 0.20 | 0.55 | 60.7 | 1.18 | 3.38 | 40.9 [39.7, 42.1] | 39.7 [38.6, 41.0] | 57.6 [56.4, 59.0] |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | 2 s | 87.4 | 0.19 | 4.05 | 92.2 | 0.73 | 7.82 | 74.2 [73.0, 75.5] | 66.2 [65.1, 67.3] | 83.2 [82.1, 84.2] |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | 3 s | 95.8 | 0.08 | 6.14 | 97.6 | 0.36 | 11.68 | 87.0 [86.0, 88.0] | 83.2 [82.2, 84.2] | 92.0 [91.1, 92.8] |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | 1 s | 65.9 | 0.23 | 3.09 | 84.9 | 1.68 | 15.82 | 52.0 [50.8, 53.3] | 51.9 [50.7, 53.1] | 67.4 [66.1, 68.5] |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | 2 s | 94.8 | 0.17 | 16.23 | 97.2 | 0.70 | 29.99 | 80.1 [78.9, 81.3] | 76.0 [74.8, 77.0] | 88.2 [87.1, 89.0] |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | 3 s | 97.7 | 0.09 | 21.31 | 99.0 | 0.34 | 39.07 | 89.9 [88.9, 90.9] | 87.0 [86.1, 87.9] | 94.5 [93.7, 95.1] |
| WeSpeaker CAM++ VoxCeleb, sherpa-onnx as shipped | 1 s | 0.0 | 0.00 | 0.00 | 0.0 | 0.01 | 0.02 | 0.3 [0.2, 0.4] | 0.3 [0.2, 0.4] | 0.2 [0.1, 0.3] |
| WeSpeaker CAM++ VoxCeleb, sherpa-onnx as shipped | 2 s | 0.0 | 0.00 | 0.00 | 0.0 | 0.00 | 0.00 | 0.4 [0.3, 0.5] | 0.3 [0.2, 0.5] | 0.3 [0.1, 0.5] |
| WeSpeaker CAM++ VoxCeleb, sherpa-onnx as shipped | 3 s | 0.2 | 0.04 | 0.10 | 24.9 | 71.97 | 96.60 | 0.5 [0.3, 0.7] | 0.2 [0.1, 0.3] | 0.3 [0.1, 0.5] |

### Gender (speakers with a male/female label only)

| model | test | EER, non-targets of same gender % | EER, non-targets of cross gender % |
|---|---|---|---|
| 3D-Speaker CAM++ (NexaRoom default) | 1 s | 10.45 [9.93, 10.90] | 5.10 [4.72, 5.51] |
| 3D-Speaker CAM++ (NexaRoom default) | 3 s | 2.61 [2.17, 3.00] | 0.72 [0.54, 0.88] |
| 3D-Speaker CAM++, 1-LSB noise floor | 1 s | 9.70 [9.24, 10.16] | 4.18 [3.81, 4.54] |
| 3D-Speaker CAM++, 1-LSB noise floor | 3 s | 2.26 [1.99, 2.60] | 0.59 [0.43, 0.75] |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | 1 s | 9.15 [8.69, 9.60] | 4.47 [4.12, 4.85] |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | 3 s | 2.13 [1.83, 2.47] | 0.63 [0.46, 0.81] |
| 3D-Speaker ERes2Net-base | 1 s | 8.50 [8.07, 8.98] | 3.18 [2.85, 3.53] |
| 3D-Speaker ERes2Net-base | 3 s | 2.20 [1.88, 2.60] | 0.53 [0.38, 0.77] |
| NeMo TitaNet-small | 1 s | 13.38 [12.88, 13.95] | 7.09 [6.65, 7.59] |
| NeMo TitaNet-small | 3 s | 3.37 [3.01, 3.77] | 1.19 [0.94, 1.52] |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | 1 s | 13.65 [13.15, 14.15] | 6.64 [6.22, 7.08] |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | 3 s | 4.00 [3.56, 4.41] | 1.17 [0.90, 1.44] |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | 1 s | 11.02 [10.58, 11.48] | 5.90 [5.50, 6.32] |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | 3 s | 2.87 [2.52, 3.17] | 0.92 [0.75, 1.14] |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | 1 s | 10.00 [9.53, 10.49] | 4.42 [4.02, 4.87] |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | 3 s | 2.91 [2.52, 3.38] | 0.93 [0.68, 1.24] |
| WeSpeaker CAM++ VoxCeleb, sherpa-onnx as shipped | 1 s | 50.56 [49.92, 51.27] | 48.35 [47.34, 49.38] |
| WeSpeaker CAM++ VoxCeleb, sherpa-onnx as shipped | 3 s | 49.64 [48.80, 50.33] | 44.48 [43.47, 45.49] |
