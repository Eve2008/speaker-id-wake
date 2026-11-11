A minus 3D-Speaker CAM++ (NexaRoom default, sherpa-onnx), same eval speakers, 1,000 paired speaker-bootstrap replicates; percentage points.

| configuration minus default | metric | dur | diff pp [95% CI] | p |
|---|---|---|---|---|
| 3D-Speaker CAM++, 1-LSB noise floor | EER (cos) | 1 | -0.72 [-0.89, -0.55] | <0.001 |
| 3D-Speaker CAM++, 1-LSB noise floor | closed-set acc. N=5 (cosine) | 1 | +0.22 [+0.09, +0.35] | 0.002 |
| 3D-Speaker CAM++, 1-LSB noise floor | closed-set acc. N=20 (cosine) | 1 | +0.38 [+0.16, +0.60] | <0.001 |
| 3D-Speaker CAM++, 1-LSB noise floor | NexaRoom rule: member named correctly | 1 | +2.07 [+1.50, +2.66] | <0.001 |
| 3D-Speaker CAM++, 1-LSB noise floor | NexaRoom rule: stranger accepted | 1 | -0.01 [-0.02, +0.01] | 0.332 |
| 3D-Speaker CAM++, 1-LSB noise floor | AS-norm @ dev 1% FA: member named correctly | 1 | +1.57 [+1.13, +1.98] | <0.001 |
| 3D-Speaker CAM++, 1-LSB noise floor | EER (cos) | 2 | -0.35 [-0.48, -0.19] | <0.001 |
| 3D-Speaker CAM++, 1-LSB noise floor | closed-set acc. N=5 (cosine) | 2 | +0.06 [+0.02, +0.11] | 0.002 |
| 3D-Speaker CAM++, 1-LSB noise floor | closed-set acc. N=20 (cosine) | 2 | +0.11 [-0.00, +0.21] | 0.056 |
| 3D-Speaker CAM++, 1-LSB noise floor | NexaRoom rule: member named correctly | 2 | -0.07 [-0.47, +0.30] | 0.666 |
| 3D-Speaker CAM++, 1-LSB noise floor | NexaRoom rule: stranger accepted | 2 | -0.67 [-0.74, -0.59] | <0.001 |
| 3D-Speaker CAM++, 1-LSB noise floor | AS-norm @ dev 1% FA: member named correctly | 2 | +0.62 [+0.29, +0.95] | <0.001 |
| 3D-Speaker CAM++, 1-LSB noise floor | EER (cos) | 3 | -0.19 [-0.32, -0.05] | 0.004 |
| 3D-Speaker CAM++, 1-LSB noise floor | closed-set acc. N=5 (cosine) | 3 | +0.02 [-0.02, +0.05] | 0.310 |
| 3D-Speaker CAM++, 1-LSB noise floor | closed-set acc. N=20 (cosine) | 3 | +0.00 [-0.08, +0.08] | 0.974 |
| 3D-Speaker CAM++, 1-LSB noise floor | NexaRoom rule: member named correctly | 3 | +0.00 [-0.27, +0.26] | 0.984 |
| 3D-Speaker CAM++, 1-LSB noise floor | NexaRoom rule: stranger accepted | 3 | -0.91 [-1.02, -0.80] | <0.001 |
| 3D-Speaker CAM++, 1-LSB noise floor | AS-norm @ dev 1% FA: member named correctly | 3 | +0.17 [-0.17, +0.52] | 0.318 |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | EER (cos) | 1 | -1.09 [-1.24, -0.92] | <0.001 |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | closed-set acc. N=5 (cosine) | 1 | +1.57 [+1.43, +1.73] | <0.001 |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | closed-set acc. N=20 (cosine) | 1 | +3.21 [+2.95, +3.46] | <0.001 |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | NexaRoom rule: member named correctly | 1 | +4.74 [+4.17, +5.35] | <0.001 |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | NexaRoom rule: stranger accepted | 1 | +0.01 [-0.00, +0.02] | 0.112 |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | AS-norm @ dev 1% FA: member named correctly | 1 | +5.47 [+4.94, +6.00] | <0.001 |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | EER (cos) | 2 | -0.05 [-0.15, +0.06] | 0.400 |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | closed-set acc. N=5 (cosine) | 2 | +0.03 [-0.00, +0.07] | 0.088 |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | closed-set acc. N=20 (cosine) | 2 | +0.00 [-0.06, +0.07] | 0.876 |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | NexaRoom rule: member named correctly | 2 | -0.13 [-0.40, +0.15] | 0.336 |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | NexaRoom rule: stranger accepted | 2 | +0.04 [+0.00, +0.08] | 0.038 |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | AS-norm @ dev 1% FA: member named correctly | 2 | +0.01 [-0.30, +0.32] | 0.918 |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | EER (cos) | 3 | -0.38 [-0.49, -0.27] | <0.001 |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | closed-set acc. N=5 (cosine) | 3 | +0.13 [+0.10, +0.17] | <0.001 |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | closed-set acc. N=20 (cosine) | 3 | +0.40 [+0.31, +0.49] | <0.001 |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | NexaRoom rule: member named correctly | 3 | +1.01 [+0.73, +1.31] | <0.001 |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | NexaRoom rule: stranger accepted | 3 | +0.13 [+0.06, +0.20] | <0.001 |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | AS-norm @ dev 1% FA: member named correctly | 3 | +1.48 [+1.13, +1.85] | <0.001 |
| 3D-Speaker ERes2Net-base | EER (cos) | 1 | -1.86 [-2.09, -1.60] | <0.001 |
| 3D-Speaker ERes2Net-base | closed-set acc. N=5 (cosine) | 1 | +2.69 [+2.46, +2.91] | <0.001 |
| 3D-Speaker ERes2Net-base | closed-set acc. N=20 (cosine) | 1 | +6.02 [+5.63, +6.41] | <0.001 |
| 3D-Speaker ERes2Net-base | NexaRoom rule: member named correctly | 1 | +35.90 [+34.83, +37.07] | <0.001 |
| 3D-Speaker ERes2Net-base | NexaRoom rule: stranger accepted | 1 | +2.19 [+2.08, +2.30] | <0.001 |
| 3D-Speaker ERes2Net-base | AS-norm @ dev 1% FA: member named correctly | 1 | +11.46 [+10.75, +12.14] | <0.001 |
| 3D-Speaker ERes2Net-base | EER (cos) | 2 | -0.30 [-0.50, -0.09] | 0.004 |
| 3D-Speaker ERes2Net-base | closed-set acc. N=5 (cosine) | 2 | +0.18 [+0.08, +0.26] | <0.001 |
| 3D-Speaker ERes2Net-base | closed-set acc. N=20 (cosine) | 2 | +0.56 [+0.37, +0.72] | <0.001 |
| 3D-Speaker ERes2Net-base | NexaRoom rule: member named correctly | 2 | +5.36 [+4.84, +5.95] | <0.001 |
| 3D-Speaker ERes2Net-base | NexaRoom rule: stranger accepted | 2 | +10.61 [+10.20, +11.03] | <0.001 |
| 3D-Speaker ERes2Net-base | AS-norm @ dev 1% FA: member named correctly | 2 | +2.31 [+1.81, +2.84] | <0.001 |
| 3D-Speaker ERes2Net-base | EER (cos) | 3 | -0.29 [-0.45, -0.08] | 0.008 |
| 3D-Speaker ERes2Net-base | closed-set acc. N=5 (cosine) | 3 | +0.14 [+0.06, +0.21] | <0.001 |
| 3D-Speaker ERes2Net-base | closed-set acc. N=20 (cosine) | 3 | +0.44 [+0.27, +0.62] | <0.001 |
| 3D-Speaker ERes2Net-base | NexaRoom rule: member named correctly | 3 | +2.21 [+1.82, +2.58] | <0.001 |
| 3D-Speaker ERes2Net-base | NexaRoom rule: stranger accepted | 3 | +12.95 [+12.47, +13.45] | <0.001 |
| 3D-Speaker ERes2Net-base | AS-norm @ dev 1% FA: member named correctly | 3 | +2.35 [+1.88, +2.86] | <0.001 |
| NeMo TitaNet-small | EER (cos) | 1 | +2.18 [+1.90, +2.48] | <0.001 |
| NeMo TitaNet-small | closed-set acc. N=5 (cosine) | 1 | -2.23 [-2.53, -1.94] | <0.001 |
| NeMo TitaNet-small | closed-set acc. N=20 (cosine) | 1 | -4.61 [-5.11, -4.12] | <0.001 |
| NeMo TitaNet-small | NexaRoom rule: member named correctly | 1 | -18.80 [-19.85, -17.71] | <0.001 |
| NeMo TitaNet-small | NexaRoom rule: stranger accepted | 1 | -0.05 [-0.07, -0.03] | <0.001 |
| NeMo TitaNet-small | AS-norm @ dev 1% FA: member named correctly | 1 | -10.02 [-11.02, -9.02] | <0.001 |
| NeMo TitaNet-small | EER (cos) | 2 | +1.18 [+0.95, +1.41] | <0.001 |
| NeMo TitaNet-small | closed-set acc. N=5 (cosine) | 2 | -0.76 [-0.93, -0.61] | <0.001 |
| NeMo TitaNet-small | closed-set acc. N=20 (cosine) | 2 | -1.99 [-2.27, -1.72] | <0.001 |
| NeMo TitaNet-small | NexaRoom rule: member named correctly | 2 | -9.75 [-10.59, -8.89] | <0.001 |
| NeMo TitaNet-small | NexaRoom rule: stranger accepted | 2 | -0.90 [-1.04, -0.76] | <0.001 |
| NeMo TitaNet-small | AS-norm @ dev 1% FA: member named correctly | 2 | -5.58 [-6.34, -4.89] | <0.001 |
| NeMo TitaNet-small | EER (cos) | 3 | +0.59 [+0.38, +0.81] | <0.001 |
| NeMo TitaNet-small | closed-set acc. N=5 (cosine) | 3 | -0.21 [-0.32, -0.10] | <0.001 |
| NeMo TitaNet-small | closed-set acc. N=20 (cosine) | 3 | -0.53 [-0.72, -0.34] | <0.001 |
| NeMo TitaNet-small | NexaRoom rule: member named correctly | 3 | -2.62 [-3.18, -2.05] | <0.001 |
| NeMo TitaNet-small | NexaRoom rule: stranger accepted | 3 | -1.72 [-1.93, -1.50] | <0.001 |
| NeMo TitaNet-small | AS-norm @ dev 1% FA: member named correctly | 3 | -0.72 [-1.28, -0.12] | 0.014 |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | EER (cos) | 1 | +2.41 [+2.15, +2.68] | <0.001 |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | closed-set acc. N=5 (cosine) | 1 | -2.91 [-3.18, -2.61] | <0.001 |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | closed-set acc. N=20 (cosine) | 1 | -5.98 [-6.47, -5.51] | <0.001 |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | NexaRoom rule: member named correctly | 1 | -16.48 [-17.60, -15.36] | <0.001 |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | NexaRoom rule: stranger accepted | 1 | +0.11 [+0.08, +0.14] | <0.001 |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | AS-norm @ dev 1% FA: member named correctly | 1 | -15.31 [-16.37, -14.27] | <0.001 |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | EER (cos) | 2 | +2.11 [+1.87, +2.37] | <0.001 |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | closed-set acc. N=5 (cosine) | 2 | -1.65 [-1.85, -1.45] | <0.001 |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | closed-set acc. N=20 (cosine) | 2 | -4.21 [-4.59, -3.86] | <0.001 |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | NexaRoom rule: member named correctly | 2 | -10.73 [-11.57, -9.88] | <0.001 |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | NexaRoom rule: stranger accepted | 2 | +0.34 [+0.19, +0.51] | <0.001 |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | AS-norm @ dev 1% FA: member named correctly | 2 | -14.35 [-15.32, -13.35] | <0.001 |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | EER (cos) | 3 | +1.01 [+0.81, +1.25] | <0.001 |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | closed-set acc. N=5 (cosine) | 3 | -0.69 [-0.84, -0.55] | <0.001 |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | closed-set acc. N=20 (cosine) | 3 | -1.78 [-2.05, -1.53] | <0.001 |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | NexaRoom rule: member named correctly | 3 | -3.17 [-3.72, -2.62] | <0.001 |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | NexaRoom rule: stranger accepted | 3 | +0.64 [+0.43, +0.90] | <0.001 |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | AS-norm @ dev 1% FA: member named correctly | 3 | -6.91 [-7.76, -6.07] | <0.001 |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | EER (cos) | 1 | +0.38 [+0.12, +0.63] | 0.004 |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | closed-set acc. N=5 (cosine) | 1 | -1.08 [-1.31, -0.81] | <0.001 |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | closed-set acc. N=20 (cosine) | 1 | -2.16 [-2.57, -1.75] | <0.001 |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | NexaRoom rule: member named correctly | 1 | -6.29 [-7.38, -5.12] | <0.001 |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | NexaRoom rule: stranger accepted | 1 | +0.30 [+0.26, +0.33] | <0.001 |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | AS-norm @ dev 1% FA: member named correctly | 1 | -7.33 [-8.36, -6.40] | <0.001 |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | EER (cos) | 2 | +0.71 [+0.47, +0.94] | <0.001 |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | closed-set acc. N=5 (cosine) | 2 | -0.85 [-1.00, -0.69] | <0.001 |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | closed-set acc. N=20 (cosine) | 2 | -2.23 [-2.52, -1.93] | <0.001 |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | NexaRoom rule: member named correctly | 2 | -3.78 [-4.45, -3.06] | <0.001 |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | NexaRoom rule: stranger accepted | 2 | +0.65 [+0.48, +0.83] | <0.001 |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | AS-norm @ dev 1% FA: member named correctly | 2 | -7.37 [-8.10, -6.64] | <0.001 |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | EER (cos) | 3 | +0.17 [-0.05, +0.40] | 0.128 |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | closed-set acc. N=5 (cosine) | 3 | -0.23 [-0.33, -0.12] | <0.001 |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | closed-set acc. N=20 (cosine) | 3 | -0.64 [-0.88, -0.41] | <0.001 |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | NexaRoom rule: member named correctly | 3 | -0.53 [-0.99, -0.10] | 0.022 |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | NexaRoom rule: stranger accepted | 3 | +0.53 [+0.30, +0.79] | <0.001 |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | AS-norm @ dev 1% FA: member named correctly | 3 | -2.44 [-3.13, -1.74] | <0.001 |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | EER (cos) | 1 | -0.50 [-0.73, -0.23] | <0.001 |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | closed-set acc. N=5 (cosine) | 1 | +1.48 [+1.23, +1.70] | <0.001 |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | closed-set acc. N=20 (cosine) | 1 | +2.82 [+2.40, +3.18] | <0.001 |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | NexaRoom rule: member named correctly | 1 | +27.95 [+26.88, +29.00] | <0.001 |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | NexaRoom rule: stranger accepted | 1 | +2.84 [+2.71, +2.99] | <0.001 |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | AS-norm @ dev 1% FA: member named correctly | 1 | +2.37 [+1.55, +3.18] | <0.001 |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | EER (cos) | 2 | +0.58 [+0.36, +0.78] | <0.001 |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | closed-set acc. N=5 (cosine) | 2 | -0.21 [-0.31, -0.10] | <0.001 |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | closed-set acc. N=20 (cosine) | 2 | -0.59 [-0.82, -0.39] | <0.001 |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | NexaRoom rule: member named correctly | 2 | +3.64 [+3.17, +4.22] | <0.001 |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | NexaRoom rule: stranger accepted | 2 | +12.82 [+12.39, +13.26] | <0.001 |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | AS-norm @ dev 1% FA: member named correctly | 2 | -2.44 [-3.07, -1.84] | <0.001 |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | EER (cos) | 3 | +0.31 [+0.07, +0.58] | 0.004 |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | closed-set acc. N=5 (cosine) | 3 | -0.01 [-0.10, +0.06] | 0.738 |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | closed-set acc. N=20 (cosine) | 3 | -0.02 [-0.21, +0.14] | 0.842 |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | NexaRoom rule: member named correctly | 3 | +1.38 [+1.01, +1.74] | <0.001 |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | NexaRoom rule: stranger accepted | 3 | +15.70 [+15.24, +16.23] | <0.001 |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | AS-norm @ dev 1% FA: member named correctly | 3 | +0.04 [-0.50, +0.61] | 0.890 |
