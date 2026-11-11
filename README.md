# speaker-id-wake

Benchmark of speaker identification from very short Persian speech (the wake
word itself, 1-3 s) with small speaker-embedding models that run on edge
hardware.

The question comes from NexaRoom, a Persian smart mirror
(`work/NexaRoom/nexaroom/voice/speakers.py`). When someone wakes it, NexaRoom
tries to name the speaker from about 1 s of wake audio plus the start of the
command, with a sherpa-onnx speaker-embedding model and a fail-closed rule:
score = 0.5 cos(query, centroid) + 0.5 mean(top-3 sample cosines); a person
is named only if the score reaches the profile's calibrated threshold (default
0.50, never below 0.35; +0.05 for audio under 2 s), beats the runner-up by
0.06, and the audio passes the 0.8 s / RMS / clipping checks. Otherwise the
result is unknown, ambiguous or too_short.

All numbers in this file were produced by the code in this folder on this
machine (x86-64 VMware guest with 20 vCPUs on an Intel i9-13900H host, no GPU;
other jobs were running). Nothing is estimated or copied from elsewhere.
Raspberry Pi timing is NOT measured.

Status: both parts are finished (benchmark runs, timing, paired comparisons,
final tables and figures); see "Status" at the end.

## Data and protocol

* Persian MSWC test split (single-word clips cut from Mozilla Common Voice,
  16 kHz, 1 s each): 120,232 clips, 3,197 speakers. Pool = speakers with at
  least 18 clips that can be split under all 5 seeds: 1,366 speakers
  (gender label: 562 male, 144 female, 4 other, 656 missing). Median 39
  clips and 18 source recordings per speaker. `results/speakers.csv`,
  `results/split_summary.json`.
* Per seed (0-4) and speaker, whole Common Voice source recordings go to
  the test side until it holds 6 clips; the rest is enrollment. Enrollment
  and test never share a source recording (asserted in `01_make_splits.py`).
* Enrollment: 3 "sentences" of 4 clips joined with 0.1 s gaps (about 4.3 s
  each, inside NexaRoom's 2-9 s prompt range). Ablations with 5x4, 3x8, 5x8.
* Test utterances: 1 clip (six per speaker), 2 clips (three), 3 clips (two),
  i.e. exactly 1, 2 and 3 s of audio.
* Speakers are split per seed into dev (1/3) and eval (2/3). All reported
  numbers are on eval speakers. Dev speakers are used only to tune thresholds
  and as the AS-norm cohort.
* Households: for every test utterance, R = 100 random households of N
  enrolled speakers that contain the true speaker (member trials) and 100
  that do not (stranger trials), N = 2, 3, 5, 10, 20.
* Values are averaged per speaker first; 95% CIs come from a 1,000-replicate
  speaker-level bootstrap over the speakers pooled across seeds; `seed_sd` is
  the SD of the five per-seed means.
* MSWC clips are zero-padded to 1 s. Over the 8,196 test clips of seed 0 the
  median share of samples that are exactly 0 is 0.62, 98% of clips contain a
  run of exact zeros of at least 0.1 s, and in the median clip 58% of the
  25 ms frames are entirely digital silence (`results/zero_padding.json`).
  Median active speech per clip is 0.40 s (`results/clip_stats.csv`). A
  microphone never delivers exact digital silence; this matters for some
  models (next sections).

## Models and configurations

Model files are from the sherpa-onnx release `speaker-recongition-models`;
SHA-256 sums match the release checksum file (`results/models.json`).
Runtime: sherpa-onnx 2.0.1 (bundles onnxruntime 1.28.2) and the onnxruntime
Python package 1.31.1.

| key | model | dim | path |
|---|---|---|---|
| `campplus_zhen_adv` | 3D-Speaker CAM++ zh+en advanced (NexaRoom default) | 192 | sherpa-onnx, as NexaRoom runs it |
| `campplus_zhen_adv_d1` | same, with a 1-LSB Gaussian noise floor added to the waveform | 192 | sherpa-onnx |
| `campplus_zhen_adv_ort131` | same model file | 192 | numpy front end with sherpa-onnx's conventions + onnxruntime 1.31.1 |
| `eres2net_base_zh` | 3D-Speaker ERes2Net-base, zh, 200k speakers | 512 | sherpa-onnx |
| `titanet_small` | NeMo TitaNet-small, en | 192 | sherpa-onnx |
| `wespeaker_campplus_vox` | WeSpeaker CAM++ VoxCeleb, as shipped | 512 | sherpa-onnx (fails, see below) |
| `wespeaker_campplus_vox_wsfe` | WeSpeaker CAM++ VoxCeleb | 512 | WeSpeaker front end + onnxruntime 1.31.1 |
| `wespeaker_resnet34_vox_wsfe` | WeSpeaker ResNet34 VoxCeleb | 256 | WeSpeaker front end + onnxruntime 1.31.1 |
| `wespeaker_resnet34_cnc_wsfe` | WeSpeaker ResNet34 CN-Celeb | 256 | WeSpeaker front end + onnxruntime 1.31.1 |

"WeSpeaker front end" (`swk/frontend.py`): int16-scaled samples, Kaldi
fbank (80 bins, 25/10 ms, Hamming), 1-LSB Gaussian dither with a fixed seed
per utterance, per-utterance mean subtraction (CMN), as in WeSpeaker's own
feature code; the ONNX graph is unchanged. "sherpa-onnx's conventions"
(`SherpaLikeOrtExtractor` in the same file): Povey window, 20-7600 Hz, sample
scale and CMN as the model metadata says, no dither.

## WeSpeaker models through sherpa-onnx: what went wrong

Run through sherpa-onnx 2.0.1 as shipped, the WeSpeaker models are close to
useless on this data. WeSpeaker CAM++ VoxCeleb gives EER 49.7% at 1 s (full
benchmark, `results/wespeaker_campplus_vox/eer.csv`) and 21.6% closed-set
accuracy with N = 5, i.e. chance. On a 100-speaker subset (seed 0, 1 s test
clips) WeSpeaker ResNet34 VoxCeleb gives EER 35.3%. Three measured causes:

1. onnxruntime version. On identical input features, onnxruntime-node 1.27.0
   returns a WeSpeaker CAM++ embedding with median cosine -0.05 to the one
   returned by onnxruntime 1.29.0 and 1.31.1 (which agree with each other to
   1.0000); sherpa-onnx 2.0.1 (bundled 1.28.2) on the same audio agrees with
   1.27.0 (0.985). The ResNet34 graphs agree across all versions
   (`results/ort_version_check.json`, 20 clips). We did not isolate the
   operator responsible; the graph's AveragePool nodes (count_include_pad = 0)
   behave the same in all versions (`results/avgpool_check.json`).
2. No CMN. The WeSpeaker files carry no `feature_normalize_type` metadata, so
   sherpa-onnx skips the per-utterance feature mean subtraction that
   WeSpeaker applies; the 3D-Speaker files do carry it. The graphs do not
   normalise internally (adding a constant to all frames changes the output,
   `results/cmn_models.json`).
3. Digital silence. WeSpeaker feeds int16-scaled audio. With no dither, the
   exactly-zero padding of MSWC clips becomes frames at the log floor
   (log of float32 eps = -15.9), a median 33 nats below the clip's loudest
   frames (`results/zero_padding.json`). WeSpeaker's training pipeline
   (`wespeaker/dataset/processor.py`, `compute_fbank` and `apply_cmvn`;
   `examples/voxceleb/v2/conf/campplus.yaml`) scales by 2^15, uses a Hamming
   window, Kaldi dither 1.0 and mean normalisation, so it never sees such
   frames.

Subset results (100 eval speakers of seed 0, cosine EER at 1 s; files in
`results/dither_check/`, `results/wespeaker_ort_check/`,
`results/cmn_validation.json`, `results/frontend_check.json`; collated in
`results/tables/diagnostics_subset.md` and `figures/fig_wespeaker_diagnostics.png`):

| configuration | WeSpeaker CAM++ vox | WeSpeaker ResNet34 vox |
|---|---|---|
| sherpa-onnx as shipped | 49.5% | 35.3% |
| sherpa-onnx + 1-LSB dither | 50.0% | 26.8% |
| sherpa-onnx + CMN metadata | 49.2% | 41.0% |
| sherpa-onnx + CMN metadata + 1-LSB dither | 50.3% | 8.3% |
| WeSpeaker front end, onnxruntime 1.31.1, no dither | 40.3% | 41.3% |
| WeSpeaker front end, onnxruntime 1.31.1, no CMN | 26.6% | 27.1% |
| WeSpeaker front end, onnxruntime 1.31.1 (dither + CMN) | 10.0% | 8.0% |

So for ResNet34 the sherpa-onnx path works once both the CMN metadata and a
noise floor are present; for CAM++ it does not, because of the runtime (cause
1). WeSpeaker ResNet34 CN-Celeb through sherpa-onnx as shipped: 16.4%, with
the noise floor 10.7%.

For reference on the same subset (sherpa-onnx, without / with the 1-LSB
noise floor): 3D-Speaker CAM++ (NexaRoom default) 8.0% / 7.5%,
ERes2Net-base 6.5% / 5.7%, TitaNet-small 11.7% / 11.7%. The noise floor
helps the 3D-Speaker models a little and is essential for the WeSpeaker
models.
The full benchmark therefore uses the `_wsfe` configurations for the
WeSpeaker models and keeps `wespeaker_campplus_vox` (as shipped) only to
document the failure. The as-shipped ResNet34 runs were stopped part-way and
are not evaluated (partial embeddings remain in `cache/emb/`).

## The onnxruntime version also matters for NexaRoom's default model

On identical input features, onnxruntime 1.27.0 and 1.29.0 / 1.31.1 return
different embeddings for the 3D-Speaker CAM++ graph too (median cosine 0.887
over 20 clips; sherpa-onnx 2.0.1's output is closest to 1.27.0, 0.971, which
is the same level of agreement as for graphs that no version changes, 0.97-0.98,
where the residual comes from small front-end differences). ERes2Net and the
ResNet34 graphs are unaffected (cosine 1.000; `results/ort_version_check.json`).

One concrete change explains at least part of it. The 3D-Speaker CAM++ graph
has 52 AveragePool nodes with kernel 100, stride 100, ceil_mode = 1 and
count_include_pad = 1 (the context-aware masking segments). On an all-ones
input of length 49, onnxruntime 1.27.0 returns 0.49 (it divides the partial
last window by the full kernel), 1.29.0 and 1.31.1 return 1.00 (they divide by
the frames present, as PyTorch's avg_pool1d does, checked with PyTorch 2.15
in the same file); with count_include_pad = 0,
as in the WeSpeaker CAM++ graph, all versions agree (`results/avgpool_check.json`,
`results/onnx_ops.json`). Short utterances are the most exposed, because
the partial last window covers a larger share of the input. We did not check
whether other operators also changed between versions.

Subset effect for NexaRoom's default model, same numpy features fed to both
runtimes (100 eval speakers, seed 0, 1 s; `results/runtime_effect_subset/`):

| runtime | EER | closed-set acc., N = 5 |
|---|---|---|
| sherpa-onnx 2.0.1 (bundled onnxruntime 1.28.2), its own front end | 8.0% | 94.7% |
| onnxruntime-node 1.27.0, numpy front end | 7.8% | 94.7% |
| onnxruntime 1.31.1 (Python), numpy front end | 7.1% | 96.3% |

The full benchmark repeats this as `campplus_zhen_adv_ort131` (paired
comparison below). Embedding agreement across runtime versions is in
`results/tables/ort_version_agreement.md`.

## Results (eval speakers, 5 seeds, 95% speaker-bootstrap CI)

Every number below is in a file under `results/tables/` (index:
`results/tables/INDEX.md`), written by `experiments/05_report.py`,
`experiments/19_paired.py` and `experiments/14_summary_tables.py` from the
per-configuration folders `results/<config>/`. "pp" = percentage points.
"Paired" differences use the same eval speakers for both sides and one shared
speaker-resampling matrix (1,000 replicates); `19_paired.py` first re-runs the
N = 5 evaluation and checks that it reproduces `main.csv` and `eer.csv`
(largest difference 1e-7, `results/paired_check.json`).

The six main configurations are the six models, each through the path that
works for it: 3D-Speaker CAM++ (NexaRoom default, sherpa-onnx), 3D-Speaker
ERes2Net-base and NeMo TitaNet-small (sherpa-onnx), and the three WeSpeaker
models with their own front end (`_wsfe`). Two variants of NexaRoom's model
(noise floor, newer onnxruntime) isolate runtime effects; WeSpeaker CAM++
through sherpa-onnx as shipped documents the failure.

### Headline table (household of N = 5 where N applies)

From `results/tables/headline.md`. NexaRoom rule = NexaRoom's decision rule
with its own defaults; "AS-norm @ dev 1% FA" = AS-norm cosine with a
threshold tuned on dev speakers for 1% stranger acceptance. CPU ms = median
single-thread time for one utterance of that length.

| model | test | EER cosine % | EER AS-norm % | closed-set N=5 % | closed-set N=20 % | NexaRoom rule N=5: correct / wrong / stranger % | AS-norm @ dev 1% FA, N=5: correct % | CPU ms |
|---|---|---|---|---|---|---|---|---|
| 3D-Speaker CAM++ (NexaRoom default) | 1 s | 8.44 | 7.20 | 94.5 | 87.0 | 37.9 / 0.05 / 0.26 | 65.0 | 22.5 |
| 3D-Speaker CAM++ (NexaRoom default) | 2 s | 3.10 | 2.28 | 99.1 | 97.6 | 91.2 / 0.06 / 3.40 | 90.6 | 37.7 |
| 3D-Speaker CAM++ (NexaRoom default) | 3 s | 2.07 | 1.56 | 99.5 | 98.7 | 96.3 / 0.05 / 5.61 | 94.4 | 54.3 |
| 3D-Speaker ERes2Net-base | 1 s | 6.58 | 5.03 | 97.2 | 93.0 | 73.8 / 0.13 / 2.45 | 76.4 | 52.2 |
| 3D-Speaker ERes2Net-base | 2 s | 2.80 | 2.01 | 99.3 | 98.2 | 96.6 / 0.11 / 14.01 | 92.9 | 104.8 |
| 3D-Speaker ERes2Net-base | 3 s | 1.78 | 1.22 | 99.6 | 99.1 | 98.5 / 0.07 / 18.55 | 96.8 | 159.5 |
| NeMo TitaNet-small | 1 s | 10.62 | 9.02 | 92.3 | 82.4 | 19.1 / 0.06 / 0.21 | 55.0 | 18.0 |
| NeMo TitaNet-small | 2 s | 4.28 | 3.33 | 98.4 | 95.6 | 81.5 / 0.12 / 2.50 | 85.0 | 31.8 |
| NeMo TitaNet-small | 3 s | 2.66 | 1.99 | 99.3 | 98.1 | 93.7 / 0.08 / 3.88 | 93.7 | 45.8 |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | 1 s | 10.84 | 9.55 | 91.6 | 81.0 | 21.5 / 0.12 / 0.37 | 49.7 | 22.0 |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | 2 s | 5.21 | 4.45 | 97.5 | 93.4 | 80.5 / 0.24 / 3.74 | 76.2 | 37.2 |
| WeSpeaker CAM++ VoxCeleb (WeSpeaker front end) | 3 s | 3.08 | 2.54 | 98.8 | 96.9 | 93.2 / 0.15 / 6.25 | 87.5 | 54.9 |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | 1 s | 8.82 | 7.64 | 93.4 | 84.8 | 31.7 / 0.20 / 0.55 | 57.6 | 50.4 |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | 2 s | 3.81 | 3.21 | 98.3 | 95.4 | 87.4 / 0.19 / 4.05 | 83.2 | 98.3 |
| WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end) | 3 s | 2.24 | 1.82 | 99.3 | 98.0 | 95.8 / 0.08 / 6.14 | 92.0 | 147.9 |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | 1 s | 7.93 | 6.29 | 96.0 | 89.8 | 65.9 / 0.23 / 3.09 | 67.4 | 51.0 |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | 2 s | 3.68 | 2.68 | 98.9 | 97.0 | 94.8 / 0.17 / 16.23 | 88.2 | 98.3 |
| WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end) | 3 s | 2.38 | 1.55 | 99.5 | 98.6 | 97.7 / 0.09 / 21.31 | 94.5 | 147.9 |
| 3D-Speaker CAM++, 1-LSB noise floor | 1 s | 7.72 | 6.81 | 94.7 | 87.3 | 40.0 / 0.05 / 0.25 | 66.6 | 22.5 |
| 3D-Speaker CAM++, 1-LSB noise floor | 2 s | 2.75 | 2.18 | 99.2 | 97.7 | 91.1 / 0.06 / 2.74 | 91.2 | 37.7 |
| 3D-Speaker CAM++, 1-LSB noise floor | 3 s | 1.88 | 1.57 | 99.5 | 98.7 | 96.3 / 0.05 / 4.69 | 94.6 | 54.3 |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | 1 s | 7.34 | 6.01 | 96.1 | 90.2 | 42.7 / 0.04 / 0.27 | 70.5 | 22.3 |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | 2 s | 3.05 | 2.30 | 99.2 | 97.6 | 91.1 / 0.06 / 3.45 | 90.6 | 38.5 |
| 3D-Speaker CAM++, onnxruntime 1.31.1 | 3 s | 1.70 | 1.25 | 99.6 | 99.1 | 97.3 / 0.03 / 5.73 | 95.9 | 55.0 |
| WeSpeaker CAM++ VoxCeleb, sherpa-onnx as shipped | 1 s | 49.73 | 50.00 | 21.6 | 5.7 | 0.0 / 0.00 / 0.00 | 0.2 | 21.7 |
| WeSpeaker CAM++ VoxCeleb, sherpa-onnx as shipped | 2 s | 48.31 | 50.00 | 33.6 | 11.9 | 0.0 / 0.00 / 0.00 | 0.3 | 37.3 |
| WeSpeaker CAM++ VoxCeleb, sherpa-onnx as shipped | 3 s | 47.82 | 50.00 | 25.7 | 7.7 | 0.2 / 0.04 / 0.10 | 0.3 | 54.2 |

### Findings

1. **Length dominates.** Across the six main configurations, cosine EER is
   6.58-10.84% at 1 s, 2.80-5.21% at 2 s and 1.78-3.08% at 3 s;
   closed-set accuracy among N = 5 enrolled speakers is 91.6-97.2% at 1 s and
   among N = 20 it is 81.0-93.0% (`eer_all.md`, `closed_set_all.md`,
   `figures/fig_closed_set_vs_N.png`, `figures/fig_eer.png`).
2. **Model ranking at 1 s (paired, vs NexaRoom's default model).**
   ERes2Net-base: EER -1.86 pp [-2.09, -1.60], N = 20 accuracy +6.02 pp
   [+5.63, +6.41]. WeSpeaker ResNet34 CN-Celeb: EER -0.50 pp [-0.73, -0.23].
   WeSpeaker ResNet34 VoxCeleb: +0.38 pp [+0.12, +0.63]. TitaNet-small:
   +2.18 pp [+1.90, +2.48]. WeSpeaker CAM++ VoxCeleb: +2.41 pp [+2.15, +2.68].
   ERes2Net-base also beats the best WeSpeaker model (ResNet34 CN-Celeb) by
   1.36 pp EER [1.11, 1.62] at 1 s. At 3 s ERes2Net-base and NexaRoom's model
   run with onnxruntime 1.31.1 do not differ (+0.09 pp [-0.08, +0.32])
   (`paired_vs_default.md`, `paired_differences.md`).
3. **Cost.** ERes2Net-base and the ResNet34 models need 50-52 ms of one vCPU
   per 1 s utterance; the CAM++ models 22-23 ms; TitaNet-small 18 ms
   (`timing_compact.md`, `figures/fig_eer_vs_cpu.png`). Model files are
   26.5-40.3 MB.
4. **AS-norm helps verification but not ranking.** With a cohort of dev
   speakers, AS-norm lowers the 1 s EER of every working configuration by
   0.91 to 1.64 pp (paired; all CIs exclude 0) and, at a dev-tuned 1% stranger
   acceptance, names 8.65 to 16.77 pp more member wakes at 1 s. It does not
   raise closed-set accuracy (-0.48 to +0.02 pp at 1 s)
   (`paired_differences.md`, rows "AS-norm minus cosine").
5. **NexaRoom's rule is conservative at 1 s because of its short-audio
   penalty.** With the default model and N = 5 it names 37.9% of member wakes
   at 1 s, 91.2% at 2 s and 96.3% at 3 s, names the wrong member in
   0.05-0.06%, and accepts 0.26%, 3.40% and 5.61% of stranger wakes. Adding
   the rule's elements one at a time (`rule_ladder_N5.md`,
   `figures/fig_rule_ladder.png`): the +0.05 penalty for audio under 2 s cuts
   1 s naming from 56.8% to 38.1% and stranger acceptance from 1.39% to
   0.28%; the 0.06 runner-up margin removes most wrong-member answers at 2 s
   (0.22% -> 0.06%) and 3 s (0.19% -> 0.05%). A plain cosine threshold
   matched to the rule's dev stranger rate names 39.8% [38.6, 41.0] at 1 s
   with 0.10% wrong and 0.25% strangers (`rules_all.md`).
6. **The rule's score adds nothing once thresholds are tuned.** At a dev-tuned
   1% stranger acceptance, NexaRoom's blend score with calibrated thresholds
   names fewer member wakes than plain cosine in every configuration:
   -0.18 to -2.36 pp at 1 s (CI includes 0 only for ResNet34 CN-Celeb) and up
   to -10.04 pp at 2 s (`paired_differences.md`). The blend score also loses
   0.08-0.18 pp of closed-set accuracy at 1 s.
7. **Fixed thresholds do not transfer between models.** With the rule's
   defaults, ERes2Net-base accepts 2.45%, 14.01% and 18.55% of strangers at
   1, 2 and 3 s, WeSpeaker ResNet34 CN-Celeb 3.09%, 16.23% and 21.31%, against
   0.26%, 3.40% and 5.61% for the default model. The 99th percentile of the
   non-target cosine at 1 s is 0.472 for the default model, 0.552 for
   ERes2Net-base and 0.566 for ResNet34 CN-Celeb (`score_distributions.md`).
   Stranger acceptance also grows with household size: for the default model
   at 1 s from 0.11% (N = 2) to 0.79% (N = 20), at 3 s from 2.74% to 11.67%
   (`nexaroom_default_vs_N.md`, `figures/fig_nexaroom_default_vs_N.png`).
8. **Runtime path of the same weights.** NexaRoom's model run by onnxruntime
   1.31.1 instead of sherpa-onnx 2.0.1 (paired): EER at 1 s -1.09 pp
   [-1.24, -0.92], N = 20 accuracy +3.21 pp [+2.95, +3.46], NexaRoom rule
   names +4.74 pp [+4.17, +5.35] more member wakes with unchanged stranger
   acceptance (+0.01 pp [-0.00, +0.02]); at 2 s no EER difference
   (-0.05 pp [-0.15, +0.06]); at 3 s -0.38 pp [-0.49, -0.27]. A 1-LSB noise
   floor through sherpa-onnx: EER at 1 s -0.72 pp [-0.89, -0.55], and the
   rule's stranger acceptance drops by 0.67 pp [0.59, 0.74] at 2 s and
   0.91 pp [0.80, 1.02] at 3 s (`figures/fig_runtime_variants.png`).
9. **Gender.** With same-gender non-targets the EER is 1.87-2.68 times the
   cross-gender EER at 1 s and 2.83-4.13 times at 3 s (six main
   configurations, `gender_eer_all.md`, `figures/fig_gender.png`). For the
   default model at 1 s, closed-set accuracy is 92.6% in households of the
   speaker's own gender, 93.9% in random ones and 97.3% in cross-gender ones;
   the rule accepts 0.45% of strangers when the household shares the
   stranger's gender and 0.03% when it does not (`gender_all.md`). Female
   test speakers (n = 144) reach 96.3% [95.4, 97.0] at 1 s, male (n = 560)
   93.3% [92.7, 93.9] (`gender_by_target.md`); male speakers also face more
   same-gender competitors, so this is not a per-gender model effect.
10. **Enrollment size** (default model, 587 speakers with >= 40 enrollment
    clips, `ablation_enrollment.md`): going from 3 sentences of 4 clips to 5
    of 8 lowers the 1 s EER from 8.40% to 7.53%, raises closed-set accuracy
    from 94.6% to 95.9% and the rule's 1 s naming from 35.8% to 54.5%, while
    stranger acceptance rises from 0.26% to 0.67% (the calibrated thresholds
    stay at 0.495-0.500). At 3 s stranger acceptance rises from 5.31% to 9.85%.
11. **Silence trimming hurts** (default model, 1 s, `ablation_trim.md`):
    trimming the test clip raises the EER from 8.44% to 10.35%
    [9.99, 10.72]; trimming only the enrollment gives 8.68% [8.33, 8.99].
12. **Fixed wake word** (default model, 1 s, `fixed_word.md`): for four of
    five frequent words a test clip of that word gives a lower EER than a
    random word of the same speakers (5.93-8.81% vs 9.36-10.22%; for برای,
    خیلی and باید the CIs overlap). The very short word این ("in", median 0.185 s of
    speech) is much worse (23.56% vs 11.27%). These are not paired tests.
13. **Errors concentrate on a few speakers.** At 1 s and N = 5 the worst 10%
    of speakers make 29.4-47.5% of all closed-set errors across the six main
    configurations; at 3 s 79.0-98.5% (`error_concentration_N5.md`).
14. **Audio check.** NexaRoom's audio check (RMS < 80 etc.) rejects 0.82-1.24%
    of 1 s test clips per seed and 0.17-0.34% of enrollment sentences
    (`data_summary.md`).

### CPU time (x86 only)

One pinned vCPU, `num_threads = 1`, 30 timed runs after 3 warm-up runs per
utterance length, run twice (`timing.md`, `timing_compact.md`,
`timing_repeatability.md`, `results/timing.csv`, `results/timing_rep2.csv`,
`results/timing_env*.json`, `results/timing_host.json`). Median CPU ms per
utterance:

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

The two runs differ by a median 2.1% and at most 12.7% per row. The machine
is a VMware guest; which host core type (performance or efficiency) backs the
pinned vCPU is not visible from inside it. Other jobs were running (load
average about 2 on 20 vCPUs). NexaRoom's own scoring step for a 5-person
household takes about 0.1 ms. The `_wsfe` and `_ort131` rows include the numpy
front end. Raspberry Pi or any ARM timing is NOT measured.

### Figures (`figures/`, PNG at 200 dpi + PDF)

| file | content |
|---|---|
| `fig_closed_set_vs_N` | closed-set accuracy vs household size, six main configurations, 1/2/3 s |
| `fig_eer` | EER with cosine and AS-norm scoring, six main configurations |
| `fig_open_set_tradeoff` | member wakes named vs stranger wakes accepted (cosine threshold swept), N = 5, with the NexaRoom-rule operating point of each model |
| `fig_open_set_tradeoff_campplus` | same for the default model: cosine, NexaRoom rule and AS-norm families |
| `fig_nexaroom_default_vs_N` | NexaRoom rule with defaults: member naming and stranger acceptance vs N |
| `fig_rule_ladder` | NexaRoom's rule built up one element at a time (default model) |
| `fig_eer_vs_cpu` | 1 s EER vs single-thread CPU time |
| `fig_gender` | EER with same- vs cross-gender non-targets |
| `fig_enrollment_size` | enrollment size ablation (default model) |
| `fig_runtime_variants` | same CAM++ weights through sherpa-onnx, with a noise floor, and through onnxruntime 1.31.1 |
| `fig_wespeaker_diagnostics` | WeSpeaker models: front-end / runtime diagnostics on the 100-speaker subset |

Older drafts of tables and figures (written at 16:10 from two models) were
moved to `cache/stale_1610/` and must not be used.

## How to reproduce

```
source /home/ali/Desktop/Articles/.venv/bin/activate
cd /home/ali/Desktop/Articles/artifacts/speaker-id-wake
python experiments/01_make_splits.py          # pool, splits, utterance specs
python experiments/08_make_cmn_models.py      # CMN-flagged WeSpeaker copies (diagnostic only)
python experiments/00_models.py               # model table + checksums
python experiments/02_extract.py --set main --models <keys> --workers 7   # resumable
python experiments/02_extract.py --set ablation --models campplus_zhen_adv
python experiments/03_evaluate.py --model <key> [--ablations]
python experiments/06_clip_stats.py
python experiments/09_validate_cmn.py; python experiments/11_dither_check.py <keys>
python experiments/12_wespeaker_ort_check.py <keys>; python experiments/10_frontend_check.py
python experiments/13_ort_version_check.py <dir with v127/v129/v131 onnxruntime-node installs>
python experiments/15_zero_padding_stats.py; python experiments/16_onnx_ops.py
python experiments/17_avgpool_check.py <same dir>
python experiments/18_runtime_effect_subset.py <same dir> campplus_zhen_adv
python experiments/07_check_results.py        # completeness checks -> results/check_results.json
python experiments/04_timing.py               # single-thread CPU time (x86 only)
python experiments/04_timing.py --rep 2       # repeat run -> results/timing_rep2.csv
python experiments/19_paired.py --dump <keys>; python experiments/19_paired.py --compare
python experiments/14_summary_tables.py       # results/tables/summary.md
python experiments/05_report.py               # all final tables + figures, results/tables/INDEX.md
python tests/test_rules.py; python tests/test_sim.py   # rule re-implementation vs NexaRoom code
```

`experiments/run_*.sh` are the queue scripts actually used (they document
each interruption and resume). Logs are in `logs/`. onnxruntime-node was
installed with `npm install --ignore-scripts onnxruntime-node@<version>` into
three scratch folders outside this repository; `experiments/ort_node_run.js`
and `experiments/ort_node_batch.js` run a graph with it. `19_paired.py`
stores its per-configuration intermediate files in `cache/paired/`.

## Limitations

* MSWC clips are single words cut automatically from read Common Voice
  sentences, not real wake-word recordings from a device microphone; the
  "wake word" here is a random Persian word (a fixed-word ablation uses five
  frequent words). Median active speech per 1 s clip is 0.40 s.
* Same-speaker enrollment and test come from different Common Voice
  recordings, but often the same microphone and session conditions per
  contributor; cross-device conditions, far-field audio and noise are not
  covered.
* Households are simulated by sampling other Common Voice speakers; real
  households share accents, rooms and often relatives' voices.
* Gender labels exist for about half of the speakers and are 4:1 male.
* The NexaRoom rule was re-implemented in numpy (`swk/rules.py`) and checked
  against the original code on random embeddings (`tests/`); it was not run
  end to end inside NexaRoom.
* Training data of the models is given as the release file names state it;
  overlap with Common Voice Persian was not checked.
* Timing is single-thread on one vCPU of an x86 VM under background load.
  Raspberry Pi (or any ARM) timing is not measured.

## Status

Part 1 and part 2 are finished.

* Full benchmark, all five seeds (`results/<key>/`, each with `eval.done`;
  `results/check_results.json` confirms complete embeddings, no NaN estimates):
  `campplus_zhen_adv` (with ablations), `campplus_zhen_adv_d1`,
  `campplus_zhen_adv_ort131`, `eres2net_base_zh`, `titanet_small`,
  `wespeaker_campplus_vox` (as shipped), `wespeaker_campplus_vox_wsfe`,
  `wespeaker_resnet34_vox_wsfe`, `wespeaker_resnet34_cnc_wsfe`.
* Part 2 added: a repeat timing run (`results/timing_rep2.csv`) and the host
  description (`results/timing_host.json`); paired same-speaker comparisons
  (`experiments/19_paired.py`, `results/tables/paired_differences.*`,
  `results/paired_check.json`); the final cross-model tables
  (`results/tables/`, 27 tables + `summary.md`) and 11 figures (`figures/`).
* Not evaluated on the full benchmark: the as-shipped sherpa-onnx runs of the
  two WeSpeaker ResNet34 models (stopped after the subset diagnosis; partial
  embeddings in `cache/emb/`), and the `*_cmn` copies (diagnostic only).
  Ablations (enrollment size, trimming, fixed word) exist only for
  `campplus_zhen_adv`.
* Earlier results computed before 16:22 are kept in
  `cache/old_results_pre_1622/`; the re-run with the current
  `03_evaluate.py` reproduced them exactly (max |difference| 0 in main.csv).
  The part-1 README is kept as `cache/README_part1_backup.md`.
* No manuscript was written in this step.

## Revision (11 Nov 2026): what was added after the internal review

Everything in this section was produced by the scripts named here, on this machine, after the
review of the first manuscript. Pre-revision result files are kept in
`cache/results_pre_revision/` (and `cache/paired_pre_revision/`); re-running the changed
`03_evaluate.py` reproduced every pre-revision point estimate exactly (max |difference| 0 in
`main.csv`, `eer.csv`, `ablations.csv`, `fixed_word.csv`, `gender.csv`); only the bootstrap
intervals moved slightly (at most 0.18 pp) because of change 2 below.

Changes to the evaluation (`experiments/03_evaluate.py`, `experiments/19_paired.py`):

1. New dev-tuned decisions (1% and 5% stranger acceptance): `cosm` (cosine + NexaRoom's 0.06
   runner-up margin), `blendm` (blend score + margin, no calibration, no short-audio penalty)
   and `asnormm` (AS-norm + margin, the margin measured on the cosines of the two members ranked
   highest by AS-norm); `cosm@dev_matched` (cosine + margin at the rule's own dev stranger rate).
   This separates the blend score from the margin, which the first analysis did not.
2. The bootstrap now resamples the 1,362 eval speakers (eval in at least one seed); before, it
   drew from all 1,368 speaker index slots, six of them empty.
3. `--backend cw`: dev-trained centring + whitening (ridge 1e-3 x mean eigenvalue) + length
   normalisation before scoring, per seed; results in `results/<model>__cw/`.

New configurations (full benchmark; `swk/data.py`):

| key | what |
|---|---|
| `campplus_zhen_adv_np128` | NumPy front end with sherpa-onnx's conventions + the onnxruntime 1.28.2 library bundled in sherpa-onnx 2.0.1, called through the C API (`swk/ortc.py`) |
| `campplus_zhen_adv_cip0` | sherpa-onnx 2.0.1 with the CAM++ file patched: the 52 AveragePool nodes get count_include_pad = 0 (one byte each; `data/models/patched/`) |
| `campplus_zhen_adv_cip0_d1` | the same + 1-LSB noise floor (matched condition) |
| `eres2net_base_zh_d1`, `titanet_small_d1` | sherpa-onnx + 1-LSB noise floor (matched condition) |
| `campplus_vox_cip0_d1`, `eres2net_vox_d1` | two more 3D-Speaker models trained on VoxCeleb (same release, SHA-256 checked, `results/models_rev.json`), matched condition only |
| `<model>__cw` | the six main configurations with the dev-trained centring + whitening back end |

New scripts and result files:

* `experiments/20_runtime_isolation.py` -> `results/runtime_isolation.json`: ctypes wrapper
  validated against the Python package (max relative difference 0 over 280 runs); one-node
  AveragePool on onnxruntime 1.28.2 (count_include_pad = 1, L = 49 -> 0.49, as 1.27.0);
  embedding agreement on identical features for 1 s and 3 s inputs across 1.28.2 / 1.31.1 and
  sherpa-onnx, original and patched CAM++ graph. Patched graph on 1.28.2 vs original on 1.31.1:
  median cosine 1.000 at 1 s and 3 s, i.e. the pooling change explains the whole runtime
  difference of this graph. The WeSpeaker CAM++ difference (-0.05) is not explained by pooling.
* `experiments/21_speech_content.py` -> `results/speech_content.json`: speech per input
  (median 0.40 / 0.80 / 1.21 s for 1 / 2 / 3 s inputs, 1.63 s per 4.3 s enrollment sentence);
  2 s inputs are 1.97-2.00 s (18 of 19,682 below 2.0 s); 28,692 of 30,190 trimmed 1 s clips
  are shorter than 0.8 s.
* `experiments/22_twoway_bootstrap.py` -> `results/twoway_bootstrap.csv/.json`: crossed (test x
  enrolled speaker) bootstrap of EERs and EER differences, 500 replicates.
* `experiments/23_error_concentration.py` -> `results/error_concentration_null.csv`: permutation
  null for "errors concentrate on few speakers".
* `experiments/24_ablation_paired.py` -> `results/ablation_paired.csv`: paired trimming and
  fixed-word differences.
* `experiments/25_timing_parts.py` -> `results/timing_parts_rev.csv`: graph-only, NumPy front end
  and sherpa-onnx end-to-end CPU time in one pinned process (under heavier load than
  `04_timing.py`; compare only within the file).
* `experiments/26_models_rev.py` -> `results/models_rev.json`; `results/release_assets_rev.json`
  (asset list of the model release with sizes, from the GitHub API).
* Queue scripts: `run_extract_rev_a.sh`, `run_extract_rev_b.sh` (stopped after its first step
  had started; replaced by `run_extract_rev_b2.sh`), `run_eval_rev.sh` (lane 1 stopped and
  replaced by `run_eval_rev_a.sh` to evaluate the decomposition configurations first),
  `run_eval_rev3.sh`, `run_eval_rev_lib.sh`.

## Second revision pass (11 Nov 2026, after the machine rebooted at 08:10)

The reboot stopped the revision queues part-way (logs end about 01:11). What was redone or added:

* Re-evaluated with the revised `03_evaluate.py` (eval-speaker bootstrap population of 1,362, margin
  rules, one-threshold-for-all-lengths rules): `eres2net_base_zh`, `titanet_small`,
  `campplus_zhen_adv_d1`, `wespeaker_campplus_vox` and the not yet evaluated
  `campplus_zhen_adv_np128`. Every configuration used in the paper now has `eval_rev.done` and
  `eval_meta.json` with `bootstrap_population` 1362. Queue: `experiments/run_eval_rev_f.sh` (lanes 1, 2).
* Back end `__cw` for all six main models (`results/<model>__cw/`). Because the dev speakers that
  fit the transform also tune the thresholds, dev-tuned thresholds overshoot on eval speakers
  (ERes2Net-base: 2.10-2.71% realized for a 1% target), so the back end is compared only by
  EER / minDCF / closed-set accuracy and by `experiments/27_curves_at_fa.py`
  (`results/curves_at_fa1.csv`, naming at 1% stranger acceptance read from the eval curves).
* Matched condition (1-LSB noise floor + a runtime path that agrees with onnxruntime 1.31.1):
  `campplus_zhen_adv_cip0_d1`, `titanet_small_d1`, `eres2net_base_zh_d1`, finished by
  `experiments/run_extract_rev_c.sh` (6 workers, resumed from the partial embedding files). The
  two 3D-Speaker VoxCeleb models (`campplus_vox_cip0_d1`, `eres2net_vox_d1`), the trimming specs of
  `eres2net_base_zh` / `campplus_zhen_adv_cip0` and the other ablation specs of
  `campplus_zhen_adv_cip0` were dropped for lack of time (not evaluated, not reported).
* `experiments/22_twoway_bootstrap.py` rewritten (second version; the first is kept as
  `cache/22_twoway_bootstrap_part1.py`): 1, 2 and 3 s, all main, matched, inference-path and
  back-end configurations, one cache file per configuration (`cache/twoway/<cfg>_B500_d123.npz`),
  same replicate weights as the first version (it reproduces the first version's replicates
  exactly). Queue: `experiments/run_twoway_rev.sh`. The paper now uses two-way intervals for
  EERs and EER differences.
* `experiments/29_release_counts.py` -> `results/release_counts.json`: the model release has
  21 ONNX models, 14 of at most 40.3 MB and 7 of 71.4-220.6 MB (the earlier note "23 / 15" was wrong).
* `experiments/30_sherpa_1x_check.py` + `experiments/sherpa_node_embed.js` ->
  `results/sherpa_1x_check.json`: NexaRoom pins `sherpa-onnx>=1.13,<2`; the newest such release,
  1.13.8 (npm `sherpa-onnx-node`, installed with `--ignore-scripts` in a scratch folder; the PyPI
  wheel cannot be downloaded here), bundles onnxruntime 1.28.2, returns 0.49 on the one-node
  AveragePool test (L = 49) and gives the same embeddings as sherpa-onnx 2.0.1 for CAM++,
  ERes2Net-base and TitaNet-small (minimum cosine 0.99999993 over 20 one-second and 20
  three-second inputs).
* `experiments/19_paired.py --compare` re-run on all revised dumps.
