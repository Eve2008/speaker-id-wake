# results/tables index

Written by experiments/05_report.py (and 19_paired.py, 14_summary_tables.py) from the per-configuration result files. Percentages unless stated; brackets are 95% speaker-level bootstrap CIs.

| file | content |
|---|---|
| `data_summary.csv` / `data_summary.md` | Data, pool, protocol and clip statistics |
| `models.csv` / `models.md` | Configurations: model files, runtime path, size, dimension |
| `eer_all.csv` / `eer_all.md` | EER and minDCF per configuration, duration and scoring (95% CI) |
| `score_distributions.csv` / `score_distributions.md` | Cosine score quantiles (target / non-target), pooled eval trials |
| `closed_set_all.csv` / `closed_set_all.md` | Closed-set identification accuracy, all N, durations and scores (95% CI) |
| `closed_set_by_score.csv` / `closed_set_by_score.md` | Closed-set accuracy by scoring function, N = 5 and 20 |
| `rules_all.csv` / `rules_all.md` | Decision-rule outcomes (NexaRoom default, cosine 0.50, ladder), all N |
| `nexaroom_default_vs_N.csv` / `nexaroom_default_vs_N.md` | NexaRoom default rule vs household size |
| `rule_ladder_N5.csv` / `rule_ladder_N5.md` | Step-by-step decomposition of NexaRoom's rule, N = 5 |
| `open_set_tuned_all.csv` / `open_set_tuned_all.md` | Open set with dev-tuned thresholds (1%/5% stranger FA), all N |
| `open_set_pooled_threshold_N5.csv` / `open_set_pooled_threshold_N5.md` | One threshold for all durations (dev-tuned), N = 5 |
| `tuned_threshold_values_N5.csv` / `tuned_threshold_values_N5.md` | Dev-tuned threshold values (mean, min, max over seeds), N = 5 |
| `gender_eer_all.csv` / `gender_eer_all.md` | EER with same- vs cross-gender non-targets |
| `gender_all.csv` / `gender_all.md` | Closed-set accuracy and NexaRoom stranger FA by household gender composition |
| `gender_by_target.csv` / `gender_by_target.md` | Closed-set accuracy by the test speaker's gender label, N = 5 |
| `timing.csv` / `timing.md` | Single-thread CPU time per utterance and model (x86 VM only) |
| `timing_repeatability.csv` / `timing_repeatability.md` | Run-to-run spread of the CPU-time medians (two full runs) |
| `timing_compact.csv` / `timing_compact.md` | CPU ms per utterance for each evaluated configuration |
| `ablation_enrollment.csv` / `ablation_enrollment.md` | Enrollment size ablation (NexaRoom default model) |
| `ablation_trim.csv` / `ablation_trim.md` | Silence trimming of enrollment / test audio (NexaRoom default model, 1 s) |
| `fixed_word.csv` / `fixed_word.md` | Fixed wake word vs random word, same speakers (NexaRoom default model, 1 s) |
| `diagnostics_subset.csv` / `diagnostics_subset.md` | Front-end / runtime diagnostics on a 100-speaker subset (seed 0, 1 s) |
| `ort_version_agreement.csv` / `ort_version_agreement.md` | Embedding agreement across onnxruntime versions (20 clips) |
| `paired_differences.csv` / `paired_differences.md` | Paired (same-speaker) bootstrap differences between configurations and rules |
| `paired_vs_default.csv` / `paired_vs_default.md` | Paired differences of each configuration vs the NexaRoom default model |
| `error_concentration_N5.csv` / `error_concentration_N5.md` | How closed-set errors concentrate on few speakers (N = 5, cosine) |
| `headline.csv` / `headline.md` | One-line-per-condition summary of every configuration |
| `summary.md` | Compact cross-model summary (14_summary_tables.py) |
