"""Do closed-set errors concentrate on speakers more than chance would make them? (revision)

For the six main configurations, N = 5 and 1 s / 3 s test utterances, this
recomputes the closed-set outcome of every eval test utterance in the 100
member households of 03_evaluate.py (same household generator), giving an
error rate per utterance. Reported:
  - speaker level: share of all errors made by the worst 10% of speakers
    (speaker value = mean over its utterances and over the seeds in which it is
    eval, as in results/tables/error_concentration_N5.md);
  - utterance level: share of all errors made by the worst 10% of utterances;
  - permutation null: utterances are shuffled across speakers within each seed
    (each speaker keeps its number of utterances), which keeps the utterance-level
    variation but removes any speaker effect; 200 permutations, mean and 95th
    percentile of the speaker-level share.

Output: results/error_concentration_null.csv
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swk import data, rules, splits  # noqa: E402

_spec = importlib.util.spec_from_file_location("ev", Path(__file__).resolve().parent / "03_evaluate.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

MAIN = ["campplus_zhen_adv", "eres2net_base_zh", "titanet_small", "wespeaker_campplus_vox_wsfe",
        "wespeaker_resnet34_vox_wsfe", "wespeaker_resnet34_cnc_wsfe"]
N_H = 5
N_PERM = 200


def top_share(err: np.ndarray, frac: float = 0.1) -> float:
    e = np.sort(err)[::-1]
    return float(e[: max(1, int(round(frac * e.size)))].sum() / max(e.sum(), 1e-12))


def speaker_values(err_by_seed: list[tuple[np.ndarray, np.ndarray]], n: int) -> np.ndarray:
    s = np.zeros((len(err_by_seed), n))
    c = np.zeros((len(err_by_seed), n))
    for i, (spk, e) in enumerate(err_by_seed):
        np.add.at(s[i], spk, e)
        np.add.at(c[i], spk, 1)
    with np.errstate(invalid="ignore"):
        per_seed = s / c
    v = np.nanmean(per_seed, axis=0)
    return v[~np.isnan(v)]


def main() -> None:
    rows = []
    rng = np.random.default_rng(5)
    for m in MAIN:
        st = ev.Store(m)
        for dur in (1, 3):
            by_seed = []
            for seed in splits.SEEDS:
                sp = st.splits[seed]
                ev_ids = np.array(sorted(k for k, v in sp.items() if not v["dev"]))
                enr = st.enroll(seed, "3x4", ev_ids)
                q, owner, *_ = st.tests(seed, f"test{dur}", ev_ids, len(splits.TEST_GROUPS[dur]))
                _, cos = rules.profile_score_matrix(q, enr)
                hrng = np.random.default_rng([seed, N_H, 0, 7])  # as 03_evaluate.py (eval set, N = 5)
                comps = ev.households(hrng, len(ev_ids), N_H - 1, ev.R)
                cand = np.concatenate([np.broadcast_to(owner[:, None, None], (owner.size, ev.R, 1)), comps[owner]], -1)
                sc = ev.gather(cos, owner, cand)
                err = 1.0 - (np.argmax(sc, axis=-1) == 0).mean(axis=1)  # per utterance
                by_seed.append((ev_ids[owner], err))
            n = int(st.n_spk)
            v = speaker_values(by_seed, n)
            obs = top_share(v)  # v = mean error per speaker
            utt = np.concatenate([e for _, e in by_seed])
            null = []
            for _ in range(N_PERM):
                perm = [(s, rng.permutation(e)) for s, e in by_seed]
                null.append(top_share(speaker_values(perm, n)))
            null = np.array(null)
            rows.append({"config": m, "dur": dur, "speakers": int(v.size), "utterances": int(utt.size),
                         "error_rate": float(utt.mean()),
                         "worst10pct_speakers_share": obs,
                         "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
                         "p_perm": float((np.sum(null >= obs) + 1) / (N_PERM + 1)),
                         "worst10pct_utterances_share": top_share(utt),
                         "utterances_with_any_error_share": float(np.mean(utt > 0))})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(data.RESULTS / "error_concentration_null.csv", index=False)


if __name__ == "__main__":
    main()
