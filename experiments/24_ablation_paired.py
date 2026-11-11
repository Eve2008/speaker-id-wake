"""Paired (same-speaker) differences for the trimming and fixed-word ablations (revision).

Trimming (1 s test, closed set N = 5 and EER): enrollment raw/trimmed x test
raw/trimmed, each condition minus raw+raw. Fixed word: a 1 s clip of the word
minus a 1 s clip of a random word, same speakers and enrollment (EER and
closed-set accuracy, N = 5). Households and enrollment are generated exactly as
in 03_evaluate.py. Bootstrap: one shared resampling matrix over the eval-speaker
population per configuration (1,000 replicates); for the EER the difference is
taken in each replicate.

Usage: python experiments/24_ablation_paired.py campplus_zhen_adv campplus_zhen_adv_cip0 eres2net_base_zh
Output: results/ablation_paired.csv
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swk import data, metrics, rules, splits  # noqa: E402

_spec = importlib.util.spec_from_file_location("ev", Path(__file__).resolve().parent / "03_evaluate.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)
EDGES = ev.EDGES["cos"]
NB = EDGES.size - 1


def acc_and_hist(q, owner, enr, m, hrng_seed):
    _, cos = rules.profile_score_matrix(q, enr)
    tg, to, nn, no = ev.eer_trials(cos, owner)
    ht = metrics.per_speaker_hist(tg, to, m, EDGES)
    hn = metrics.per_speaker_hist(nn, no, m, EDGES)
    hrng = np.random.default_rng(hrng_seed)
    comps = ev.households(hrng, m, 4, ev.R)
    cand = np.concatenate([np.broadcast_to(owner[:, None, None], (owner.size, ev.R, 1)), comps[owner]], -1)
    sc = ev.gather(cos, owner, cand)
    acc = ev.per_speaker(np.argmax(sc, axis=-1) == 0, owner, m)
    return acc, ht, hn


def run(model: str) -> list[dict]:
    st = ev.Store(model)
    pop = st.pop
    col = {int(k): i for i, k in enumerate(pop)}
    P = len(pop)
    W = metrics.boot_weights(P, np.random.default_rng(778))
    cells: dict = {}

    def put(key, seed_ids, acc, ht, hn):
        c = cells.setdefault(key, {"acc": np.full((len(splits.SEEDS), P), np.nan), "ht": np.zeros((P, NB)),
                                   "hn": np.zeros((P, NB))})
        idx = np.array([col[int(k)] for k in seed_ids])
        c["acc"][put.seed, idx] = acc
        c["ht"][idx] += ht
        c["hn"][idx] += hn

    for si, seed in enumerate(splits.SEEDS):
        put.seed = si
        sp = st.splits[seed]
        ev_ids = np.array(sorted(k for k, v in sp.items() if not v["dev"]))
        m = len(ev_ids)
        enr = st.enroll(seed, "3x4", ev_ids)
        # trimming
        if st.has(seed, "test1_trim") and st.has(seed, "enroll_3x4_trim"):
            tid = st.spec_ids(seed, "test1_trim", ev_ids, 6).astype(np.int64)
            eid = st.spec_ids(seed, "enroll_3x4_trim", ev_ids, 3).astype(np.int64)
            if not (np.isnan(st.emb[tid, 0]).any() or np.isnan(st.emb[eid, 0]).any()):
                e_tr = st.enroll(seed, "3x4", ev_ids, trim=True)
                for en_name, e_arr in (("raw", enr), ("trim", e_tr)):
                    for te_name, role in (("raw", "test1"), ("trim", "test1_trim")):
                        q, owner, *_ = st.tests(seed, role, ev_ids, 6)
                        a, ht, hn = acc_and_hist(q, owner, e_arr, m, [seed, 5, 0, 7])
                        put(("trim", f"enroll_{en_name}+test_{te_name}"), ev_ids, a, ht, hn)
        # fixed word vs random word
        for w in ev.FW_WORDS:
            role = f"fw_{w}"
            if not st.has(seed, role):
                continue
            ser = st.roles[(seed, role)]
            have = np.array([k for k in ev_ids if (k, 0) in ser.index])
            if have.size < 30:
                continue
            fids = ser.reindex(pd.MultiIndex.from_product([have, [0]])).to_numpy().astype(np.int64)
            if np.isnan(st.emb[fids, 0]).any():
                continue
            pos = np.searchsorted(ev_ids, have)
            e_w = enr[pos]
            mm = len(have)
            q = st._tf(np.asarray(st.emb[fids], dtype=np.float64))
            a, ht, hn = acc_and_hist(q, np.arange(mm), e_w, mm, [seed, 5, 21, ev.FW_WORDS.index(w)])
            put(("word", w, "fixed"), have, a, ht, hn)
            q, owner, *_ = st.tests(seed, "test1", have, 6)
            a, ht, hn = acc_and_hist(q, owner, e_w, mm, [seed, 5, 21, ev.FW_WORDS.index(w)])
            put(("word", w, "random"), have, a, ht, hn)
        print(f"[{model}] seed {seed}", flush=True)

    rows = []

    def compare(ka, kb, label):
        A, B = cells[ka], cells[kb]
        with np.errstate(invalid="ignore"):
            va, vb = np.nanmean(A["acc"], 0), np.nanmean(B["acc"], 0)
        d = va - vb
        est, lo, hi = metrics.boot_mean(d, W)
        ea = float(metrics.eer_from_hist(A["ht"].sum(0), A["hn"].sum(0))[0])
        eb = float(metrics.eer_from_hist(B["ht"].sum(0), B["hn"].sum(0))[0])
        bd = metrics.eer_from_hist(W @ A["ht"], W @ A["hn"])[0] - metrics.eer_from_hist(W @ B["ht"], W @ B["hn"])[0]
        elo, ehi = np.percentile(bd, [2.5, 97.5])
        rows.append({"config": model, "comparison": label, "speakers": int(np.sum(~np.isnan(d))),
                     "EER_A": ea, "EER_B": eb, "EER_diff": ea - eb, "EER_diff_lo": elo, "EER_diff_hi": ehi,
                     "acc_A": float(np.nanmean(va)), "acc_B": float(np.nanmean(vb)),
                     "acc_diff": est, "acc_diff_lo": lo, "acc_diff_hi": hi})

    base = ("trim", "enroll_raw+test_raw")
    if base in cells:
        for c in ("enroll_raw+test_trim", "enroll_trim+test_raw", "enroll_trim+test_trim"):
            compare(("trim", c), base, f"trim: {c} minus raw+raw")
    for w in ev.FW_WORDS:
        if ("word", w, "fixed") in cells:
            compare(("word", w, "fixed"), ("word", w, "random"), f"word {w}: fixed minus random")
    return rows


def main() -> None:
    rows = []
    for m in sys.argv[1:]:
        rows += run(m)
    out = data.RESULTS / "ablation_paired.csv"
    df = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out)
        df = pd.concat([old[~old.config.isin(sys.argv[1:])], df], ignore_index=True)
    df.to_csv(out, index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
