"""Two-way (crossed) bootstrap of EER and EER differences (revision).

The speaker-bootstrap intervals of 03_evaluate.py and 19_paired.py resample test
speakers only. A verification trial pairs a test speaker i with an enrolled
speaker j, so the trial set is crossed. Here each replicate draws two independent
multinomial weight vectors over the eval-speaker population, w for the test side
and v for the enrollment side (the "pigeonhole" bootstrap for crossed designs,
Owen 2007), and gives trial (i, j) the weight w_i * v_j (a target trial gets
w_i * v_i). The same speaker keeps its weights in every seed in which it is an eval
speaker. Paired differences use the same (w, v) for both configurations. One-way
intervals (v = 1) are computed from the same w for comparison.

Second version (11 Nov, after the reboot): durations 1, 2 and 3 s; all main, matched,
inference-path and back-end configurations; the one-way replicates come from a matrix
product over per-test-speaker histograms and the two-way non-target histograms from one
weighted bincount per replicate over the full score matrix (minus the target cells),
which is faster than the first version (kept as cache/22_twoway_bootstrap_part1.py; its
cache files *_B500.npz covered 1 s and 3 s for three configurations and are not used).

Usage:
  python experiments/22_twoway_bootstrap.py --only <config> ...   # compute and cache (one file per config)
  python experiments/22_twoway_bootstrap.py                         # assemble from the cache
Output: cache/twoway/<config>_B500_d123.npz; results/twoway_bootstrap.csv, results/twoway_bootstrap.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swk import data, metrics, rules, splits  # noqa: E402

_spec = importlib.util.spec_from_file_location("ev", Path(__file__).resolve().parent / "03_evaluate.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

MAIN = ["campplus_zhen_adv", "eres2net_base_zh", "titanet_small", "wespeaker_campplus_vox_wsfe",
        "wespeaker_resnet34_vox_wsfe", "wespeaker_resnet34_cnc_wsfe"]
MATCHED_NEW = ["campplus_zhen_adv_cip0_d1", "eres2net_base_zh_d1", "titanet_small_d1"]
PATHS = ["campplus_zhen_adv_d1", "campplus_zhen_adv_cip0", "campplus_zhen_adv_np128", "campplus_zhen_adv_ort131"]
CW = [f"{m}__cw" for m in MAIN]
CONFIGS = MAIN + MATCHED_NEW + PATHS + CW
MATCHED = ["campplus_zhen_adv_cip0_d1", "eres2net_base_zh_d1", "titanet_small_d1", "wespeaker_campplus_vox_wsfe",
           "wespeaker_resnet34_vox_wsfe", "wespeaker_resnet34_cnc_wsfe"]
PAIRS = ([(m, "campplus_zhen_adv") for m in MAIN[1:]]
         + [(m, "campplus_zhen_adv_cip0_d1") for m in MATCHED[1:]]
         + [("eres2net_base_zh_d1", "wespeaker_resnet34_cnc_wsfe"), ("eres2net_base_zh", "wespeaker_resnet34_cnc_wsfe"),
            ("eres2net_base_zh_d1", "eres2net_base_zh"), ("titanet_small_d1", "titanet_small")]
         + [(m, "campplus_zhen_adv") for m in PATHS + ["campplus_zhen_adv_cip0_d1"]]
         + [("campplus_zhen_adv_ort131", "campplus_zhen_adv_np128"), ("campplus_zhen_adv_ort131", "campplus_zhen_adv_cip0"),
            ("campplus_zhen_adv_cip0_d1", "campplus_zhen_adv_cip0")]
         + [(f"{m}__cw", m) for m in MAIN])
DURS = (1, 2, 3)
EDGES = ev.EDGES["cos"]
N_BOOT = 500
SEED = 2024


def weights(n_pop: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)  # same draws as the first version
    W = metrics.boot_weights(n_pop, rng, N_BOOT).astype(np.float64)
    V = metrics.boot_weights(n_pop, rng, N_BOOT).astype(np.float64)
    return W, V


def histograms(cfg: str, pop_index: dict, W: np.ndarray, V: np.ndarray) -> dict:
    st = ev.Store(cfg)
    n_pop = W.shape[1]
    nb = EDGES.size - 1
    out = {d: {"n2": np.zeros((N_BOOT, nb)), "Ht": np.zeros((n_pop, nb)), "Hn": np.zeros((n_pop, nb))} for d in DURS}
    t0 = time.time()
    for seed in splits.SEEDS:
        sp = st.splits[seed]
        ids_all = np.array(sorted(sp))
        dev_ids = np.array([k for k in ids_all if sp[k]["dev"]])
        ev_ids = np.array(sorted(k for k, v in sp.items() if not v["dev"]))
        st.fit_backend(seed, dev_ids)  # no-op without a back end
        col = np.array([pop_index[k] for k in ev_ids])
        enr = st.enroll(seed, "3x4", ev_ids)
        for dur in DURS:
            q, owner, *_ = st.tests(seed, f"test{dur}", ev_ids, len(splits.TEST_GROUPS[dur]))
            assert not (np.isnan(q).any() or np.isnan(enr).any()), f"{cfg}: embeddings incomplete"
            _, cos = rules.profile_score_matrix(q, enr)
            bins = np.clip(np.searchsorted(EDGES, cos, side="right") - 1, 0, nb - 1)
            u = np.arange(cos.shape[0])
            tb = bins[u, owner]
            o = out[dur]
            row_pop = col[owner]  # population index of each test input's speaker
            # per-test-speaker histograms (one-way replicates and point estimate)
            np.add.at(o["Ht"], (row_pop, tb), 1.0)
            flat = bins.ravel()
            np.add.at(o["Hn"], (np.repeat(row_pop, cos.shape[1]), flat), 1.0)
            np.add.at(o["Hn"], (row_pop, tb), -1.0)  # remove the target cells
            for b in range(N_BOOT):
                wt = np.outer(W[b, row_pop], V[b, col]).ravel()
                h = np.bincount(flat, weights=wt, minlength=nb)
                h -= np.bincount(tb, weights=W[b, row_pop] * V[b, row_pop], minlength=nb)
                o["n2"][b] += h
        print(f"[{cfg}] seed {seed}, {(time.time() - t0) / 60:.1f} min", flush=True)
    res = {}
    for d, o in out.items():
        t2 = (W * V) @ o["Ht"]
        t1 = W @ o["Ht"]
        n1 = W @ o["Hn"]
        res[d] = {"eer": float(metrics.eer_from_hist(o["Ht"].sum(0), o["Hn"].sum(0))[0]),
                  "boot2": metrics.eer_from_hist(t2, o["n2"])[0], "boot1": metrics.eer_from_hist(t1, n1)[0]}
    return res


def cache_file(cfg: str) -> Path:
    return data.CACHE / "twoway" / f"{cfg}_B{N_BOOT}_d123.npz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="compute (and cache) only these configurations")
    a = ap.parse_args()
    st = ev.Store(CONFIGS[0])
    pop = st.pop
    pop_index = {int(k): i for i, k in enumerate(pop)}
    W, V = weights(len(pop))
    (data.CACHE / "twoway").mkdir(exist_ok=True)
    if a.only:
        for m in a.only:
            f = cache_file(m)
            if f.exists():
                print("cached already:", m)
                continue
            t0 = time.time()
            r = histograms(m, pop_index, W, V)
            tmp = f.with_suffix(".tmp.npz")
            np.savez(tmp, minutes=(time.time() - t0) / 60,
                     **{k: v for d in DURS for k, v in ((f"eer{d}", r[d]["eer"]), (f"b2_{d}", r[d]["boot2"]),
                                                       (f"b1_{d}", r[d]["boot1"]))})
            tmp.rename(f)
            print(f"[{m}] cached in {(time.time() - t0) / 60:.1f} min", flush=True)
        return
    res = {}
    for m in CONFIGS:
        f = cache_file(m)
        if f.exists():
            z = np.load(f)
            res[m] = {d: {"eer": float(z[f"eer{d}"]), "boot2": z[f"b2_{d}"], "boot1": z[f"b1_{d}"]} for d in DURS}
    rows = []
    for m in res:
        for d in DURS:
            r = res[m][d]
            for kind in ("boot1", "boot2"):
                lo, hi = np.percentile(r[kind], [2.5, 97.5])
                rows.append({"what": "EER", "A": m, "B": "", "dur": d, "estimate": r["eer"],
                             "bootstrap": "test speakers only" if kind == "boot1" else "test and enrolled speakers",
                             "lo": lo, "hi": hi, "width": hi - lo})
    for x, y in PAIRS:
        if x not in res or y not in res:
            continue
        for d in DURS:
            for kind in ("boot1", "boot2"):
                dd = res[x][d][kind] - res[y][d][kind]
                lo, hi = np.percentile(dd, [2.5, 97.5])
                rows.append({"what": "EER difference A-B", "A": x, "B": y, "dur": d,
                             "estimate": res[x][d]["eer"] - res[y][d]["eer"],
                             "bootstrap": "test speakers only" if kind == "boot1" else "test and enrolled speakers",
                             "lo": lo, "hi": hi, "width": hi - lo})
    df = pd.DataFrame(rows)
    df.to_csv(data.RESULTS / "twoway_bootstrap.csv", index=False)
    w = df.pivot_table(index=["what", "A", "B", "dur"], columns="bootstrap", values="width").reset_index()
    w["ratio"] = w["test and enrolled speakers"] / w["test speakers only"]
    sig = df[df.what != "EER"].assign(excl=lambda t: (t.lo > 0) | (t.hi < 0))
    flips = sig.pivot_table(index=["A", "B", "dur"], columns="bootstrap", values="excl").reset_index()
    flips = flips[flips["test speakers only"] != flips["test and enrolled speakers"]]
    summ = {"n_boot": N_BOOT, "population": int(len(pop)), "configs": list(res),
            "width_ratio_EER_min": float(w[w.what == "EER"].ratio.min()),
            "width_ratio_EER_max": float(w[w.what == "EER"].ratio.max()),
            "width_ratio_diff_min": float(w[w.what != "EER"].ratio.min()),
            "width_ratio_diff_max": float(w[w.what != "EER"].ratio.max()),
            "width_ratio_median": float(w.ratio.median()),
            "differences_whose_ci_excludes_0_only_one_way": [
                f"{r['A']} minus {r['B']}, {r['dur']} s" for _, r in flips.iterrows() if bool(r["test speakers only"])],
            "differences_whose_ci_excludes_0_only_two_way": [
                f"{r['A']} minus {r['B']}, {r['dur']} s" for _, r in flips.iterrows()
                if bool(r["test and enrolled speakers"])]}
    with open(data.RESULTS / "twoway_bootstrap.json", "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
