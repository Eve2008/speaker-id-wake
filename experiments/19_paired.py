"""Paired (same-speaker) comparisons between configurations and between scoring rules.

Stage 1 (``--dump <model> ...``): for one configuration, re-run the N = 5
open-set evaluation of 03_evaluate.py (same splits, same households, same
dev-tuned thresholds; the script checks that its estimates reproduce
results/<model>/main.csv and eer.csv) and save per-speaker values plus the
bootstrap replicates of the EER, all under ONE shared speaker-resampling
matrix W (fixed seed). Output: cache/paired/<model>.npz (one file per model,
so an interruption loses at most one model).

Stage 2 (``--compare``): differences between two configurations (or two
rules of one configuration) on the same speakers. For a per-speaker mean the
difference is bootstrapped from the per-speaker differences with W; for the
EER the difference of the two EERs is taken in each bootstrap replicate.
Output: results/tables/paired_differences.csv (+ .md).

Usage:
  python experiments/19_paired.py --dump campplus_zhen_adv eres2net_base_zh ...
  python experiments/19_paired.py --compare

Revision: configurations may carry a back end ("<model>__cw"); the shared
resampling matrix is drawn over the eval-speaker population (speakers eval in at
least one seed, 1,362), not over all index slots; the margin rules of
03_evaluate.py (cosm, blendm, asnormm) and member_wrong are dumped as well.
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

OUT = data.CACHE / "paired"
TAB = data.RESULTS / "tables"
N_H = 5
W_SEED = 777
SCORES = ("cos", "asnorm")


def shared_weights(n_spk: int) -> np.ndarray:
    return metrics.boot_weights(n_spk, np.random.default_rng(W_SEED))


TUNED_FAMS = ("cos", "nexaroom", "asnorm", "cosm", "blendm", "asnormm")


def dump(model: str) -> None:
    t0 = time.time()
    st = ev.Store(model)
    n = st.n_spk
    pop = st.pop
    vals: dict[str, np.ndarray] = {}
    hist: dict[tuple, list] = {}

    def put(key: str, seed: int, ids: np.ndarray, v: np.ndarray) -> None:
        if key not in vals:
            vals[key] = np.full((len(splits.SEEDS), n), np.nan)
        vals[key][splits.SEEDS.index(seed), ids] = v

    for seed in splits.SEEDS:
        sp = st.splits[seed]
        ids_all = np.array(sorted(sp))
        dev_ids = np.array([k for k in ids_all if sp[k]["dev"]])
        ev_ids = np.array([k for k in ids_all if not sp[k]["dev"]])
        sets = {"eval": ev_ids, "dev": dev_ids}
        st.fit_backend(seed, dev_ids)
        enr = {name: st.enroll(seed, "3x4", ids) for name, ids in sets.items()}
        cen = {name: rules.centroids(e) for name, e in enr.items()}
        loo = {name: rules.loo_floor(e) for name, e in enr.items()}
        crng = np.random.default_rng([seed, 99])
        cohort = {"eval": cen["dev"],
                  "dev": cen["eval"][np.sort(crng.choice(len(ev_ids), len(dev_ids), replace=False))]}
        hh = {}
        for name, ids in sets.items():  # same generator calls as 03_evaluate.py for N = 5
            hrng = np.random.default_rng([seed, N_H, 0 if name == "eval" else 1, 7])
            hh[name] = (ev.households(hrng, len(ids), N_H - 1, ev.R), ev.households(hrng, len(ids), N_H, ev.R))
        for dur in ev.DURS:
            n_per = len(splits.TEST_GROUPS[dur])
            cell = {}
            for name, ids in sets.items():
                q, owner, sec, rms, clp = st.tests(seed, f"test{dur}", ids, n_per)
                blend, cos = rules.profile_score_matrix(q, enr[name])
                ok = rules.audio_ok(sec, rms, clp)
                pen = rules.short_penalty(sec)
                asn = rules.asnorm(cos, q, cen[name], cohort[name])
                if name == "eval":
                    for stype, mat in (("cos", cos), ("asnorm", asn)):
                        tg, to, nn, no = ev.eer_trials(mat, owner)
                        edges = ev.EDGES[stype]
                        key = (dur, stype)
                        if key not in hist:
                            hist[key] = [np.zeros((n, edges.size - 1)), np.zeros((n, edges.size - 1))]
                        hist[key][0][ids] += metrics.per_speaker_hist(tg, to, len(ids), edges)
                        hist[key][1][ids] += metrics.per_speaker_hist(nn, no, len(ids), edges)
                comps, strangers = hh[name]
                cand_m = np.concatenate([np.broadcast_to(owner[:, None, None], (owner.size, ev.R, 1)),
                                         comps[owner]], axis=-1)
                tm = ev.TrialSet(cos, blend, asn, cand_m, loo[name], ok, pen, owner, True)
                ts = ev.TrialSet(cos, blend, asn, strangers[owner], loo[name], ok, pen, owner, False)
                cell[name] = (tm, ts, owner, len(ids))
            tm_e, ts_e, ow_e, m_e = cell["eval"]
            tm_d, ts_d, ow_d, m_d = cell["dev"]
            ps = ev.per_speaker
            for sname, arr in (("cos", tm_e.cos_t), ("blend", tm_e.blend_t), ("asnorm", tm_e.asn_t)):
                put(f"d{dur}|closed_set|acc_{sname}", seed, ev_ids, ps(arr, ow_e, m_e))
            for rname, param, label in (("nexaroom", None, "nexaroom_default"), ("cos", 0.5, "cos@0.50")):
                rm, rs = tm_e.rule(rname, param), ts_e.rule(rname, param)
                put(f"d{dur}|{label}|member_correct", seed, ev_ids, ps(rm["correct"], ow_e, m_e))
                put(f"d{dur}|{label}|member_wrong", seed, ev_ids, ps(rm["wrong"], ow_e, m_e))
                put(f"d{dur}|{label}|stranger_fa", seed, ev_ids, ps(rs["named"], ow_e, m_e))
            fa_dev_nx = ps(ts_d.rule("nexaroom")["named"], ow_d, m_d).mean()
            todo = [(t, f"fa{int(t * 100)}", fam) for t in ev.FA_TARGETS for fam in TUNED_FAMS]
            todo += [(fa_dev_nx, "matched", fam) for fam in ("cos", "cosm")]
            for target, tname, fam in todo:
                if True:
                    a, e = ts_d.accept_score(fam)
                    th = ev.tune(a, e, ow_d, m_d, target)
                    label = f"{fam}@dev_{tname}"
                    rm, rs = tm_e.rule(fam, th), ts_e.rule(fam, th)
                    put(f"d{dur}|{label}|member_correct", seed, ev_ids, ps(rm["correct"], ow_e, m_e))
                    put(f"d{dur}|{label}|member_wrong", seed, ev_ids, ps(rm["wrong"], ow_e, m_e))
                    put(f"d{dur}|{label}|stranger_fa", seed, ev_ids, ps(rs["named"], ow_e, m_e))
        print(f"[{model}] seed {seed} done, {(time.time() - t0) / 60:.1f} min", flush=True)

    # N = 20 closed set from the per-speaker file written by 03_evaluate.py
    p = pd.read_csv(data.RESULTS / st.config / "per_speaker_main.csv.gz")
    p = p[(p.rule == "closed_set") & (p.metric == "acc_cos") & (p.N == 20)]
    for (dur, seed), g in p.groupby(["dur", "seed"]):
        put(f"d{dur}|closed_set_N20|acc_cos", int(seed), g.spk.to_numpy(), g.value.to_numpy())

    W = shared_weights(len(pop))
    per_spk = {k: np.nanmean(v, axis=0)[pop] for k, v in vals.items()}
    out = {f"spk|{k}": v for k, v in per_spk.items()}
    out["pop"] = pop
    for (dur, stype), (ht, hn) in hist.items():
        ht, hn = ht[pop], hn[pop]
        e, dcf = metrics.eer_from_hist(ht.sum(0), hn.sum(0))
        be, bd = metrics.eer_from_hist(W @ ht, W @ hn)
        out[f"eer|d{dur}|{stype}"] = np.array([e])
        out[f"eerboot|d{dur}|{stype}"] = be
        out[f"dcf|d{dur}|{stype}"] = np.array([dcf])
        out[f"dcfboot|d{dur}|{stype}"] = bd
    # reproduction check against the files written by 03_evaluate.py
    mm = pd.read_csv(data.RESULTS / st.config / "main.csv")
    ee = pd.read_csv(data.RESULTS / st.config / "eer.csv")
    diffs = []
    for k, v in per_spk.items():
        d, rule, metric = k.split("|")
        nn = 20 if rule == "closed_set_N20" else N_H
        rule = "closed_set" if rule == "closed_set_N20" else rule
        r = mm[(mm.dur == int(d[1:])) & (mm.N == nn) & (mm.rule == rule) & (mm.metric == metric)]
        if len(r):
            diffs.append(abs(float(np.nanmean(v)) - float(r.est.iloc[0])))
    eer_diffs = []
    for dur in ev.DURS:
        for stype in SCORES:
            r = ee[(ee.dur == dur) & (ee.score == stype)]
            eer_diffs.append(abs(float(out[f"eer|d{dur}|{stype}"][0]) - float(r.eer.iloc[0])))
    check = {"model": model, "n_metric_keys_checked": len(diffs), "max_abs_diff_vs_main_csv": max(diffs),
             "max_abs_diff_eer_vs_eer_csv": max(eer_diffs), "minutes": (time.time() - t0) / 60}
    out["check_json"] = np.array([json.dumps(check)])
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"{model}.tmp.npz"
    np.savez_compressed(tmp, **out)
    tmp.rename(OUT / f"{model}.npz")
    print(json.dumps(check), flush=True)


# ---------------------------------------------------------------- stage 2
BASE = "campplus_zhen_adv"
WORKING = ["eres2net_base_zh", "titanet_small", "wespeaker_campplus_vox_wsfe", "wespeaker_resnet34_vox_wsfe",
           "wespeaker_resnet34_cnc_wsfe"]
VARIANTS = ["campplus_zhen_adv_d1", "campplus_zhen_adv_ort131", "campplus_zhen_adv_np128", "campplus_zhen_adv_cip0",
            "campplus_zhen_adv_cip0_d1"]
MAIN6 = [BASE] + WORKING
MATCHED = ["campplus_zhen_adv_cip0_d1", "eres2net_base_zh_d1", "titanet_small_d1", "wespeaker_campplus_vox_wsfe",
           "wespeaker_resnet34_vox_wsfe", "wespeaker_resnet34_cnc_wsfe", "campplus_vox_cip0_d1", "eres2net_vox_d1"]
CW = [f"{m}__cw" for m in MAIN6]
CONFIGS = list(dict.fromkeys([BASE] + VARIANTS + WORKING + MATCHED + CW))
METRICS_X = [  # (key suffix, label) compared between configurations
    ("closed_set|acc_cos", "closed-set acc. N=5 (cosine)"),
    ("closed_set_N20|acc_cos", "closed-set acc. N=20 (cosine)"),
    ("nexaroom_default|member_correct", "NexaRoom rule: member named correctly"),
    ("nexaroom_default|member_wrong", "NexaRoom rule: wrong member named"),
    ("nexaroom_default|stranger_fa", "NexaRoom rule: stranger accepted"),
    ("cos@dev_fa1|member_correct", "cosine @ dev 1% FA: member named correctly"),
    ("asnorm@dev_fa1|member_correct", "AS-norm @ dev 1% FA: member named correctly"),
    ("cos@dev_fa1|member_wrong", "cosine @ dev 1% FA: wrong member named"),
    ("asnorm@dev_fa1|member_wrong", "AS-norm @ dev 1% FA: wrong member named"),
    ("cos@dev_fa1|stranger_fa", "cosine @ dev 1% FA: stranger accepted"),
    ("asnorm@dev_fa1|stranger_fa", "AS-norm @ dev 1% FA: stranger accepted"),
    ("asnormm@dev_fa1|member_correct", "AS-norm + margin @ dev 1% FA: member named correctly"),
]
RULE_PAIRS = [  # within one configuration: (A, B, label); difference = A - B
    ("asnorm@dev_fa1|member_correct", "cos@dev_fa1|member_correct", "AS-norm minus cosine, member correct @ dev 1% FA"),
    ("nexaroom@dev_fa1|member_correct", "cos@dev_fa1|member_correct",
     "NexaRoom rule (tuned base) minus cosine, member correct @ dev 1% FA"),
    ("asnorm@dev_fa1|stranger_fa", "cos@dev_fa1|stranger_fa", "AS-norm minus cosine, stranger FA @ dev 1% FA"),
    ("nexaroom@dev_fa1|stranger_fa", "cos@dev_fa1|stranger_fa",
     "NexaRoom rule (tuned base) minus cosine, stranger FA @ dev 1% FA"),
    ("asnorm@dev_fa5|member_correct", "cos@dev_fa5|member_correct", "AS-norm minus cosine, member correct @ dev 5% FA"),
    ("nexaroom@dev_fa5|member_correct", "cos@dev_fa5|member_correct",
     "NexaRoom rule (tuned base) minus cosine, member correct @ dev 5% FA"),
    ("nexaroom_default|member_correct", "cos@0.50|member_correct", "NexaRoom default minus cosine >= 0.50, member correct"),
    ("nexaroom_default|stranger_fa", "cos@0.50|stranger_fa", "NexaRoom default minus cosine >= 0.50, stranger FA"),
    ("closed_set|acc_blend", "closed_set|acc_cos", "blend score minus cosine, closed-set acc. N=5"),
    ("closed_set|acc_asnorm", "closed_set|acc_cos", "AS-norm minus cosine, closed-set acc. N=5"),
    # revision: isolate the elements of the tuned rule (all at dev-tuned 1% stranger acceptance)
    ("nexaroom@dev_fa1|member_wrong", "cos@dev_fa1|member_wrong",
     "NexaRoom rule (tuned base) minus cosine, wrong member @ dev 1% FA"),
    ("cosm@dev_fa1|member_correct", "cos@dev_fa1|member_correct", "cosine+margin minus cosine, member correct @ dev 1% FA"),
    ("cosm@dev_fa1|member_wrong", "cos@dev_fa1|member_wrong", "cosine+margin minus cosine, wrong member @ dev 1% FA"),
    ("blendm@dev_fa1|member_correct", "cosm@dev_fa1|member_correct",
     "blend+margin minus cosine+margin, member correct @ dev 1% FA"),
    ("blendm@dev_fa1|member_wrong", "cosm@dev_fa1|member_wrong",
     "blend+margin minus cosine+margin, wrong member @ dev 1% FA"),
    ("nexaroom@dev_fa1|member_correct", "blendm@dev_fa1|member_correct",
     "NexaRoom rule (tuned base) minus blend+margin, member correct @ dev 1% FA"),
    ("nexaroom@dev_fa1|member_correct", "cosm@dev_fa1|member_correct",
     "NexaRoom rule (tuned base) minus cosine+margin, member correct @ dev 1% FA"),
    ("nexaroom@dev_fa1|member_wrong", "cosm@dev_fa1|member_wrong",
     "NexaRoom rule (tuned base) minus cosine+margin, wrong member @ dev 1% FA"),
    ("asnormm@dev_fa1|member_correct", "asnorm@dev_fa1|member_correct",
     "AS-norm+margin minus AS-norm, member correct @ dev 1% FA"),
    ("asnormm@dev_fa1|member_wrong", "asnorm@dev_fa1|member_wrong",
     "AS-norm+margin minus AS-norm, wrong member @ dev 1% FA"),
    ("asnormm@dev_fa1|member_correct", "cosm@dev_fa1|member_correct",
     "AS-norm+margin minus cosine+margin, member correct @ dev 1% FA"),
    ("asnormm@dev_fa1|member_correct", "nexaroom@dev_fa1|member_correct",
     "AS-norm+margin minus NexaRoom rule (tuned base), member correct @ dev 1% FA"),
    ("asnormm@dev_fa1|member_wrong", "nexaroom@dev_fa1|member_wrong",
     "AS-norm+margin minus NexaRoom rule (tuned base), wrong member @ dev 1% FA"),
    ("cosm@dev_matched|member_correct", "nexaroom_default|member_correct",
     "cosine+margin at the rule's dev stranger rate minus NexaRoom default, member correct"),
    ("cos@dev_matched|member_correct", "nexaroom_default|member_correct",
     "cosine at the rule's dev stranger rate minus NexaRoom default, member correct"),
]


def boot_diff(v: np.ndarray, W: np.ndarray) -> tuple[float, float, float, float]:
    """Mean of per-speaker differences, 95% CI, and two-sided bootstrap p (share of replicates across 0)."""

    est, lo, hi = metrics.boot_mean(v, W)
    ok = ~np.isnan(v)
    x = np.where(ok, v, 0.0)
    bs = (W @ x) / np.maximum(W @ ok.astype(np.float32), 1e-12)
    p = 2 * min(np.mean(bs <= 0), np.mean(bs >= 0))
    return est, lo, hi, float(min(p, 1.0))


def compare() -> None:
    have = [m for m in CONFIGS if (OUT / f"{m}.npz").exists()]
    z = {m: dict(np.load(OUT / f"{m}.npz")) for m in have}
    stale = [m for m in have if "pop" not in z[m]]  # dumps made before the revision (other population)
    if stale:
        print("skipping pre-revision dumps:", stale)
    have = [m for m in have if m not in stale]
    z = {m: z[m] for m in have}
    n = len(next(iter(z.values()))[f"spk|d1|closed_set|acc_cos"])
    pops = [tuple(z[m]["pop"]) for m in have]
    assert all(p == pops[0] for p in pops), "configurations differ in their eval-speaker population"
    W = shared_weights(n)
    rows = []

    def add(kind, a, b, metric, dur, est, lo, hi, p, n_spk, va=None, vb=None):
        rows.append({"comparison": kind, "A": a, "B": b, "metric": metric, "dur": dur, "A_value": va,
                     "B_value": vb, "diff_A_minus_B": est, "lo": lo, "hi": hi, "p_boot": p, "n_spk": n_spk})

    pairs = [(m, BASE) for m in VARIANTS + WORKING]
    # revision: decomposition of the inference-path effect for NexaRoom's weights
    pairs += [("campplus_zhen_adv_ort131", "campplus_zhen_adv_np128"),  # runtime only (same NumPy features)
              ("campplus_zhen_adv_ort131", "campplus_zhen_adv_cip0"),  # front end only (same pooling semantics)
              ("campplus_zhen_adv_cip0_d1", "campplus_zhen_adv_cip0")]  # noise floor on the patched graph
    # revision: matched condition (1-LSB noise floor, runtime path agreeing with onnxruntime 1.31.1)
    pairs += [(m, "campplus_zhen_adv_cip0_d1") for m in MATCHED[1:]]
    pairs += [("eres2net_base_zh_d1", "wespeaker_resnet34_cnc_wsfe"), ("eres2net_base_zh_d1", "eres2net_base_zh"),
              ("titanet_small_d1", "titanet_small"),
              # training data within one toolkit and architecture, and toolkit within one architecture + data
              ("campplus_vox_cip0_d1", "wespeaker_campplus_vox_wsfe"), ("eres2net_vox_d1", "eres2net_base_zh_d1"),
              ("eres2net_vox_d1", "campplus_vox_cip0_d1")]
    # revision: dev-trained centring + whitening back end against raw cosine, per model
    pairs += [(f"{m}__cw", m) for m in MAIN6]
    pairs += [("eres2net_base_zh", "wespeaker_resnet34_cnc_wsfe"), ("eres2net_base_zh", "campplus_zhen_adv_ort131"),
              ("wespeaker_resnet34_cnc_wsfe", "wespeaker_resnet34_vox_wsfe"),
              ("wespeaker_resnet34_vox_wsfe", "wespeaker_campplus_vox_wsfe"),
              ("titanet_small", "wespeaker_campplus_vox_wsfe")]
    for a, b in pairs:
        if a not in z or b not in z:
            continue
        for dur in ev.DURS:
            for stype in SCORES:
                da = z[a][f"eerboot|d{dur}|{stype}"] - z[b][f"eerboot|d{dur}|{stype}"]
                ea, eb = float(z[a][f"eer|d{dur}|{stype}"][0]), float(z[b][f"eer|d{dur}|{stype}"][0])
                lo, hi = np.percentile(da, [2.5, 97.5])
                p = 2 * min(np.mean(da <= 0), np.mean(da >= 0))
                add("config", a, b, f"EER ({stype})", dur, ea - eb, lo, hi, min(p, 1.0), n, ea, eb)
            for suf, lab in METRICS_X:
                if f"spk|d{dur}|{suf}" not in z[a] or f"spk|d{dur}|{suf}" not in z[b]:
                    continue
                va, vb = z[a][f"spk|d{dur}|{suf}"], z[b][f"spk|d{dur}|{suf}"]
                d = va - vb
                est, lo, hi, p = boot_diff(d, W)
                add("config", a, b, lab, dur, est, lo, hi, p, int(np.sum(~np.isnan(d))), float(np.nanmean(va)),
                    float(np.nanmean(vb)))
    for m in have:
        for dur in ev.DURS:
            da = z[m][f"eerboot|d{dur}|asnorm"] - z[m][f"eerboot|d{dur}|cos"]
            ea, eb = float(z[m][f"eer|d{dur}|asnorm"][0]), float(z[m][f"eer|d{dur}|cos"][0])
            lo, hi = np.percentile(da, [2.5, 97.5])
            p = 2 * min(np.mean(da <= 0), np.mean(da >= 0))
            add("rule", m, m, "AS-norm minus cosine, EER", dur, ea - eb, lo, hi, min(p, 1.0), n, ea, eb)
            for ka, kb, lab in RULE_PAIRS:
                if f"spk|d{dur}|{ka}" not in z[m] or f"spk|d{dur}|{kb}" not in z[m]:
                    continue
                va, vb = z[m][f"spk|d{dur}|{ka}"], z[m][f"spk|d{dur}|{kb}"]
                est, lo, hi, p = boot_diff(va - vb, W)
                add("rule", m, m, lab, dur, est, lo, hi, p, int(np.sum(~np.isnan(va - vb))), float(np.nanmean(va)),
                    float(np.nanmean(vb)))
    df = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / "paired_differences.csv", index=False)
    checks = [json.loads(str(z[m]["check_json"][0])) for m in have]
    with open(data.RESULTS / "paired_check.json", "w") as f:
        json.dump({"shared_weight_seed": W_SEED, "n_boot": int(W.shape[0]), "models": checks}, f, indent=1)
    # markdown (percentage points)
    md = df.copy()
    for c in ("A_value", "B_value", "diff_A_minus_B", "lo", "hi"):
        md[c] = (100 * md[c]).round(2)
    md["diff (pp) [95% CI]"] = md.apply(lambda r: f"{r.diff_A_minus_B:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]", axis=1)
    md["p_boot"] = md.p_boot.map(lambda p: "<0.001" if p < 0.001 else f"{p:.3f}")
    cols = ["comparison", "A", "B", "metric", "dur", "A_value", "B_value", "diff (pp) [95% CI]", "p_boot"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in md.iterrows()]
    (TAB / "paired_differences.md").write_text(
        "Paired speaker-level bootstrap (1,000 replicates, one shared resampling matrix). Values in %, "
        "differences in percentage points, A minus B. p_boot = two-sided bootstrap share of replicates on "
        "the other side of 0 (resolution 0.002).\n\n" + "\n".join(lines) + "\n")
    print(f"wrote {len(df)} rows for {len(have)} configurations: {have}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", nargs="*", default=[])
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for mdl in a.dump:
        if (OUT / f"{mdl}.npz").exists() and not a.force:
            print(f"[{mdl}] exists, skipping")
            continue
        dump(mdl)
    if a.compare:
        compare()
