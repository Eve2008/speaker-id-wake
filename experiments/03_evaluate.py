"""Evaluate one speaker-embedding model on all benchmark conditions.

Usage: python experiments/03_evaluate.py --model campplus_zhen_adv [--ablations] [--backend cw]

Revision additions: (1) dev-tuned rules with NexaRoom's 0.06 runner-up margin for
plain cosine (cosm), the blend score without calibration or short-audio penalty
(blendm) and AS-norm (asnormm; margin on the cosine of the two highest-ranked
members); (2) the bootstrap resamples the eval speakers (those eval in at least
one seed), not all speaker index slots; (3) --backend cw: embeddings are centred
on the dev-speaker mean and whitened with the dev-speaker covariance of the same
seed before scoring (results go to results/<model>__cw/).

For every seed the speakers are split into dev (1/3) and eval (2/3). All
reported numbers are on eval speakers; dev speakers are used only to tune
thresholds and as the AS-norm cohort. Households are simulated per speaker:
for each test utterance of speaker s, R random households of N enrolled
speakers containing s (member trials) and R random households of N enrolled
speakers not containing s (stranger trials). Values are averaged per speaker
first; speakers are the resampling unit of the bootstrap.

Outputs go to results/<model>/ (see README).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swk import data, metrics, rules, splits  # noqa: E402

NS = (2, 3, 5, 10, 20)
DURS = (1, 2, 3)
R = 100
FA_TARGETS = (0.01, 0.05)
POOLED_NS = (5,)
COS_GRID = np.round(np.arange(0.0, 1.0001, 0.01), 3)
TAU_GRID = np.round(np.arange(0.20, 0.9001, 0.01), 3)
ASN_GRID = np.round(np.arange(-3.0, 15.0001, 0.05), 3)
FW_WORDS = ["این", "برای", "خیلی", "باید", "اینجا"]


# ---------------------------------------------------------------- data access
class Store:
    def __init__(self, model: str, backend: str = "") -> None:
        if "__" in model:  # config key "<model>__<backend>"
            model, backend = model.split("__", 1)
        self.model = model
        self.backend = backend
        self.config = model + (f"__{backend}" if backend else "")
        self.transform = None
        self.emb = np.load(data.CACHE / "emb" / f"{model}.npy", mmap_mode="r")
        self.audio = np.load(data.CACHE / "spec_audio.npy", mmap_mode="r")
        roles = pd.read_csv(data.CACHE / "roles.csv")
        self.roles = {k: g.set_index(["spk", "index"]).spec_id for k, g in roles.groupby(["seed", "role"])}
        self.splits = {}
        for s in splits.SEEDS:
            with open(data.RESULTS / "splits" / f"seed{s}.json") as f:
                self.splits[s] = {int(k): v for k, v in json.load(f).items()}
        spk = pd.read_csv(data.RESULTS / "speakers.csv")
        self.gender = dict(zip(spk.spk, spk.gender))
        self.n_spk = int(spk.spk.max()) + 1
        # bootstrap population: speakers that are eval in at least one seed
        self.pop = np.array(sorted({k for s in splits.SEEDS for k, v in self.splits[s].items() if not v["dev"]}))

    def fit_backend(self, seed: int, dev_ids: np.ndarray) -> None:
        """Dev-trained back end for this seed (revision): mean centring + whitening (cw)."""

        self.transform = None
        if not self.backend:
            return
        assert self.backend == "cw", self.backend
        ids = [self.spec_ids(seed, "enroll_3x4", dev_ids, 3)]
        for dur in DURS:
            ids.append(self.spec_ids(seed, f"test{dur}", dev_ids, len(splits.TEST_GROUPS[dur])))
        x = np.asarray(self.emb[np.concatenate(ids).astype(np.int64)], dtype=np.float64)
        mu = x.mean(axis=0)
        lam, vec = np.linalg.eigh(np.cov(x - mu, rowvar=False))
        lam = np.maximum(lam, 0.0) + 1e-3 * lam.mean()  # small ridge against near-empty directions
        w = vec @ np.diag(lam ** -0.5) @ vec.T
        self.transform = lambda e: rules.l2n((e - mu) @ w)
        self.backend_info = {"n_dev_vectors": int(x.shape[0]), "dim": int(x.shape[1]),
                             "ridge": float(1e-3 * lam.mean())}

    def _tf(self, e: np.ndarray) -> np.ndarray:
        return e if self.transform is None else self.transform(e)

    def has(self, seed: int, role: str) -> bool:
        return (seed, role) in self.roles

    def spec_ids(self, seed: int, role: str, spk_ids: np.ndarray, n_per: int) -> np.ndarray:
        ser = self.roles[(seed, role)]
        idx = pd.MultiIndex.from_product([spk_ids, range(n_per)])
        return ser.reindex(idx).to_numpy()

    def enroll(self, seed: int, cfg: str, spk_ids: np.ndarray, trim: bool = False) -> np.ndarray:
        n_sent = splits.ENROLL_CONFIGS[cfg][0]
        role = f"enroll_{cfg}" + ("_trim" if trim else "")
        ids = self.spec_ids(seed, role, spk_ids, n_sent)
        e = self._tf(np.asarray(self.emb[ids.astype(np.int64)], dtype=np.float64))
        return e.reshape(len(spk_ids), n_sent, -1)

    def tests(self, seed: int, role: str, spk_ids: np.ndarray, n_per: int):
        ids = self.spec_ids(seed, role, spk_ids, n_per).astype(np.int64)
        q = self._tf(np.asarray(self.emb[ids], dtype=np.float64))
        a = np.asarray(self.audio[ids], dtype=np.float64)
        owner = np.repeat(np.arange(len(spk_ids)), n_per)
        return q, owner, a[:, 0], a[:, 1], a[:, 2]


# ---------------------------------------------------------------- households
def households(rng: np.random.Generator, m: int, k: int, r: int,
               target_pools: list[np.ndarray] | None = None) -> np.ndarray:
    """(m, r, k) indices of k distinct speakers != i, for each speaker i.

    target_pools: optional per-speaker candidate index arrays (already excluding i).
    """

    out = np.empty((m, r, k), dtype=np.int32)
    for i in range(m):
        pool = target_pools[i] if target_pools is not None else np.delete(np.arange(m), i)
        if pool.size < k:
            out[i] = -1
            continue
        keys = rng.random((r, pool.size))
        out[i] = pool[np.argpartition(keys, k - 1, axis=1)[:, :k]]
    return out


def gather(scores: np.ndarray, owner: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """scores (U, M); cand (U, R, N) -> (U, R, N)."""

    u = np.arange(scores.shape[0])[:, None, None]
    return scores[u, cand]


class TrialSet:
    """Per-trial summaries for one (set, duration, N, member|stranger) cell."""

    def __init__(self, cos: np.ndarray, blend: np.ndarray, asn: np.ndarray | None, cand: np.ndarray,
                 loo: np.ndarray, ok: np.ndarray, pen: np.ndarray, owner: np.ndarray, member: bool) -> None:
        sc = gather(cos, owner, cand)
        sb = gather(blend, owner, cand)
        n = cand.shape[-1]
        bc = np.argmax(sc, axis=-1)
        bb = np.argmax(sb, axis=-1)
        self.cos_best = np.take_along_axis(sc, bc[..., None], -1)[..., 0]
        self.blend_best = np.take_along_axis(sb, bb[..., None], -1)[..., 0]
        cand_b = np.take_along_axis(cand, bb[..., None], -1)[..., 0]
        self.loo_best = loo[cand_b]
        if n > 1:
            second = np.partition(sb, n - 2, axis=-1)[..., n - 2]
            self.blend_gap = self.blend_best - second
            secc = np.partition(sc, n - 2, axis=-1)[..., n - 2]
            self.cos_gap = self.cos_best - secc
        else:
            self.blend_gap = np.full(self.blend_best.shape, np.inf)
            self.cos_gap = np.full(self.cos_best.shape, np.inf)
        self.ok = np.broadcast_to(ok[:, None], self.cos_best.shape)
        self.pen = np.broadcast_to(pen[:, None], self.cos_best.shape)
        self.member = member
        if asn is not None:
            sa = gather(asn, owner, cand)
            ba = np.argmax(sa, axis=-1)
            self.asn_best = np.take_along_axis(sa, ba[..., None], -1)[..., 0]
            if n > 1:  # runner-up by AS-norm; the margin is measured on the cosines of the two
                np.put_along_axis(sa, ba[..., None], -np.inf, -1)
                b2 = np.argmax(sa, axis=-1)
                self.asn_cos_gap = (np.take_along_axis(sc, ba[..., None], -1)[..., 0]
                                    - np.take_along_axis(sc, b2[..., None], -1)[..., 0])
            else:
                self.asn_cos_gap = np.full(self.asn_best.shape, np.inf)
            del sa
        if member:  # the true speaker is candidate 0
            self.cos_t = bc == 0
            self.blend_t = bb == 0
            if asn is not None:
                self.asn_t = ba == 0
        self.owner = owner

    # --- decision rules: return (accepted, eligible-but-rejected-reasons)
    def rule(self, name: str, param: float | None = None) -> dict[str, np.ndarray]:
        """Returns boolean arrays: named (any member named), and for members correct/wrong."""

        ok = self.ok
        if name == "nexaroom":  # full rule with base threshold param (default 0.5), floor param-0.15
            tau = rules.BASE_THRESHOLD if param is None else param
            thr = rules.calibrated_thresholds(self.loo_best, base=tau, floor=tau - (rules.BASE_THRESHOLD - rules.FLOOR))
            above = self.blend_best >= thr + self.pen
            amb = self.blend_gap < rules.MARGIN
            named = ok & above & ~amb
            parts = {"unknown": ok & ~above, "ambiguous": ok & above & amb, "too_short": ~ok}
            t = self.blend_t if self.member else None
        elif name == "asnorm":  # AS-norm cosine to centroid, fixed threshold, no margin
            named = ok & (self.asn_best >= param)
            parts = {"unknown": ok & ~named, "ambiguous": np.zeros_like(named), "too_short": ~ok}
            t = self.asn_t if self.member else None
        elif name == "cos":  # plain cosine to centroid, fixed threshold, no margin
            named = ok & (self.cos_best >= param)
            parts = {"unknown": ok & ~named, "ambiguous": np.zeros_like(named), "too_short": ~ok}
            t = self.cos_t if self.member else None
        elif name in ("cosm", "blendm", "asnormm"):  # family score, fixed threshold, 0.06 runner-up margin
            score = {"cosm": self.cos_best, "blendm": self.blend_best, "asnormm": self.asn_best}[name]
            gap = {"cosm": self.cos_gap, "blendm": self.blend_gap, "asnormm": self.asn_cos_gap}[name]
            above = score >= param
            amb = gap < rules.MARGIN
            named = ok & above & ~amb
            parts = {"unknown": ok & ~above, "ambiguous": ok & above & amb, "too_short": ~ok}
            t = {"cosm": getattr(self, "cos_t", None), "blendm": getattr(self, "blend_t", None),
                 "asnormm": getattr(self, "asn_t", None)}[name] if self.member else None
        elif name.startswith("ladder_"):
            step = name.split("_", 1)[1]
            score = self.cos_best if step == "a" else self.blend_best
            if step in ("a", "b"):
                thr = np.full(score.shape, rules.BASE_THRESHOLD)
            else:
                thr = rules.calibrated_thresholds(self.loo_best)
            if step in ("d", "e"):
                thr = thr + self.pen
            above = score >= thr
            amb = (self.blend_gap < rules.MARGIN) if step == "e" else np.zeros_like(above)
            named = ok & above & ~amb
            parts = {"unknown": ok & ~above, "ambiguous": ok & above & amb, "too_short": ~ok}
            t = (self.cos_t if step == "a" else self.blend_t) if self.member else None
        else:
            raise ValueError(name)
        out = {"named": named, **parts}
        if self.member:
            out["correct"] = named & t
            out["wrong"] = named & ~t
        return out

    def accept_score(self, family: str) -> tuple[np.ndarray, np.ndarray]:
        """(score a_t, eligible mask): accepted iff eligible and a_t >= parameter."""

        if family == "cos":
            return self.cos_best, self.ok
        if family == "asnorm":
            return self.asn_best, self.ok
        if family == "cosm":
            return self.cos_best, self.ok & (self.cos_gap >= rules.MARGIN)
        if family == "blendm":
            return self.blend_best, self.ok & (self.blend_gap >= rules.MARGIN)
        if family == "asnormm":
            return self.asn_best, self.ok & (self.asn_cos_gap >= rules.MARGIN)
        x = self.blend_best - self.pen
        c = self.loo_best - rules.CALIB_OFFSET
        span = rules.BASE_THRESHOLD - rules.FLOOR
        tau_star = np.where(x >= c, x + span, x)
        return tau_star, self.ok & (self.blend_gap >= rules.MARGIN)


def per_speaker(x: np.ndarray, owner: np.ndarray, m: int) -> np.ndarray:
    v = x.reshape(x.shape[0], -1).mean(axis=1)
    s = np.bincount(owner, weights=v, minlength=m)
    c = np.bincount(owner, minlength=m)
    return s / np.maximum(c, 1)


def macro_survival(a: np.ndarray, elig: np.ndarray, owner: np.ndarray, m: int, grid: np.ndarray,
                   extra: np.ndarray | None = None) -> np.ndarray:
    """Macro (per-speaker averaged) fraction of trials with elig & extra & a >= g, for g in grid."""

    mask = elig if extra is None else (elig & extra)
    n_per = np.bincount(owner, minlength=m) * a.shape[1]
    w = (1.0 / np.maximum(n_per[owner], 1) / m)[:, None] * np.ones(a.shape[1])
    av = a[mask]
    wv = w[mask]
    order = np.argsort(av)
    av, wv = av[order], wv[order]
    tail = np.concatenate([np.cumsum(wv[::-1])[::-1], [0.0]])
    return tail[np.searchsorted(av, grid, side="left")]


def tune(a: np.ndarray, elig: np.ndarray, owner: np.ndarray, m: int, target: float) -> float:
    """Smallest parameter with macro FA <= target (exact, from sorted trial scores)."""

    return tune_multi([(a, elig, owner, m)], target)


def tune_multi(parts: list[tuple], target: float) -> float:
    """As tune(), with the FA averaged over several trial sets (equal weight each),
    e.g. one threshold for all utterance durations."""

    avs, wvs = [], []
    for a, elig, owner, m in parts:
        n_per = np.bincount(owner, minlength=m) * a.shape[1]
        w = (1.0 / np.maximum(n_per[owner], 1) / m / len(parts))[:, None] * np.ones(a.shape[1])
        avs.append(a[elig])
        wvs.append(w[elig])
    av = np.concatenate(avs)
    wv = np.concatenate(wvs)
    order = np.argsort(av)[::-1]
    av, wv = av[order], wv[order]
    cum = np.cumsum(wv)  # FA if threshold just at/below av[i]
    # choose threshold slightly above av[i] where cum[i] > target first happens
    over = np.flatnonzero(cum > target + 1e-12)
    if over.size == 0:
        return float(av[-1]) if av.size else 0.0
    i = over[0]
    return float(np.nextafter(av[i], np.inf))


# ---------------------------------------------------------------- accumulators
class Acc:
    def __init__(self, n_spk: int, pop: np.ndarray | None = None) -> None:
        self.n = n_spk
        self.pop = np.arange(n_spk) if pop is None else pop  # bootstrap population (eval speakers)
        self.vals: dict[tuple, np.ndarray] = {}

    def put(self, key: tuple, seed: int, spk_ids: np.ndarray, v: np.ndarray) -> None:
        if key not in self.vals:
            self.vals[key] = np.full((len(splits.SEEDS), self.n), np.nan, dtype=np.float64)
        self.vals[key][splits.SEEDS.index(seed), spk_ids] = v

    def summary(self, fields: list[str], rng: np.random.Generator) -> pd.DataFrame:
        w = metrics.boot_weights(len(self.pop), rng)
        rows = []
        for key, arr in self.vals.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                per_spk = np.nanmean(arr, axis=0)[self.pop]
                seed_means = np.nanmean(arr, axis=1)
            est, lo, hi = metrics.boot_mean(per_spk, w)
            row = dict(zip(fields, key))
            row.update({"est": est, "lo": lo, "hi": hi, "n_spk": int(np.sum(~np.isnan(per_spk))),
                        "seed_sd": float(np.nanstd(seed_means, ddof=1))})
            for s, v in zip(splits.SEEDS, seed_means):
                row[f"seed{s}"] = float(v)
            rows.append(row)
        return pd.DataFrame(rows)


class EerAcc:
    def __init__(self, n_spk: int) -> None:
        self.n = n_spk
        self.h: dict[tuple, list] = {}
        self.per_seed: dict[tuple, list] = {}

    def put(self, key: tuple, seed: int, spk_ids: np.ndarray, tgt: np.ndarray, tgt_owner: np.ndarray,
            non: np.ndarray, non_owner: np.ndarray, edges: np.ndarray) -> None:
        ht = metrics.per_speaker_hist(tgt, tgt_owner, len(spk_ids), edges)
        hn = metrics.per_speaker_hist(non, non_owner, len(spk_ids), edges)
        if key not in self.h:
            self.h[key] = [np.zeros((self.n, edges.size - 1)), np.zeros((self.n, edges.size - 1)), edges]
            self.per_seed[key] = [np.nan] * len(splits.SEEDS)
        self.h[key][0][spk_ids] += ht
        self.h[key][1][spk_ids] += hn
        e, _ = metrics.eer_from_hist(ht.sum(0), hn.sum(0))
        self.per_seed[key][splits.SEEDS.index(seed)] = float(e)

    def summary(self, fields: list[str], rng: np.random.Generator) -> pd.DataFrame:
        rows = []
        for key, (ht, hn, edges) in self.h.items():
            keep = (ht.sum(1) > 0) & (hn.sum(1) > 0)
            w = metrics.boot_weights(int(keep.sum()), rng)
            r = metrics.boot_eer(ht[keep], hn[keep], w)
            row = dict(zip(fields, key))
            row.update(r)
            # score distribution quantiles (pooled trials, bin upper edges)
            for nm, h, qs in (("target", ht, (0.05, 0.5, 0.95)), ("nontarget", hn, (0.5, 0.95, 0.99, 0.999))):
                c = np.cumsum(h.sum(0))
                c = c / c[-1]
                for qv in qs:
                    row[f"{nm}_q{qv:g}"] = float(edges[1:][np.searchsorted(c, qv)])
            row["n_spk"] = int(keep.sum())
            ps = np.array(self.per_seed[key], dtype=float)
            row["seed_sd"] = float(np.nanstd(ps, ddof=1))
            for s, v in zip(splits.SEEDS, ps):
                row[f"seed{s}"] = v
            rows.append(row)
        return pd.DataFrame(rows)


def eer_trials(scores: np.ndarray, owner: np.ndarray, pair_mask: np.ndarray | None = None):
    """Target / non-target scores of a (U, M) matrix where model j belongs to speaker j."""

    u = np.arange(scores.shape[0])
    tgt = scores[u, owner]
    mask = np.ones_like(scores, dtype=bool)
    mask[u, owner] = False
    if pair_mask is not None:
        mask &= pair_mask
    non_owner = np.broadcast_to(owner[:, None], scores.shape)[mask]
    return tgt, owner, scores[mask], non_owner


EDGES = {"cos": metrics.hist_edges(-1.0, 1.0, 4000), "blend": metrics.hist_edges(-1.0, 1.0, 4000),
         "asnorm": metrics.hist_edges(-15.0, 45.0, 6000)}


# ---------------------------------------------------------------- main evaluation
def evaluate(model: str, ablations: bool, backend: str = "") -> None:
    t0 = time.time()
    st = Store(model, backend)
    model = st.config
    out = data.RESULTS / model
    out.mkdir(parents=True, exist_ok=True)
    backend_info = []
    main = Acc(st.n_spk, st.pop)
    eer = EerAcc(st.n_spk)
    curves: list[dict] = []
    tuned: list[dict] = []
    gen = Acc(st.n_spk, st.pop)
    gen_eer = EerAcc(st.n_spk)
    fw = Acc(st.n_spk, st.pop)
    fw_eer = EerAcc(st.n_spk)
    abl = Acc(st.n_spk, st.pop)
    abl_eer = EerAcc(st.n_spk)
    counts: dict = {"too_short_1s": [], "enroll_audio_problem": []}

    for seed in splits.SEEDS:
        sp = st.splits[seed]
        ids_all = np.array(sorted(sp))
        dev_ids = np.array([k for k in ids_all if sp[k]["dev"]])
        ev_ids = np.array([k for k in ids_all if not sp[k]["dev"]])
        sets = {"eval": ev_ids, "dev": dev_ids}
        st.fit_backend(seed, dev_ids)
        if st.transform is not None:
            backend_info.append({"seed": seed, **st.backend_info})
        enr = {name: st.enroll(seed, "3x4", ids) for name, ids in sets.items()}
        cen = {name: rules.centroids(e) for name, e in enr.items()}
        loo = {name: rules.loo_floor(e) for name, e in enr.items()}
        crng = np.random.default_rng([seed, 99])
        cohort = {"eval": cen["dev"],
                  "dev": cen["eval"][np.sort(crng.choice(len(ev_ids), len(dev_ids), replace=False))]}
        # NexaRoom refuses an enrollment sentence whose audio fails audio_problem(); report how often
        eids = st.spec_ids(seed, "enroll_3x4", ids_all, 3).astype(np.int64)
        ea = np.asarray(st.audio[eids])
        counts["enroll_audio_problem"].append(float(np.mean(~rules.audio_ok(ea[:, 0], ea[:, 1], ea[:, 2], 2.0))))
        hh: dict = {}  # households depend on (set, N) only, shared by all durations and rules
        pooled: dict = {}  # accept scores for one-threshold-for-all-durations tuning
        for name, ids in sets.items():
            for n_h in NS:
                hrng = np.random.default_rng([seed, n_h, 0 if name == "eval" else 1, 7])
                hh[(name, n_h)] = (households(hrng, len(ids), n_h - 1, R), households(hrng, len(ids), n_h, R))
        for dur in DURS:
            cell: dict = {}
            n_per = len(splits.TEST_GROUPS[dur])
            for name, ids in sets.items():
                q, owner, sec, rms, clp = st.tests(seed, f"test{dur}", ids, n_per)
                blend, cos = rules.profile_score_matrix(q, enr[name])
                ok = rules.audio_ok(sec, rms, clp)
                pen = rules.short_penalty(sec)
                if dur == 1 and name == "eval":
                    counts["too_short_1s"].append(float(np.mean(~ok)))
                # AS-norm cohort: enrollment centroids of the other speaker set (disjoint speakers);
                # for dev trials a random subset of eval speakers of the same size as dev, so that
                # thresholds tuned on dev transfer to eval
                asn = rules.asnorm(cos, q, cen[name], cohort[name])
                if name == "eval":
                    for stype, mat in (("cos", cos), ("blend", blend), ("asnorm", asn)):
                        tg, to, nn, no = eer_trials(mat, owner)
                        eer.put((dur, stype), seed, ids, tg, to, nn, no, EDGES[stype])
                for n_h in NS:
                    comps, strangers = hh[(name, n_h)]
                    cand_m = np.concatenate([np.broadcast_to(owner[:, None, None], (owner.size, R, 1)),
                                             comps[owner]], axis=-1)
                    cand_s = strangers[owner]
                    tm = TrialSet(cos, blend, asn, cand_m, loo[name], ok, pen, owner, True)
                    ts = TrialSet(cos, blend, asn, cand_s, loo[name], ok, pen, owner, False)
                    cell[(name, dur, n_h)] = (tm, ts, owner, len(ids))
                    if n_h in POOLED_NS:
                        for fam in ("cos", "nexaroom", "asnorm"):
                            a_s, e_s = ts.accept_score(fam)
                            entry = {"s": (a_s, e_s, owner, len(ids))}
                            if name == "eval":
                                a_m, e_m = tm.accept_score(fam)
                                t_m = {"cos": tm.cos_t, "nexaroom": tm.blend_t, "asnorm": tm.asn_t}[fam]
                                entry["m"] = (a_m, e_m, t_m, owner, len(ids))
                            pooled[(name, dur, n_h, fam)] = entry
                    if name == "eval":
                        m = len(ids)
                        for sname, arr in (("cos", tm.cos_t), ("blend", tm.blend_t), ("asnorm", tm.asn_t)):
                            main.put((dur, n_h, "closed_set", f"acc_{sname}"), seed, ids, per_speaker(arr, owner, m))
                        rule_list = [("nexaroom", None, "nexaroom_default"), ("cos", 0.5, "cos@0.50")]
                        rule_list += [(f"ladder_{s}", None, f"ladder_{s}") for s in "abcde"]
                        for rname, param, label in rule_list:
                            rm = tm.rule(rname, param)
                            rs = ts.rule(rname, param)
                            for k in ("correct", "wrong", "unknown", "ambiguous", "too_short"):
                                main.put((dur, n_h, label, f"member_{k}"), seed, ids, per_speaker(rm[k], owner, m))
                            main.put((dur, n_h, label, "member_reject"), seed, ids,
                                     per_speaker(~rm["named"], owner, m))
                            main.put((dur, n_h, label, "stranger_fa"), seed, ids, per_speaker(rs["named"], owner, m))
                            main.put((dur, n_h, label, "stranger_ambiguous"), seed, ids,
                                     per_speaker(rs["ambiguous"], owner, m))
                        # curves on eval
                        for fam, grid in (("cos", COS_GRID), ("nexaroom", TAU_GRID), ("asnorm", ASN_GRID)):
                            am, em = tm.accept_score(fam)
                            as_, es = ts.accept_score(fam)
                            tmask = {"cos": tm.cos_t, "nexaroom": tm.blend_t, "asnorm": tm.asn_t}[fam]
                            ca = macro_survival(am, em, owner, m, grid, tmask)
                            wa = macro_survival(am, em, owner, m, grid, ~tmask)
                            fa = macro_survival(as_, es, owner, m, grid)
                            for g, c_, w_, f_ in zip(grid, ca, wa, fa):
                                curves.append({"seed": seed, "dur": dur, "N": n_h, "family": fam, "param": float(g),
                                               "member_correct": c_, "member_wrong": w_, "stranger_fa": f_})
            # threshold tuning on dev, applied to eval
            for n_h in NS:
                tm_d, ts_d, ow_d, m_d = cell[("dev", dur, n_h)]
                tm_e, ts_e, ow_e, m_e = cell[("eval", dur, n_h)]
                # NexaRoom default FA on dev -> matched plain-cosine threshold
                fa_dev_nx = per_speaker(ts_d.rule("nexaroom")["named"], ow_d, m_d).mean()
                a, e = ts_d.accept_score("cos")
                targets = [("matched", fa_dev_nx)] + [(f"fa{int(t*100)}", t) for t in FA_TARGETS]
                for tname, target in targets:
                    th_cos = tune(a, e, ow_d, m_d, target)
                    entries = [("cos", th_cos, f"cos@dev_{tname}")]
                    fams = ["cosm"] if tname == "matched" else ["nexaroom", "asnorm", "cosm", "blendm", "asnormm"]
                    for fam in fams:
                        a2, e2 = ts_d.accept_score(fam)
                        entries.append((fam, tune(a2, e2, ow_d, m_d, target), f"{fam}@dev_{tname}"))
                    for fam, th, label in entries:
                        tuned.append({"seed": seed, "dur": dur, "N": n_h, "label": label, "threshold": th,
                                      "dev_target_fa": target})
                        rm = tm_e.rule(fam, th)
                        rs = ts_e.rule(fam, th)
                        for k in ("correct", "wrong"):
                            main.put((dur, n_h, label, f"member_{k}"), seed, ev_ids, per_speaker(rm[k], ow_e, m_e))
                        main.put((dur, n_h, label, "member_reject"), seed, ev_ids, per_speaker(~rm["named"], ow_e, m_e))
                        main.put((dur, n_h, label, "stranger_fa"), seed, ev_ids, per_speaker(rs["named"], ow_e, m_e))
        # one threshold for all durations (tuned on dev, pooled 1/2/3 s), applied to eval per duration
        for n_h in POOLED_NS:
            for target in FA_TARGETS:
                for fam in ("cos", "nexaroom", "asnorm"):
                    th = tune_multi([pooled[("dev", d, n_h, fam)]["s"] for d in DURS], target)
                    label = f"{fam}@pooled_fa{int(target * 100)}"
                    tuned.append({"seed": seed, "dur": "pooled", "N": n_h, "label": label, "threshold": th,
                                  "dev_target_fa": target})
                    for d in DURS:
                        a_m, e_m, t_m, ow, m = pooled[("eval", d, n_h, fam)]["m"]
                        a_s, e_s, ow_s, _ = pooled[("eval", d, n_h, fam)]["s"]
                        acc = e_m & (a_m >= th)
                        main.put((d, n_h, label, "member_correct"), seed, ev_ids, per_speaker(acc & t_m, ow, m))
                        main.put((d, n_h, label, "member_wrong"), seed, ev_ids, per_speaker(acc & ~t_m, ow, m))
                        main.put((d, n_h, label, "member_reject"), seed, ev_ids, per_speaker(~acc, ow, m))
                        main.put((d, n_h, label, "stranger_fa"), seed, ev_ids,
                                 per_speaker(e_s & (a_s >= th), ow_s, m))
        del pooled
        # ------------------------------------------------ gender (eval, labelled speakers)
        gl = np.array([st.gender.get(k, "NAN") for k in ev_ids])
        lab = np.flatnonzero(np.isin(gl, ["MALE", "FEMALE"]))
        lab_ids = ev_ids[lab]
        lab_g = gl[lab]
        e_lab = enr["eval"][lab]
        loo_lab = loo["eval"][lab]
        for dur in DURS:
            n_per = len(splits.TEST_GROUPS[dur])
            q, owner, sec, rms, clp = st.tests(seed, f"test{dur}", lab_ids, n_per)
            blend, cos = rules.profile_score_matrix(q, e_lab)
            ok = rules.audio_ok(sec, rms, clp)
            pen = rules.short_penalty(sec)
            same_pair = lab_g[owner][:, None] == lab_g[None, :]
            for rel, pm in (("same", same_pair), ("cross", ~same_pair)):
                tg, to, nn, no = eer_trials(cos, owner, pm)
                gen_eer.put((dur, rel), seed, lab_ids, tg, to, nn, no, EDGES["cos"])
            m = len(lab_ids)
            idx = np.arange(m)
            for rel in ("same", "cross", "any"):
                pools = []
                for i in range(m):
                    if rel == "same":
                        p = idx[(lab_g == lab_g[i]) & (idx != i)]
                    elif rel == "cross":
                        p = idx[lab_g != lab_g[i]]
                    else:
                        p = idx[idx != i]
                    pools.append(p)
                hrng = np.random.default_rng([seed, 5, 11, ["same", "cross", "any"].index(rel)])
                comps = households(hrng, m, 4, R, target_pools=pools)
                strangers = households(hrng, m, 5, R, target_pools=pools)
                cand_m = np.concatenate([np.broadcast_to(owner[:, None, None], (owner.size, R, 1)), comps[owner]], -1)
                tm = TrialSet(cos, blend, None, cand_m, loo_lab, ok, pen, owner, True)
                ts = TrialSet(cos, blend, None, strangers[owner], loo_lab, ok, pen, owner, False)
                for gname in ("ALL", "MALE", "FEMALE"):
                    sel = np.ones(m, bool) if gname == "ALL" else (lab_g == gname)
                    vals = {
                        "acc_cos": per_speaker(tm.cos_t, owner, m),
                        "acc_blend": per_speaker(tm.blend_t, owner, m),
                        "nexaroom_member_correct": per_speaker(tm.rule("nexaroom")["correct"], owner, m),
                        "nexaroom_member_wrong": per_speaker(tm.rule("nexaroom")["wrong"], owner, m),
                        "nexaroom_stranger_fa": per_speaker(ts.rule("nexaroom")["named"], owner, m),
                        "cos0.5_stranger_fa": per_speaker(ts.rule("cos", 0.5)["named"], owner, m),
                    }
                    for k, v in vals.items():
                        gen.put((dur, rel, gname, k), seed, lab_ids[sel], v[sel])
        # ------------------------------------------------ ablations
        print(f"[{model}] seed {seed} main done, {(time.time()-t0)/60:.1f} min", flush=True)
        if not ablations:
            continue
        # fixed word vs random word (1 s), within speakers that have the word
        for w in FW_WORDS:
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
            e_w = enr["eval"][pos]
            loo_w = loo["eval"][pos]
            m = len(have)
            for cond in ("fixed_word", "random_word"):
                if cond == "fixed_word":
                    q = np.asarray(st.emb[fids], dtype=np.float64)
                    a = np.asarray(st.audio[fids], dtype=np.float64)
                    owner = np.arange(m)
                    sec, rms, clp = a[:, 0], a[:, 1], a[:, 2]
                else:
                    q, owner, sec, rms, clp = st.tests(seed, "test1", have, 6)
                blend, cos = rules.profile_score_matrix(q, e_w)
                ok = rules.audio_ok(sec, rms, clp)
                pen = rules.short_penalty(sec)
                tg, to, nn, no = eer_trials(cos, owner)
                fw_eer.put((w, cond), seed, have, tg, to, nn, no, EDGES["cos"])
                hrng = np.random.default_rng([seed, 5, 21, FW_WORDS.index(w)])
                comps = households(hrng, m, 4, R)
                strangers = households(hrng, m, 5, R)
                cand_m = np.concatenate([np.broadcast_to(owner[:, None, None], (owner.size, R, 1)), comps[owner]], -1)
                tm = TrialSet(cos, blend, None, cand_m, loo_w, ok, pen, owner, True)
                ts = TrialSet(cos, blend, None, strangers[owner], loo_w, ok, pen, owner, False)
                vals = {
                    "acc_cos": per_speaker(tm.cos_t, owner, m),
                    "acc_blend": per_speaker(tm.blend_t, owner, m),
                    "nexaroom_member_correct": per_speaker(tm.rule("nexaroom")["correct"], owner, m),
                    "nexaroom_member_wrong": per_speaker(tm.rule("nexaroom")["wrong"], owner, m),
                    "nexaroom_stranger_fa": per_speaker(ts.rule("nexaroom")["named"], owner, m),
                    "top_blend_score_member": per_speaker(tm.blend_best, owner, m),
                }
                for k, v in vals.items():
                    fw.put((w, cond, k), seed, have, v)
        # silence trimming (enrollment and/or 1 s test), closed-set N=5 + EER
        if st.has(seed, "enroll_3x4_trim") and st.has(seed, "test1_trim"):
            tid = st.spec_ids(seed, "test1_trim", ev_ids, 6).astype(np.int64)
            eid = st.spec_ids(seed, "enroll_3x4_trim", ev_ids, 3).astype(np.int64)
            if not (np.isnan(st.emb[tid, 0]).any() or np.isnan(st.emb[eid, 0]).any()):
                e_tr = st.enroll(seed, "3x4", ev_ids, trim=True)
                m = len(ev_ids)
                for e_name, e_arr in (("raw", enr["eval"]), ("trim", e_tr)):
                    for t_name, role in (("raw", "test1"), ("trim", "test1_trim")):
                        q, owner, *_ = st.tests(seed, role, ev_ids, 6)
                        blend, cos = rules.profile_score_matrix(q, e_arr)
                        tg, to, nn, no = eer_trials(cos, owner)
                        abl_eer.put(("trim", f"enroll_{e_name}+test_{t_name}", 1), seed, ev_ids, tg, to, nn, no,
                                    EDGES["cos"])
                        hrng = np.random.default_rng([seed, 5, 0, 7])
                        comps = households(hrng, m, 4, R)
                        cand_m = np.concatenate([np.broadcast_to(owner[:, None, None], (owner.size, R, 1)),
                                                 comps[owner]], -1)
                        lo_ = rules.loo_floor(e_arr)
                        tm = TrialSet(cos, blend, None, cand_m, lo_, np.ones(owner.size, bool),
                                      np.zeros(owner.size), owner, True)
                        abl.put(("trim", f"enroll_{e_name}+test_{t_name}", 1, "acc_cos"), seed, ev_ids,
                                per_speaker(tm.cos_t, owner, m))
                        abl.put(("trim", f"enroll_{e_name}+test_{t_name}", 1, "acc_blend"), seed, ev_ids,
                                per_speaker(tm.blend_t, owner, m))
        # enrollment size, on eval speakers with >= 40 enrollment clips
        rich = np.array([k for k in ev_ids if len(sp[k]["enroll"]) >= splits.RICH_ENROLL])
        if rich.size >= 50 and all(st.has(seed, f"enroll_{c}") for c in splits.ENROLL_CONFIGS):
            ok_all = True
            arrs = {}
            for cfg in splits.ENROLL_CONFIGS:
                ids = st.spec_ids(seed, f"enroll_{cfg}", rich, splits.ENROLL_CONFIGS[cfg][0]).astype(np.int64)
                if np.isnan(st.emb[ids, 0]).any():
                    ok_all = False
                    break
                arrs[cfg] = st.enroll(seed, cfg, rich)
            if ok_all:
                m = len(rich)
                for dur in DURS:
                    n_per = len(splits.TEST_GROUPS[dur])
                    q, owner, sec, rms, clp = st.tests(seed, f"test{dur}", rich, n_per)
                    ok = rules.audio_ok(sec, rms, clp)
                    pen = rules.short_penalty(sec)
                    hrng = np.random.default_rng([seed, 5, 31, dur])
                    comps = households(hrng, m, 4, R)
                    strangers = households(hrng, m, 5, R)
                    cand_m = np.concatenate([np.broadcast_to(owner[:, None, None], (owner.size, R, 1)),
                                             comps[owner]], -1)
                    for cfg, e_arr in arrs.items():
                        blend, cos = rules.profile_score_matrix(q, e_arr)
                        lo_ = rules.loo_floor(e_arr)
                        tg, to, nn, no = eer_trials(cos, owner)
                        abl_eer.put(("enroll", cfg, dur), seed, rich, tg, to, nn, no, EDGES["cos"])
                        tm = TrialSet(cos, blend, None, cand_m, lo_, ok, pen, owner, True)
                        ts = TrialSet(cos, blend, None, strangers[owner], lo_, ok, pen, owner, False)
                        rm = tm.rule("nexaroom")
                        vals = {
                            "acc_cos": per_speaker(tm.cos_t, owner, m),
                            "acc_blend": per_speaker(tm.blend_t, owner, m),
                            "nexaroom_member_correct": per_speaker(rm["correct"], owner, m),
                            "nexaroom_member_wrong": per_speaker(rm["wrong"], owner, m),
                            "nexaroom_member_reject": per_speaker(~rm["named"], owner, m),
                            "nexaroom_stranger_fa": per_speaker(ts.rule("nexaroom")["named"], owner, m),
                            "calibrated_threshold": per_speaker(
                                rules.calibrated_thresholds(lo_)[owner][:, None], owner, m),
                        }
                        for k, v in vals.items():
                            abl.put(("enroll", cfg, dur, k), seed, rich, v)
        print(f"[{model}] seed {seed} done, {(time.time()-t0)/60:.1f} min", flush=True)

    rng = np.random.default_rng(12345)
    main.summary(["dur", "N", "rule", "metric"], rng).to_csv(out / "main.csv", index=False)
    eer.summary(["dur", "score"], rng).to_csv(out / "eer.csv", index=False)
    gen.summary(["dur", "household_gender", "target_gender", "metric"], rng).to_csv(out / "gender.csv", index=False)
    gen_eer.summary(["dur", "nontarget_gender"], rng).to_csv(out / "gender_eer.csv", index=False)
    pd.DataFrame(curves).groupby(["dur", "N", "family", "param"]).mean(numeric_only=True).drop(columns="seed") \
        .reset_index().to_csv(out / "curves.csv", index=False)
    pd.DataFrame(tuned).to_csv(out / "tuned_thresholds.csv", index=False)
    if ablations:
        fw.summary(["word", "condition", "metric"], rng).to_csv(out / "fixed_word.csv", index=False)
        fw_eer.summary(["word", "condition"], rng).to_csv(out / "fixed_word_eer.csv", index=False)
        abl.summary(["ablation", "config", "dur", "metric"], rng).to_csv(out / "ablations.csv", index=False)
        abl_eer.summary(["ablation", "config", "dur"], rng).to_csv(out / "ablations_eer.csv", index=False)
    # per-speaker values of the main grid, for re-analysis
    rows = []
    for key, arr in main.vals.items():
        if key[2] not in ("nexaroom_default", "cos@0.50", "closed_set"):
            continue
        for si, s in enumerate(splits.SEEDS):
            ok = ~np.isnan(arr[si])
            for spk, v in zip(np.flatnonzero(ok), arr[si][ok]):
                rows.append((*key, s, int(spk), float(v)))
    pd.DataFrame(rows, columns=["dur", "N", "rule", "metric", "seed", "spk", "value"]).to_csv(
        out / "per_speaker_main.csv.gz", index=False, float_format="%.5f")
    with open(out / "counts.json", "w") as f:
        json.dump({k: v for k, v in counts.items()}, f, indent=1)
    meta = {"config": model, "backend": st.backend or "none", "bootstrap_population": int(len(st.pop)),
            "backend_fit": backend_info}
    with open(out / "eval_meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[{model}] finished in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--backend", default="", choices=["", "cw"])
    a = ap.parse_args()
    evaluate(a.model, a.ablations, a.backend)
