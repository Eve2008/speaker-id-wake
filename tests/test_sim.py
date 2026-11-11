"""Checks for the household sampler, the exact threshold tuner and the tau* shortcut."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("ev", HERE / "experiments" / "03_evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)  # type: ignore[union-attr]

from swk import rules  # noqa: E402


def test_households_distinct_and_exclude_self():
    rng = np.random.default_rng(0)
    h = ev.households(rng, 40, 19, 50)
    for i in range(40):
        assert not np.any(h[i] == i)
        for r in range(50):
            assert len(set(h[i, r])) == 19


def test_tune_hits_target():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(300, 20))
    owner = np.repeat(np.arange(60), 5)
    elig = np.ones_like(a, dtype=bool)
    for target in (0.01, 0.05, 0.2):
        th = ev.tune(a, elig, owner, 60, target)
        fa = ev.per_speaker(elig & (a >= th), owner, 60).mean()
        assert fa <= target + 1e-12
        # one step lower must exceed the target (threshold is the smallest one meeting it)
        below = np.max(a[a < th])
        assert ev.per_speaker(elig & (a >= below), owner, 60).mean() > target


def test_tau_star_matches_rule():
    """accept_score('nexaroom') >= tau  <=>  rule('nexaroom', tau) names someone."""

    rng = np.random.default_rng(2)
    m, d = 30, 16
    enroll = rules.l2n(rules.l2n(rng.standard_normal((m, 1, d))) + 0.7 * rng.standard_normal((m, 3, d)))
    q = rules.l2n(enroll.mean(1)[np.repeat(np.arange(m), 4)] + 1.0 * rng.standard_normal((m * 4, d)))
    owner = np.repeat(np.arange(m), 4)
    blend, cos = rules.profile_score_matrix(q, enroll)
    loo = rules.loo_floor(enroll)
    hh = ev.households(rng, m, 4, 20)
    cand = np.concatenate([np.broadcast_to(owner[:, None, None], (owner.size, 20, 1)), hh[owner]], -1)
    sec = np.where(rng.random(owner.size) < 0.5, 1.0, 3.0)
    ts = ev.TrialSet(cos, blend, None, cand, loo, np.ones(owner.size, bool), rules.short_penalty(sec), owner, True)
    a, e = ts.accept_score("nexaroom")
    for tau in (0.3, 0.4, 0.5, 0.6):
        named = ts.rule("nexaroom", tau)["named"]
        fast = e & (a >= tau)
        # thresholds are rounded to 4 decimals in NexaRoom; allow disagreement only at the rounding edge
        assert np.mean(named != fast) < 0.002


if __name__ == "__main__":
    test_households_distinct_and_exclude_self()
    test_tune_hits_target()
    test_tau_star_matches_rule()
    print("ok")
