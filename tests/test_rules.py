"""Check swk.rules against NexaRoom's own SpeakerEngine (work copy, read-only).

Run: python -m pytest -q tests  (or python tests/test_rules.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/ali/Desktop/Articles/work/NexaRoom")

from nexaroom.voice.speakers import SpeakerEngine, SpeakerProfile  # noqa: E402

from swk import rules  # noqa: E402


def _unit(rng, *shape):
    return rules.l2n(rng.standard_normal(shape))


def _correlated(rng, n_spk, k, d, spread):
    base = _unit(rng, n_spk, d)
    return rules.l2n(base[:, None, :] + spread * rng.standard_normal((n_spk, k, d)))


def test_profile_score_and_calibration():
    rng = np.random.default_rng(0)
    for k in (3, 4, 5):
        enroll = _correlated(rng, 7, k, 32, 0.6)
        q = _unit(rng, 11, 32)
        blend, _ = rules.profile_score_matrix(q, enroll)
        floors = rules.loo_floor(enroll)
        thr = rules.calibrated_thresholds(floors)
        eng = SpeakerEngine(None)
        for m in range(enroll.shape[0]):
            embs = list(enroll[m])
            for u in range(q.shape[0]):
                assert abs(SpeakerEngine.profile_score(q[u], embs) - blend[u, m]) < 1e-9
            cal = eng.calibrate(embs)
            assert abs(cal["threshold"] - thr[m]) < 1e-9, (cal, thr[m])
            assert abs(min(cal["loo"]) - floors[m]) < 1e-4


def test_identify_decisions_match():
    rng = np.random.default_rng(1)
    eng = SpeakerEngine(None)
    eng.status = lambda: {"ready": True}  # type: ignore[assignment]
    mismatches = 0
    total = 0
    counts = np.zeros(5, dtype=int)
    for trial in range(400):
        n = int(rng.integers(1, 7))
        k = int(rng.integers(3, 6))
        d = 24
        enroll = _correlated(rng, n, k, d, float(rng.uniform(0.3, 1.2)))
        target = int(rng.integers(0, n))
        q = rules.l2n(enroll[target].mean(0) + float(rng.uniform(0.2, 2.0)) * rng.standard_normal(d))
        thr = rules.calibrated_thresholds(rules.loo_floor(enroll))
        profiles = [SpeakerProfile(str(i), str(i), list(enroll[i]), float(thr[i])) for i in range(n)]
        seconds = float(rng.choice([1.0, 1.5, 2.0, 3.0]))
        amp = float(rng.choice([20.0, 2000.0]))
        pcm = (amp * rng.standard_normal(int(seconds * 16000))).clip(-32000, 32000).astype("<i2")
        eng.embed = lambda _pcm, _sr=16000, _q=q: _q  # type: ignore[assignment]
        ref = eng.identify(pcm.tobytes(), profiles)
        blend, _ = rules.profile_score_matrix(q[None], enroll)
        s, r, c = seconds, float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2))), 0.0
        ok = rules.audio_ok(np.array(s), np.array(r), np.array(c))
        status, best = rules.decide(blend[0], thr, rules.short_penalty(np.array(s)), rules.MARGIN, ok)
        mine = {0: "identified", rules.UNKNOWN: "unknown", rules.AMBIGUOUS: "ambiguous",
                rules.TOO_SHORT: "too_short"}[int(status)]
        counts[int(status)] += 1
        total += 1
        if mine != ref["status"] or (mine == "identified" and ref["id"] != str(int(best))):
            mismatches += 1
    assert mismatches == 0, mismatches
    # make sure every branch was exercised
    assert counts[0] > 0 and counts[rules.UNKNOWN] > 0 and counts[rules.TOO_SHORT] > 0
    assert counts[rules.AMBIGUOUS] > 0, counts


if __name__ == "__main__":
    test_profile_score_and_calibration()
    test_identify_decisions_match()
    print("ok")
