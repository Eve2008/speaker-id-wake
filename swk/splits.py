"""Speaker pool and per-seed enrollment/test splits.

For every seed and speaker, the speaker's Common Voice source recordings
(``src``; one MSWC clip is one word cut from one source sentence) are shuffled.
Whole source recordings are assigned to the test side until it holds 6 clips;
all other source recordings form the enrollment side. Enrollment and test
therefore never share a source recording.

Enrollment "sentences" are consecutive chunks of the shuffled enrollment
clips, so the configurations are nested: 3x4 uses clips [0:12], 5x4 uses
[0:20] (its first three sentences are the 3x4 sentences), 3x8 uses [0:24],
5x8 uses [0:40] (first three sentences are the 3x8 sentences).

Test utterances: 1 clip -> t0..t5 (6 utterances); 2 clips -> (t0 t1), (t2 t3),
(t4 t5); 3 clips -> (t0 t1 t2), (t3 t4 t5). Durations are exactly 1, 2, 3 s.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

N_TEST = 6
MIN_ENROLL = 12   # 3 sentences x 4 clips
RICH_ENROLL = 40  # 5 sentences x 8 clips
MIN_CLIPS = N_TEST + MIN_ENROLL
SEEDS = (0, 1, 2, 3, 4)
DEV_FRACTION = 1.0 / 3.0

ENROLL_CONFIGS = {  # name -> (sentences, clips per sentence)
    "3x4": (3, 4),
    "5x4": (5, 4),
    "3x8": (3, 8),
    "5x8": (5, 8),
}
TEST_GROUPS = {
    1: [(0,), (1,), (2,), (3,), (4,), (5,)],
    2: [(0, 1), (2, 3), (4, 5)],
    3: [(0, 1, 2), (3, 4, 5)],
}


def _rng(seed: int, speaker_hash: str) -> np.random.Generator:
    h = int(hashlib.sha256(f"{seed}:{speaker_hash}".encode()).hexdigest()[:16], 16)
    return np.random.default_rng(h)


def speaker_table(meta: pd.DataFrame) -> pd.DataFrame:
    g = meta.groupby("speaker_hash").agg(n_clips=("clip", "size"), n_src=("src", "nunique"),
                                        gender=("gender", "first")).reset_index()
    g = g[g.n_clips >= MIN_CLIPS].sort_values("speaker_hash").reset_index(drop=True)
    g.insert(0, "spk", np.arange(len(g), dtype=np.int64))
    return g


def split_speaker(seed: int, speaker_hash: str, clips_by_src: dict[str, list[int]]) -> dict | None:
    rng = _rng(seed, speaker_hash)
    srcs = sorted(clips_by_src)
    for _ in range(200):
        order = rng.permutation(len(srcs))
        test: list[int] = []
        test_src: list[str] = []
        k = 0
        while len(test) < N_TEST and k < len(order):
            s = srcs[order[k]]
            c = list(clips_by_src[s])
            rng.shuffle(c)
            test.extend(c)
            test_src.append(s)
            k += 1
        enroll = [c for s in (srcs[i] for i in order[k:]) for c in clips_by_src[s]]
        if len(test) >= N_TEST and len(enroll) >= MIN_ENROLL:
            enroll = list(np.asarray(enroll)[rng.permutation(len(enroll))])
            return {"test": [int(c) for c in test[:N_TEST]], "test_src": test_src,
                    "enroll": [int(c) for c in enroll[:RICH_ENROLL]]}
    return None


def make_splits(meta: pd.DataFrame, seeds=SEEDS) -> tuple[pd.DataFrame, dict[int, dict[int, dict]]]:
    spk = speaker_table(meta)
    by = meta[meta.speaker_hash.isin(spk.speaker_hash)].groupby(["speaker_hash", "src"]).clip.apply(list)
    clips = {}
    for (h, s), c in by.items():
        clips.setdefault(h, {})[s] = c
    out: dict[int, dict[int, dict]] = {}
    ok = np.ones(len(spk), dtype=bool)
    for seed in seeds:
        out[seed] = {}
        for row in spk.itertuples():
            r = split_speaker(seed, row.speaker_hash, clips[row.speaker_hash])
            if r is None:
                ok[row.spk] = False
            else:
                out[seed][row.spk] = r
    # keep only speakers that could be split under every seed
    keep = set(spk.spk[ok])
    for seed in seeds:
        out[seed] = {k: v for k, v in out[seed].items() if k in keep}
    spk = spk[ok].reset_index(drop=True)
    # dev/eval speaker split, redrawn per seed
    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        ids = np.array(sorted(out[seed]))
        dev = set(ids[rng.permutation(ids.size)[: int(round(ids.size * DEV_FRACTION))]].tolist())
        for k in out[seed]:
            out[seed][k]["dev"] = k in dev
    return spk, out
