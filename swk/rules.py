"""Vectorised re-implementation of NexaRoom's speaker scoring and decision rule.

Source: work/NexaRoom/nexaroom/voice/speakers.py (SpeakerEngine.profile_score,
calibrate, identify). tests/test_rules.py checks this module against the
original code on random embeddings.
"""

from __future__ import annotations

import numpy as np

# NexaRoom defaults (speakers.py / config)
BASE_THRESHOLD = 0.5
FLOOR = 0.35
CALIB_OFFSET = 0.08
MARGIN = 0.06
SHORT_PENALTY = 0.05
SHORT_SECONDS = 2.0
MIN_IDENTIFY_SECONDS = 0.8
MIN_RMS = 80.0
MAX_CLIPPED = 0.05
MAX_SECONDS = 30.0

# outcome codes
CORRECT, WRONG, UNKNOWN, AMBIGUOUS, TOO_SHORT = 0, 1, 2, 3, 4


def l2n(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-9)


def centroids(enroll: np.ndarray) -> np.ndarray:
    """enroll: (M, K, D) unit vectors -> (M, D) normalised means."""

    return l2n(enroll.mean(axis=1))


def profile_score_matrix(q: np.ndarray, enroll: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """q: (U, D); enroll: (M, K, D). Returns (blend, cos_centroid), both (U, M).

    blend = 0.5 * cos(q, centroid) + 0.5 * mean of the top-3 sample cosines.
    """

    cos_c = q @ centroids(enroll).T
    sims = np.einsum("ud,mkd->umk", q, enroll, optimize=True)
    k = enroll.shape[1]
    top = min(3, k)
    if top == k:
        topmean = sims.mean(axis=2)
    else:
        topmean = -np.sort(-sims, axis=2)[:, :, :top].mean(axis=2)
    return 0.5 * cos_c + 0.5 * topmean, cos_c


def loo_floor(enroll: np.ndarray) -> np.ndarray:
    """Minimum leave-one-out self score per profile (NexaRoom calibrate)."""

    m, k, _ = enroll.shape
    loo = np.empty((m, k))
    for i in range(k):
        others = np.delete(enroll, i, axis=1)
        q = enroll[:, i, :]
        cen = l2n(others.mean(axis=1))
        cos_c = np.einsum("md,md->m", q, cen)
        sims = np.einsum("md,mkd->mk", q, others)
        top = min(3, k - 1)
        topmean = -np.sort(-sims, axis=1)[:, :top].mean(axis=1)
        loo[:, i] = 0.5 * cos_c + 0.5 * topmean
    return loo.min(axis=1)


def calibrated_thresholds(floor_loo: np.ndarray, base: float = BASE_THRESHOLD, floor: float = FLOOR,
                          offset: float = CALIB_OFFSET) -> np.ndarray:
    thr = np.maximum(floor, np.minimum(base, floor_loo - offset))
    return np.round(thr, 4)


def decide(scores: np.ndarray, thresholds: np.ndarray, penalty: np.ndarray | float, margin: float,
           audio_ok: np.ndarray | bool = True) -> tuple[np.ndarray, np.ndarray]:
    """NexaRoom identify() on candidate sets.

    scores, thresholds: (..., N) with the candidate set on the last axis.
    penalty: broadcastable to (...,). Returns (status, best_index) where status is
    0 identified, UNKNOWN, AMBIGUOUS or TOO_SHORT; best_index is the arg-max.
    """

    n = scores.shape[-1]
    best = np.argmax(scores, axis=-1)
    best_s = np.take_along_axis(scores, best[..., None], axis=-1)[..., 0]
    best_t = np.take_along_axis(thresholds, best[..., None], axis=-1)[..., 0]
    status = np.zeros(best.shape, dtype=np.int8)
    if n > 1 and margin > 0:
        second = np.partition(scores, n - 2, axis=-1)[..., n - 2]
        status[best_s - second < margin] = AMBIGUOUS
    status[best_s < best_t + penalty] = UNKNOWN  # checked first in NexaRoom, so it wins
    status[~np.broadcast_to(np.asarray(audio_ok), best.shape)] = TOO_SHORT
    return status, best


def audio_ok(seconds: np.ndarray, rms: np.ndarray, clipped: np.ndarray,
             minimum_seconds: float = MIN_IDENTIFY_SECONDS) -> np.ndarray:
    return (seconds >= minimum_seconds) & (seconds <= MAX_SECONDS) & (rms >= MIN_RMS) & (clipped <= MAX_CLIPPED)


def short_penalty(seconds: np.ndarray) -> np.ndarray:
    return np.where(seconds < SHORT_SECONDS, SHORT_PENALTY, 0.0)


def asnorm(raw: np.ndarray, q: np.ndarray, models: np.ndarray, cohort: np.ndarray, top_k: int = 100) -> np.ndarray:
    """Adaptive symmetric score normalisation (AS-norm) of a (U, M) cosine matrix.

    q: (U, D) test embeddings, models: (M, D) enrollment centroids,
    cohort: (C, D) centroids of speakers outside the evaluation set.
    """

    def stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = x @ cohort.T
        k = min(top_k, s.shape[1])
        top = np.partition(s, s.shape[1] - k, axis=1)[:, -k:]
        return top.mean(axis=1), top.std(axis=1) + 1e-6

    mu_q, sd_q = stats(q)
    mu_m, sd_m = stats(models)
    return 0.5 * ((raw - mu_m[None, :]) / sd_m[None, :] + (raw - mu_q[:, None]) / sd_q[:, None])
