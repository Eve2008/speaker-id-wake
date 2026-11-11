"""EER / minDCF from score histograms and speaker-level bootstrap helpers."""

from __future__ import annotations

import numpy as np

N_BOOT = 1000


def hist_edges(lo: float, hi: float, bins: int = 3000) -> np.ndarray:
    return np.linspace(lo, hi, bins + 1)


def per_speaker_hist(scores: np.ndarray, owner: np.ndarray, n_owner: int, edges: np.ndarray) -> np.ndarray:
    """Histogram of ``scores`` per owner index -> (n_owner, bins)."""

    b = np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, edges.size - 2)
    h = np.zeros((n_owner, edges.size - 1), dtype=np.float64)
    np.add.at(h, (owner, b), 1.0)
    return h


def eer_from_hist(ht: np.ndarray, hn: np.ndarray) -> tuple[float, float]:
    """EER and minDCF (P_target=0.01, C_miss=C_fa=1, normalised) from histograms.

    ht, hn: (..., bins). Accept when score >= lower bin edge.
    """

    ct = np.cumsum(ht, axis=-1)
    cn = np.cumsum(hn, axis=-1)
    tt = ct[..., -1:]
    tn = cn[..., -1:]
    # threshold at lower edge of bin b: miss = target mass below b, fa = nontarget mass at/above b
    zeros = np.zeros(ht.shape[:-1] + (1,))
    frr = np.concatenate([zeros, ct[..., :-1]], axis=-1) / tt
    far = 1.0 - np.concatenate([zeros, cn[..., :-1]], axis=-1) / tn
    d = frr - far  # increasing with threshold
    idx = np.argmax(d >= 0, axis=-1)
    idx = np.maximum(idx, 1)
    i0 = idx - 1
    take = lambda a, i: np.take_along_axis(a, i[..., None], axis=-1)[..., 0]  # noqa: E731
    d0, d1 = take(d, i0), take(d, idx)
    w = np.where(d1 - d0 > 0, -d0 / np.maximum(d1 - d0, 1e-12), 0.0)
    e0 = 0.5 * (take(frr, i0) + take(far, i0))
    e1 = 0.5 * (take(frr, idx) + take(far, idx))
    eer = e0 + w * (e1 - e0)
    p = 0.01
    dcf = p * frr + (1 - p) * far
    mindcf = dcf.min(axis=-1) / min(p, 1 - p)
    return eer, mindcf


def boot_weights(n: int, rng: np.random.Generator, n_boot: int = N_BOOT) -> np.ndarray:
    """Multinomial resampling weights (n_boot, n): speaker-level bootstrap."""

    idx = rng.integers(0, n, size=(n_boot, n))
    w = np.zeros((n_boot, n), dtype=np.float32)
    np.add.at(w, (np.repeat(np.arange(n_boot), n), idx.ravel()), 1.0)
    return w


def boot_mean(values: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """Mean and 95 % percentile CI of per-speaker values (NaN = speaker absent)."""

    ok = ~np.isnan(values)
    v = np.where(ok, values, 0.0)
    est = float(v[ok].mean()) if ok.any() else float("nan")
    num = w @ v
    den = w @ ok.astype(np.float32)
    bs = num / np.maximum(den, 1e-12)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, float(lo), float(hi)


def boot_eer(ht: np.ndarray, hn: np.ndarray, w: np.ndarray) -> dict[str, float]:
    """ht, hn: (S, bins) per test speaker. Returns EER/minDCF with 95 % CIs."""

    eer, dcf = eer_from_hist(ht.sum(0), hn.sum(0))
    bt = w @ ht
    bn = w @ hn
    beer, bdcf = eer_from_hist(bt, bn)
    lo, hi = np.percentile(beer, [2.5, 97.5])
    dlo, dhi = np.percentile(bdcf, [2.5, 97.5])
    return {"eer": float(eer), "eer_lo": float(lo), "eer_hi": float(hi),
            "mindcf": float(dcf), "mindcf_lo": float(dlo), "mindcf_hi": float(dhi),
            "n_target": float(ht.sum()), "n_nontarget": float(hn.sum())}
