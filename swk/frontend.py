"""WeSpeaker-style front end + onnxruntime, used for the WeSpeaker models.

Why this exists (see README, "WeSpeaker models"): run through sherpa-onnx
2.0.1, the WeSpeaker ONNX files give near-chance speaker verification on this
data. Two causes were measured:

1. sherpa-onnx 2.0.1 bundles onnxruntime 1.28.2. For the WeSpeaker CAM++ graph
   that version (like onnxruntime-node 1.27.0) returns an embedding unrelated
   to the one onnxruntime 1.29.0 / 1.31.1 return on identical input features
   (cosine about -0.1); the ResNet34 graphs agree across versions.
2. The WeSpeaker files carry no feature_normalize_type metadata, so sherpa-onnx
   skips the per-utterance mean subtraction (CMN) WeSpeaker uses, and it uses
   no dither, so the exact digital silence that pads MSWC clips becomes
   log(float eps) frames far outside the training distribution.

This module reproduces WeSpeaker's own front end (wespeaker/dataset/processor.py
compute_fbank: int16-scaled samples, Kaldi fbank with 80 bins, 25/10 ms,
Hamming window, dither; then CMN) in numpy and runs the ONNX graph with the
onnxruntime Python package (1.31.1 here). Dither is a fixed-seed Gaussian of
``dither`` int16 LSB added to the waveform (a microphone noise floor).

The class mimics the small part of sherpa_onnx.SpeakerEmbeddingExtractor that
02_extract.py uses (dim, create_stream, is_ready, compute).
"""

from __future__ import annotations

import numpy as np

SR = 16000


def mel_banks(n_bins: int = 80, n_fft: int = 512, sr: int = SR, low: float = 20.0, high: float = 0.0) -> np.ndarray:
    """Kaldi mel filterbank (mel = 1127 ln(1 + f/700)), (n_bins, n_fft/2); Nyquist bin dropped."""

    nyq = sr / 2
    if high <= 0:
        high = nyq + high
    mel = lambda f: 1127.0 * np.log(1.0 + f / 700.0)  # noqa: E731
    ml, mh = mel(low), mel(high)
    delta = (mh - ml) / (n_bins + 1)
    fmel = mel(np.arange(n_fft // 2) * sr / n_fft)
    w = np.zeros((n_bins, n_fft // 2))
    for b in range(n_bins):
        left, center, right = ml + b * delta, ml + (b + 1) * delta, ml + (b + 2) * delta
        w[b] = np.maximum(0.0, np.minimum((fmel - left) / (center - left), (right - fmel) / (right - center)))
    return w


_BANKS: dict = {}


def fbank(x: np.ndarray, window: str = "hamming", low: float = 20.0, high: float = 0.0) -> np.ndarray:
    """Kaldi log-mel fbank (snip_edges, DC removal, pre-emphasis 0.97, power spectrum, no energy)."""

    key = (low, high)
    if key not in _BANKS:
        _BANKS[key] = mel_banks(low=low, high=high).T
    x = np.asarray(x, dtype=np.float64)
    fl, fs = 400, 160
    n = 1 + (x.size - fl) // fs
    fr = np.lib.stride_tricks.sliding_window_view(x, fl)[::fs][:n].copy()
    fr -= fr.mean(axis=1, keepdims=True)
    fr[:, 1:] -= 0.97 * fr[:, :-1].copy()
    fr[:, 0] -= 0.97 * fr[:, 0]
    i = np.arange(fl)
    if window == "povey":
        win = (0.5 - 0.5 * np.cos(2 * np.pi * i / (fl - 1))) ** 0.85
    else:
        win = 0.54 - 0.46 * np.cos(2 * np.pi * i / (fl - 1))
    spec = np.abs(np.fft.rfft(fr * win, 512)) ** 2
    e = spec[:, :256] @ _BANKS[key]
    return np.log(np.maximum(e, np.finfo(np.float32).eps))


class _Stream:
    def __init__(self) -> None:
        self.parts: list[np.ndarray] = []
        self.seed = None

    def accept_waveform(self, sample_rate: int, waveform) -> None:
        assert sample_rate == SR
        self.parts.append(np.asarray(waveform, dtype=np.float64))

    def input_finished(self) -> None:
        pass


class WeSpeakerOrtExtractor:
    """WeSpeaker front end (int16 scale, Hamming, dither, CMN) + onnxruntime."""

    def __init__(self, model_path: str, dither: float = 1.0, cmn: bool = True) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name
        self.dim = int(self.sess.get_outputs()[0].shape[-1])
        self.dither = float(dither)
        self.cmn = cmn

    def create_stream(self, seed=None) -> _Stream:
        s = _Stream()
        s.seed = seed
        return s

    def is_ready(self, s: _Stream) -> bool:
        return sum(p.size for p in s.parts) >= 400

    def compute(self, s: _Stream) -> np.ndarray:
        x = np.concatenate(s.parts) * 32768.0  # WeSpeaker: waveform * (1 << 15)
        if self.dither > 0:
            x = x + self.dither * np.random.default_rng(s.seed).standard_normal(x.size)
        f = fbank(x, window="hamming")
        if self.cmn:
            f = f - f.mean(axis=0, keepdims=True)
        return self.sess.run(None, {self.input: f[None].astype(np.float32)})[0][0]


class SherpaLikeOrtExtractor(WeSpeakerOrtExtractor):
    """sherpa-onnx's front-end conventions in numpy + onnxruntime (Python package).

    Povey window, 20-7600 Hz (sherpa-onnx defaults); sample scale and CMN taken
    from the model metadata exactly as sherpa-onnx reads them
    (normalize_samples, feature_normalize_type). Only the runtime differs from
    sherpa-onnx 2.0.1, which bundles onnxruntime 1.28.2; on the same features
    this numpy front end gives embeddings with median cosine 0.97 to
    sherpa-onnx's own (results/ort_version_check.json, onnxruntime 1.27.0 rows).
    Optional dither (int16 LSB, fixed seed per utterance) as in the parent class.
    """

    def __init__(self, model_path: str, dither: float = 0.0) -> None:
        super().__init__(model_path, dither=dither, cmn=False)
        md = self.sess.get_modelmeta().custom_metadata_map
        self.float_scale = md.get("normalize_samples", "1") == "1"
        self.cmn = md.get("feature_normalize_type", "") == "global-mean"

    def compute(self, s: _Stream) -> np.ndarray:
        x = np.concatenate(s.parts) * 32768.0
        if self.dither > 0:
            x = x + self.dither * np.random.default_rng(s.seed).standard_normal(x.size)
        if self.float_scale:
            x = x / 32768.0
        f = fbank(x, window="povey", high=-400.0)
        if self.cmn:
            f = f - f.mean(axis=0, keepdims=True)
        return self.sess.run(None, {self.input: f[None].astype(np.float32)})[0][0]


class SherpaLikeOrtcExtractor(SherpaLikeOrtExtractor):
    """As SherpaLikeOrtExtractor, but the graph runs on a given libonnxruntime.so through
    the C API (swk/ortc.py), by default the ONNX Runtime 1.28.2 library that sherpa-onnx
    2.0.1 bundles. Same NumPy features, so a comparison with SherpaLikeOrtExtractor
    (onnxruntime 1.31.1) isolates the runtime version, and a comparison with
    sherpa-onnx itself isolates the front end (same runtime library)."""

    def __init__(self, model_path: str, dither: float = 0.0, lib: str | None = None) -> None:
        super().__init__(model_path, dither=dither)
        from .ortc import SHERPA_ORT, OrtC

        out = self.sess.get_outputs()[0].name
        self.ortc = OrtC(lib or SHERPA_ORT, model_path, self.input, out, self.dim)
        self.sess_py = self.sess
        self.sess = _OrtcSession(self.ortc)


class _OrtcSession:
    def __init__(self, ortc) -> None:
        self.ortc = ortc

    def run(self, _names, feeds: dict):
        (x,) = feeds.values()
        return [self.ortc.run(x)[None]]
