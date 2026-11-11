"""Diagnose the WeSpeaker models with an independent front end.

sherpa-onnx feeds the WeSpeaker ONNX models features that give chance-level
speaker verification on this data (results/wespeaker_campplus_vox/eer.csv,
results/cmn_validation.json), while the 3D-Speaker and NeMo models work. Here
the ONNX graphs are run directly with onnxruntime on Kaldi-style log-mel
filterbanks computed in numpy, with several front-end settings, and compared
with sherpa-onnx's own output on the same audio.

Kaldi fbank (as in torchaudio.compliance.kaldi.fbank / kaldi-native-fbank):
25 ms frames, 10 ms shift, snip_edges, DC removal, pre-emphasis 0.97, window,
512-point FFT power spectrum, 80 triangular mel filters (mel = 1127 ln(1 +
f/700)), natural log with float32-epsilon floor, no dither.

Subset: seed 0, first 100 eval speakers, 3x4 enrollment sentences and the six
1 s test clips (same as 09_validate_cmn.py). Output: results/frontend_check.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swk import data, metrics, rules  # noqa: E402

N_SPK = 100


def mel_banks(n_bins=80, n_fft=512, sr=16000, low=20.0, high=0.0):
    nyq = sr / 2
    if high <= 0:
        high = nyq + high
    mel = lambda f: 1127.0 * np.log(1.0 + f / 700.0)  # noqa: E731
    ml, mh = mel(low), mel(high)
    delta = (mh - ml) / (n_bins + 1)
    n_spec = n_fft // 2  # Kaldi drops the Nyquist bin
    fmel = mel(np.arange(n_spec) * sr / n_fft)
    w = np.zeros((n_bins, n_spec))
    for b in range(n_bins):
        left, center, right = ml + b * delta, ml + (b + 1) * delta, ml + (b + 2) * delta
        up = (fmel - left) / (center - left)
        down = (right - fmel) / (right - center)
        w[b] = np.maximum(0.0, np.minimum(up, down))
    return w


def fbank(x: np.ndarray, window="povey", high=0.0, low=20.0, snip_edges=True) -> np.ndarray:
    x = x.astype(np.float64)
    fl, fs = 400, 160
    if not snip_edges:  # Kaldi: reflect-pad so that frames are centred
        n = (x.size + fs // 2) // fs
        pad_l = fl // 2 - fs // 2
        total = (n - 1) * fs + fl
        pad_r = total - x.size - pad_l
        x = np.concatenate([x[:pad_l][::-1], x, x[x.size - pad_r:][::-1] if pad_r > 0 else x[:0]])
    n = 1 + (x.size - fl) // fs
    fr = np.lib.stride_tricks.sliding_window_view(x, fl)[::fs][:n].copy()
    fr -= fr.mean(axis=1, keepdims=True)
    fr[:, 1:] -= 0.97 * fr[:, :-1].copy()
    fr[:, 0] -= 0.97 * fr[:, 0]
    i = np.arange(fl)
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * i / (fl - 1))
    win = hann ** 0.85 if window == "povey" else 0.54 - 0.46 * np.cos(2 * np.pi * i / (fl - 1))
    spec = np.abs(np.fft.rfft(fr * win, 512)) ** 2
    e = spec[:, :256] @ mel_banks(high=high, low=low).T
    return np.log(np.maximum(e, np.finfo(np.float32).eps))


def main() -> None:
    import onnxruntime as ort
    import sherpa_onnx

    meta = data.load_meta().set_index("clip")
    with open(data.RESULTS / "splits" / "seed0.json") as f:
        sp = {int(k): v for k, v in json.load(f).items()}
    ids = [k for k in sorted(sp) if not sp[k]["dev"]][:N_SPK]
    enroll, test, owner = [], [], []
    for i, k in enumerate(ids):
        clips = {c: data.load_clip_int16(meta.at[c, "file"]) for c in sp[k]["enroll"][:12] + sp[k]["test"]}
        e = sp[k]["enroll"]
        enroll.append([data.join([clips[c] for c in e[j * 4:(j + 1) * 4]], 0.1) for j in range(3)])
        for c in sp[k]["test"]:
            test.append(clips[c])
            owner.append(i)
    owner = np.array(owner)
    edges = metrics.hist_edges(-1.0, 1.0, 4000)

    def evaluate(embed):
        enr = np.array([[embed(p) for p in s] for s in enroll])
        q = np.array([embed(p) for p in test])
        _, cos = rules.profile_score_matrix(q, enr)
        tgt = cos[np.arange(owner.size), owner]
        mask = np.ones_like(cos, bool)
        mask[np.arange(owner.size), owner] = False
        eer, _ = metrics.eer_from_hist(np.histogram(tgt, edges)[0].astype(float),
                                       np.histogram(cos[mask], edges)[0].astype(float))
        return float(eer), q

    out = {"n_speakers": len(ids), "n_test": int(owner.size), "runs": []}
    configs = [  # name, waveform scale, window, high_freq, cmn, snip_edges
        ("povey_7600_cmn_int16", 32768.0, "povey", -400.0, True, True),
        ("povey_7600_nocmn_int16", 32768.0, "povey", -400.0, False, True),
        ("hamming_8000_cmn_int16", 32768.0, "hamming", 0.0, True, True),
        ("hamming_8000_nocmn_int16", 32768.0, "hamming", 0.0, False, True),
        ("hamming_8000_cmn_float", 1.0, "hamming", 0.0, True, True),
        ("povey_7600_cmn_int16_nosnip", 32768.0, "povey", -400.0, True, False),
    ]
    for key in ("wespeaker_campplus_vox", "wespeaker_resnet34_vox", "campplus_zhen_adv"):
        path = data.model_path(key)
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(path), num_threads=1, provider="cpu")
        ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

        def sherpa_embed(pcm):
            st = ex.create_stream()
            st.accept_waveform(sample_rate=16000, waveform=pcm.astype(np.float32) / 32768.0)
            st.input_finished()
            v = np.asarray(ex.compute(st), dtype=np.float64)
            return v / np.linalg.norm(v)

        eer_s, q_s = evaluate(sherpa_embed)
        out["runs"].append({"model": key, "frontend": "sherpa-onnx", "eer_cos_1s": eer_s})
        print(key, "sherpa-onnx", round(eer_s, 4), flush=True)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        for name, scale, window, high, cmn, snip in configs:
            def onnx_embed(pcm, scale=scale, window=window, high=high, cmn=cmn, snip=snip):
                f = fbank(pcm.astype(np.float64) / 32768.0 * scale, window=window, high=high, snip_edges=snip)
                if cmn:
                    f = f - f.mean(axis=0, keepdims=True)
                v = sess.run(None, {iname: f[None].astype(np.float32)})[0][0].astype(np.float64)
                return v / np.linalg.norm(v)

            eer, q = evaluate(onnx_embed)
            agree = float(np.median(np.sum(q * q_s, axis=1)))
            out["runs"].append({"model": key, "frontend": name, "eer_cos_1s": eer,
                                "median_cos_to_sherpa_embedding": agree})
            print(key, name, round(eer, 4), "cos-to-sherpa", round(agree, 4), flush=True)
            with open(data.RESULTS / "frontend_check.json", "w") as f:
                json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
