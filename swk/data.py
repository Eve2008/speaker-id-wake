"""Data access for the Persian MSWC test split and the speaker models.

All paths are absolute so the scripts run from any working directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/home/ali/Desktop/Articles/data")
MSWC_CSV = DATA / "mswc" / "test.csv"
AUDIO_DIR = DATA / "mswc" / "audio"
MODELS_DIR = DATA / "models"
RESULTS = ROOT / "results"
CACHE = ROOT / "cache"
SR = 16000


@dataclass(frozen=True)
class ModelInfo:
    key: str
    filename: str
    label: str
    family: str
    training_data: str
    # "sherpa": sherpa_onnx.SpeakerEmbeddingExtractor (as NexaRoom runs it);
    # "wespeaker_ort": swk.frontend.WeSpeakerOrtExtractor (WeSpeaker front end + onnxruntime);
    # "sherpalike_ort": swk.frontend.SherpaLikeOrtExtractor (sherpa-onnx conventions + onnxruntime)
    backend: str = "sherpa"
    # Gaussian noise floor added to the int16 waveform before embedding, in LSB (0 = none).
    dither: float = 0.0


# Six sherpa-onnx speaker-embedding models (k2-fsa release
# "speaker-recongition-models"). SHA-256 sums are in
# data/models/speaker_models_checksum.txt and results/models.json.
MODELS: dict[str, ModelInfo] = {
    m.key: m
    for m in [
        ModelInfo("campplus_zhen_adv", "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
                  "3D-Speaker CAM++ (zh+en, advanced)", "CAM++", "Chinese + English, 16k-common advanced (per file name)"),
        ModelInfo("eres2net_base_zh", "3dspeaker_speech_eres2net_base_200k_sv_zh-cn_16k-common.onnx",
                  "3D-Speaker ERes2Net-base (zh, 200k)", "ERes2Net", "Chinese, 200k speakers, 16k-common (per file name)"),
        ModelInfo("titanet_small", "nemo_en_titanet_small.onnx",
                  "NeMo TitaNet-small (en)", "TitaNet", "English (per file name; NeMo TitaNet-small)"),
        ModelInfo("wespeaker_campplus_vox", "wespeaker_en_voxceleb_CAM++.onnx",
                  "WeSpeaker CAM++ (VoxCeleb)", "CAM++", "VoxCeleb (per file name)"),
        ModelInfo("wespeaker_resnet34_vox", "wespeaker_en_voxceleb_resnet34.onnx",
                  "WeSpeaker ResNet34 (VoxCeleb)", "ResNet34", "VoxCeleb (per file name)"),
        ModelInfo("wespeaker_resnet34_cnc", "wespeaker_zh_cnceleb_resnet34.onnx",
                  "WeSpeaker ResNet34 (CN-Celeb)", "ResNet34", "CN-Celeb (per file name)"),
        # Same WeSpeaker weights, plus the metadata entry feature_normalize_type=global-mean so that
        # sherpa-onnx applies per-utterance feature mean subtraction (CMN), as WeSpeaker's own
        # inference does. Made by experiments/08_make_cmn_models.py.
        ModelInfo("wespeaker_campplus_vox_cmn", "cmn/wespeaker_en_voxceleb_CAM++_cmn.onnx",
                  "WeSpeaker CAM++ (VoxCeleb) + CMN", "CAM++", "VoxCeleb (per file name)"),
        ModelInfo("wespeaker_resnet34_vox_cmn", "cmn/wespeaker_en_voxceleb_resnet34_cmn.onnx",
                  "WeSpeaker ResNet34 (VoxCeleb) + CMN", "ResNet34", "VoxCeleb (per file name)"),
        ModelInfo("wespeaker_resnet34_cnc_cmn", "cmn/wespeaker_zh_cnceleb_resnet34_cmn.onnx",
                  "WeSpeaker ResNet34 (CN-Celeb) + CMN", "ResNet34", "CN-Celeb (per file name)"),
        # Same as-shipped WeSpeaker files, run with WeSpeaker's own front end (int16 scale, Hamming,
        # 1-LSB dither, CMN) and the onnxruntime Python package instead of sherpa-onnx (swk/frontend.py).
        ModelInfo("wespeaker_campplus_vox_wsfe", "wespeaker_en_voxceleb_CAM++.onnx",
                  "WeSpeaker CAM++ (VoxCeleb), WeSpeaker front end", "CAM++", "VoxCeleb (per file name)",
                  "wespeaker_ort", 1.0),
        ModelInfo("wespeaker_resnet34_vox_wsfe", "wespeaker_en_voxceleb_resnet34.onnx",
                  "WeSpeaker ResNet34 (VoxCeleb), WeSpeaker front end", "ResNet34", "VoxCeleb (per file name)",
                  "wespeaker_ort", 1.0),
        # NexaRoom's default model with the same 1-LSB noise floor (sherpa-onnx, as shipped otherwise).
        ModelInfo("campplus_zhen_adv_d1", "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
                  "3D-Speaker CAM++ (zh+en, advanced), 1-LSB dither", "CAM++",
                  "Chinese + English, 16k-common advanced (per file name)", "sherpa", 1.0),
        # NexaRoom's default model with sherpa-onnx's front-end conventions but onnxruntime 1.31.1
        # (sherpa-onnx 2.0.1 bundles 1.28.2, whose AveragePool differs; results/avgpool_check.json).
        ModelInfo("campplus_zhen_adv_ort131", "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
                  "3D-Speaker CAM++ (zh+en, advanced), onnxruntime 1.31.1", "CAM++",
                  "Chinese + English, 16k-common advanced (per file name)", "sherpalike_ort", 0.0),
        ModelInfo("wespeaker_resnet34_cnc_wsfe", "wespeaker_zh_cnceleb_resnet34.onnx",
                  "WeSpeaker ResNet34 (CN-Celeb), WeSpeaker front end", "ResNet34", "CN-Celeb (per file name)",
                  "wespeaker_ort", 1.0),
        # Revision (runtime isolation, experiments/20_runtime_isolation.py):
        # same NumPy front end as campplus_zhen_adv_ort131, graph run by the ONNX Runtime 1.28.2
        # library bundled with sherpa-onnx 2.0.1 (C API via ctypes, swk/ortc.py).
        ModelInfo("campplus_zhen_adv_np128", "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
                  "3D-Speaker CAM++ (zh+en, advanced), NumPy front end, onnxruntime 1.28.2", "CAM++",
                  "Chinese + English, 16k-common advanced (per file name)", "sherpalike_ort128", 0.0),
        # NexaRoom's model file with the 52 AveragePool nodes set to count_include_pad=0 (one byte
        # each, experiments/20_runtime_isolation.py), run by sherpa-onnx 2.0.1 as NexaRoom runs it.
        ModelInfo("campplus_zhen_adv_cip0", "patched/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced_cip0.onnx",
                  "3D-Speaker CAM++ (zh+en, advanced), pooling patched", "CAM++",
                  "Chinese + English, 16k-common advanced (per file name)", "sherpa", 0.0),
        # Matched condition: every model with the same 1-LSB noise floor on a runtime path whose
        # graph output agrees with onnxruntime 1.31.1 (results/runtime_isolation.json).
        ModelInfo("campplus_zhen_adv_cip0_d1", "patched/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced_cip0.onnx",
                  "3D-Speaker CAM++ (zh+en, advanced), pooling patched, 1-LSB dither", "CAM++",
                  "Chinese + English, 16k-common advanced (per file name)", "sherpa", 1.0),
        ModelInfo("eres2net_base_zh_d1", "3dspeaker_speech_eres2net_base_200k_sv_zh-cn_16k-common.onnx",
                  "3D-Speaker ERes2Net-base (zh, 200k), 1-LSB dither", "ERes2Net",
                  "Chinese, 200k speakers, 16k-common (per file name)", "sherpa", 1.0),
        # Revision: two more 3D-Speaker models from the same release, trained on VoxCeleb, to separate
        # training data from architecture (matched condition only; CAM++ graph pooling-patched as above).
        ModelInfo("campplus_vox_cip0_d1", "patched/3dspeaker_speech_campplus_sv_en_voxceleb_16k_cip0.onnx",
                  "3D-Speaker CAM++ (VoxCeleb), pooling patched, 1-LSB dither", "CAM++", "VoxCeleb (per file name)",
                  "sherpa", 1.0),
        ModelInfo("eres2net_vox_d1", "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx",
                  "3D-Speaker ERes2Net (VoxCeleb), 1-LSB dither", "ERes2Net", "VoxCeleb (per file name)",
                  "sherpa", 1.0),
        ModelInfo("titanet_small_d1", "nemo_en_titanet_small.onnx",
                  "NeMo TitaNet-small (en), 1-LSB dither", "TitaNet", "English (per file name; NeMo TitaNet-small)",
                  "sherpa", 1.0),
    ]
}


def model_path(key: str) -> Path:
    return MODELS_DIR / MODELS[key].filename


def load_meta() -> pd.DataFrame:
    """One row per clip, ``clip`` is the stable integer id (row order of test.csv)."""

    df = pd.read_csv(MSWC_CSV)
    df = df.rename(columns={"LINK": "link", "WORD": "word", "SPEAKER": "speaker_hash", "GENDER": "gender"})
    df["clip"] = np.arange(len(df), dtype=np.int64)
    df["src"] = df["link"].str.split("/").str[1].str.replace(".opus", "", regex=False)
    df["file"] = df["link"].str.replace("/", "_", regex=False).str.replace(".opus", ".wav", regex=False)
    df["gender"] = df["gender"].fillna("NAN").astype(str)
    return df[["clip", "word", "speaker_hash", "gender", "src", "file"]]


def load_clip_int16(file: str) -> np.ndarray:
    """Read one clip as int16 PCM (what NexaRoom's pipeline receives)."""

    x, sr = sf.read(AUDIO_DIR / file, dtype="int16")
    if sr != SR:
        raise ValueError(f"{file}: sample rate {sr}")
    if x.ndim != 1:
        x = x[:, 0]
    return x


def trim_int16(x: np.ndarray, rel_db: float = 30.0, margin_s: float = 0.05) -> np.ndarray:
    """Energy trim: keep the span of 25 ms frames within ``rel_db`` of the loudest
    frame, plus ``margin_s`` on each side. Used only in the trimming ablation."""

    f = x.astype(np.float32) / 32768.0
    win, hop = 400, 160
    if f.size < win:
        return x
    frames = np.lib.stride_tricks.sliding_window_view(f, win)[::hop]
    e = 10.0 * np.log10(np.mean(frames * frames, axis=1) + 1e-10)
    active = np.flatnonzero(e > e.max() - rel_db)
    start = max(0, active[0] * hop - int(margin_s * SR))
    end = min(f.size, active[-1] * hop + win + int(margin_s * SR))
    return x[start:end]


def join(clips: list[np.ndarray], gap_s: float) -> np.ndarray:
    if gap_s <= 0 or len(clips) == 1:
        return np.concatenate(clips)
    gap = np.zeros(int(round(gap_s * SR)), dtype=np.int16)
    parts: list[np.ndarray] = []
    for i, c in enumerate(clips):
        if i:
            parts.append(gap)
        parts.append(c)
    return np.concatenate(parts)


def audio_problem_flags(pcm: np.ndarray) -> tuple[float, float, float]:
    """Seconds, RMS (int16 units) and clipped fraction: the three quantities
    NexaRoom's ``audio_problem`` checks."""

    v = pcm.astype(np.float32)
    seconds = pcm.size / SR
    rms = float(np.sqrt(np.mean(v * v))) if pcm.size else 0.0
    clipped = float(np.mean(np.abs(v) >= 32600)) if pcm.size else 0.0
    return seconds, rms, clipped
