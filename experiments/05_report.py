"""Final cross-model tables and figures.

Reads results/<config>/*.csv, results/timing*.csv, the diagnostic JSON files
and results/tables/paired_differences.csv (from 19_paired.py). Writes
results/tables/*.csv + *.md and figures/*.png (200 dpi) + figures/*.pdf.
Every number is copied or derived arithmetically from those files; nothing
is typed in by hand. results/tables/INDEX.md lists the tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swk import data  # noqa: E402

TAB = data.RESULTS / "tables"
FIG = data.ROOT / "figures"
R = data.RESULTS

MAIN = ["campplus_zhen_adv", "eres2net_base_zh", "titanet_small", "wespeaker_campplus_vox_wsfe",
        "wespeaker_resnet34_vox_wsfe", "wespeaker_resnet34_cnc_wsfe"]
VARIANTS = ["campplus_zhen_adv_d1", "campplus_zhen_adv_ort131"]
FAILED = ["wespeaker_campplus_vox"]
ALL = MAIN + VARIANTS + FAILED
LABEL = {"campplus_zhen_adv": "3D-Speaker CAM++ (NexaRoom default)",
         "campplus_zhen_adv_d1": "3D-Speaker CAM++, 1-LSB noise floor",
         "campplus_zhen_adv_ort131": "3D-Speaker CAM++, onnxruntime 1.31.1",
         "eres2net_base_zh": "3D-Speaker ERes2Net-base",
         "titanet_small": "NeMo TitaNet-small",
         "wespeaker_campplus_vox_wsfe": "WeSpeaker CAM++ VoxCeleb (WeSpeaker front end)",
         "wespeaker_resnet34_vox_wsfe": "WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end)",
         "wespeaker_resnet34_cnc_wsfe": "WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end)",
         "wespeaker_campplus_vox": "WeSpeaker CAM++ VoxCeleb, sherpa-onnx as shipped",
         # revision configurations (labels only; the tables of this script keep the original set)
         "campplus_zhen_adv_np128": "3D-Speaker CAM++, NumPy front end + onnxruntime 1.28.2",
         "campplus_zhen_adv_cip0": "3D-Speaker CAM++, pooling patched (sherpa-onnx)",
         "campplus_zhen_adv_cip0_d1": "3D-Speaker CAM++, pooling patched + 1-LSB noise floor",
         "eres2net_base_zh_d1": "3D-Speaker ERes2Net-base, 1-LSB noise floor",
         "titanet_small_d1": "NeMo TitaNet-small, 1-LSB noise floor",
         "campplus_vox_cip0_d1": "3D-Speaker CAM++ VoxCeleb, patched + noise floor",
         "eres2net_vox_d1": "3D-Speaker ERes2Net VoxCeleb, noise floor"}
SHORT = {"campplus_zhen_adv": "3D-Speaker CAM++ (NexaRoom)", "eres2net_base_zh": "3D-Speaker ERes2Net-base",
         "titanet_small": "NeMo TitaNet-small", "wespeaker_campplus_vox_wsfe": "WeSpeaker CAM++ VoxCeleb",
         "wespeaker_resnet34_vox_wsfe": "WeSpeaker ResNet34 VoxCeleb",
         "wespeaker_resnet34_cnc_wsfe": "WeSpeaker ResNet34 CN-Celeb",
         "campplus_zhen_adv_d1": "CAM++ + 1-LSB noise floor", "campplus_zhen_adv_ort131": "CAM++, onnxruntime 1.31.1",
         "wespeaker_campplus_vox": "WeSpeaker CAM++ via sherpa-onnx"}
# timing key used for each configuration (d1 = same graph and front end as the default, plus noise)
TIME_KEY = {"campplus_zhen_adv": "campplus_zhen_adv", "campplus_zhen_adv_d1": "campplus_zhen_adv",
            "campplus_zhen_adv_ort131": "campplus_zhen_adv_ort131", "eres2net_base_zh": "eres2net_base_zh",
            "titanet_small": "titanet_small", "wespeaker_campplus_vox_wsfe": "wespeaker_campplus_vox_wsfe",
            "wespeaker_resnet34_vox_wsfe": "wespeaker_resnet34_vox_wsfe",
            "wespeaker_resnet34_cnc_wsfe": "wespeaker_resnet34_cnc_wsfe",
            "wespeaker_campplus_vox": "wespeaker_campplus_vox"}
# categorical slots 1-6 of the validated reference palette, fixed order (light surface)
COLORS = dict(zip(MAIN, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
COLORS.update({"campplus_zhen_adv_d1": "#6250d6", "campplus_zhen_adv_ort131": "#e34948",
               "wespeaker_campplus_vox": "#52514e"})
MARKERS = dict(zip(MAIN, ["o", "s", "^", "D", "v", "P"]))
MARKERS.update({"campplus_zhen_adv_d1": "X", "campplus_zhen_adv_ort131": "*", "wespeaker_campplus_vox": "x"})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DURS = (1, 2, 3)
NS = (2, 3, 5, 10, 20)

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5, "lines.markersize": 5,
    "figure.dpi": 100, "savefig.dpi": 200, "font.family": "DejaVu Sans",
})

INDEX: list[tuple[str, str]] = []


# ---------------------------------------------------------------- helpers
def present(keys: list[str]) -> list[str]:
    return [m for m in keys if (R / m / "eval.done").exists()]


def load(name: str, keys: list[str] | None = None) -> pd.DataFrame:
    frames = []
    for m in present(keys or ALL):
        p = R / m / name
        if p.exists():
            d = pd.read_csv(p)
            d.insert(0, "config", m)
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def ci(r, k="est", lo="lo", hi="hi", d=1, scale=100.0) -> str:
    return f"{scale * r[k]:.{d}f} [{scale * r[lo]:.{d}f}, {scale * r[hi]:.{d}f}]"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def write(df: pd.DataFrame, name: str, desc: str, md: pd.DataFrame | None = None, note: str = "") -> None:
    df.to_csv(TAB / f"{name}.csv", index=False)
    if md is not None:
        (TAB / f"{name}.md").write_text((f"{note}\n\n" if note else "") + md_table(md))
    INDEX.append((name, desc))


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


PLAIN = FuncFormatter(lambda v, _: f"{v:g}")


def plain_log(ax, axis="x") -> None:
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_formatter(PLAIN)
    a.set_minor_formatter(NullFormatter())


LOG_TICKS = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]


def log_y_ticks(ax) -> None:
    lo, hi = ax.get_ylim()
    ax.set_yticks([t for t in LOG_TICKS if lo <= t <= hi])
    plain_log(ax, "y")


def logx(ax) -> None:
    ax.set_xscale("log")
    ax.set_xticks(list(NS))
    ax.set_xticklabels([str(n) for n in NS])
    ax.minorticks_off()


def get(mm: pd.DataFrame, dur, n, rule, metric):
    x = mm[(mm.dur == dur) & (mm.N == n) & (mm.rule == rule) & (mm.metric == metric)]
    return x.iloc[0] if len(x) else None


# ---------------------------------------------------------------- tables
def t_data(mainr, eer) -> None:
    ss = json.loads((R / "split_summary.json").read_text())
    sp = pd.read_csv(R / "speakers.csv")
    zp = json.loads((R / "zero_padding.json").read_text())
    cs = pd.read_csv(R / "clip_stats.csv")
    cnt = json.loads((R / "campplus_zhen_adv" / "counts.json").read_text())
    meta = pd.read_csv(data.MSWC_CSV)
    e0 = eer[(eer.config == "campplus_zhen_adv") & (eer.score == "cos")]
    a = mainr[(mainr.config == "campplus_zhen_adv") & (mainr.rule == "closed_set") & (mainr.metric == "acc_cos")]
    rows = [
        ("MSWC Persian test split: clips", len(meta)),
        ("MSWC Persian test split: speakers", int(meta.SPEAKER.nunique())),
        ("speaker pool (>= 18 clips, splittable under all seeds)", ss["n_speakers"]),
        ("pool: male / female / other / missing gender label",
         f"{ss['gender']['MALE']} / {ss['gender']['FEMALE']} / {ss['gender']['OTHER']} / {ss['gender']['NAN']}"),
        ("pool: clips per speaker, median [IQR]",
         f"{sp.n_clips.median():.0f} [{sp.n_clips.quantile(.25):.0f}, {sp.n_clips.quantile(.75):.0f}]"),
        ("pool: source recordings per speaker, median [IQR]",
         f"{sp.n_src.median():.0f} [{sp.n_src.quantile(.25):.0f}, {sp.n_src.quantile(.75):.0f}]"),
        ("pool: clips per speaker, min / max", f"{sp.n_clips.min()} / {sp.n_clips.max()}"),
        ("seeds", len(ss["seeds"])),
        ("dev speakers per seed", ss["seeds"]["0"]["dev"]),
        ("eval speakers per seed", ss["seeds"]["0"]["n"] - ss["seeds"]["0"]["dev"]),
        ("speakers that are eval in >= 1 seed (bootstrap units)", int(a.n_spk.iloc[0])),
        ("eval speakers with >= 40 enrollment clips (enrollment ablation), per seed",
         " / ".join(str(ss["seeds"][s]["rich_enroll_5x8"]) for s in sorted(ss["seeds"]))),
        ("enrollment: sentences x clips (main), gap between clips (s)", f"3 x 4, {ss['enroll_gap_s']}"),
        ("test utterances per speaker and seed: 1 s / 2 s / 3 s", "6 / 3 / 2"),
        ("households per test utterance (member and stranger, each)", 100),
        ("household sizes N", ", ".join(map(str, NS))),
        ("distinct words among test clips", ss["words_in_test_clips"]),
        ("embedded utterances per configuration: main / ablation", f"{ss['n_specs']['main']} / {ss['n_specs']['ablation']}"),
    ]
    for d in DURS:
        r = e0[e0.dur == d].iloc[0]
        rows.append((f"EER trials pooled over seeds, {d} s: target / non-target",
                     f"{int(r.n_target):,} / {int(r.n_nontarget):,}"))
    rows += [
        ("test clips analysed for padding (seed 0)", zp["n_clips"]),
        ("share of samples exactly 0 per clip, median [IQR]",
         f"{zp['zero_share_median']:.3f} [{zp['zero_share_p25']:.3f}, {zp['zero_share_p75']:.3f}]"),
        ("clips with a run of exact zeros >= 0.1 s", f"{zp['share_clips_zero_run_ge_0.1s']:.3f}"),
        ("longest zero run per clip, median (s)", f"{zp['longest_zero_run_s_median']:.3f}"),
        ("25 ms frames entirely zero, median share per clip", f"{zp['frames_at_log_floor_share_median']:.3f}"),
        ("log-mel floor of an all-zero frame (int16 scale, nats)", f"{zp['log_floor']:.2f}"),
        ("gap between loudest 10% frames and the floor, median (nats)", f"{zp['gap_nats_median']:.1f}"),
    ]
    for _, r in cs.iterrows():
        rows.append((f"active speech per clip, median [IQR] (s): {r.group} (n={r.n_clips})",
                     f"{r.active_s_median:.3f} [{r.active_s_p25:.3f}, {r.active_s_p75:.3f}]"))
    rows += [
        ("1 s test clips failing NexaRoom's audio check (RMS < 80 etc.), per seed min-max",
         f"{min(cnt['too_short_1s']):.4f}-{max(cnt['too_short_1s']):.4f}"),
        ("enrollment sentences failing NexaRoom's audio check, per seed min-max",
         f"{min(cnt['enroll_audio_problem']):.4f}-{max(cnt['enroll_audio_problem']):.4f}"),
    ]
    df = pd.DataFrame(rows, columns=["quantity", "value"])
    write(df, "data_summary", "Data, pool, protocol and clip statistics", df)


def t_models(timing) -> None:
    mj = json.loads((R / "models.json").read_text())
    info = {m["key"]: m for m in mj["models"]}
    backend = {"sherpa": "sherpa-onnx 2.0.1 (onnxruntime 1.28.2 bundled)",
               "wespeaker_ort": "WeSpeaker front end (numpy) + onnxruntime 1.31.1",
               "sherpalike_ort": "sherpa-onnx conventions (numpy) + onnxruntime 1.31.1"}
    rows = []
    for m in ALL:
        if m not in info:
            continue
        x = info[m]
        t1 = timing[(timing.model == TIME_KEY[m]) & timing.utterance.str.startswith("1 s")]
        rows.append({"config": m, "model": LABEL[m], "family": x["family"], "training data (as named)":
                     x["training_data_as_named"], "file": x["file"], "file MB": round(x["bytes"] / 1e6, 1),
                     "dim": x["embedding_dim"], "runtime path": backend[x["backend"]],
                     "noise floor (LSB)": x["dither_lsb"], "sha256 matches release": x["sha256_matches_release_checksum"],
                     "CPU ms, 1 s utterance": round(float(t1.cpu_ms_median.iloc[0]), 1) if len(t1) else ""})
    df = pd.DataFrame(rows)
    write(df, "models", "Configurations: model files, runtime path, size, dimension", df)


def t_eer(eer) -> None:
    df = eer.copy()
    df["model"] = df.config.map(LABEL)
    rows = []
    for m in present(ALL):
        for d in DURS:
            r = {"model": LABEL[m], "test": f"{d} s"}
            for sc, lab in (("cos", "cosine"), ("blend", "NexaRoom blend score"), ("asnorm", "AS-norm")):
                x = df[(df.config == m) & (df.dur == d) & (df.score == sc)].iloc[0]
                r[f"EER {lab} %"] = ci(x, "eer", "eer_lo", "eer_hi", 2)
            x = df[(df.config == m) & (df.dur == d) & (df.score == "cos")].iloc[0]
            r["minDCF(0.01) cosine"] = ci(x, "mindcf", "mindcf_lo", "mindcf_hi", 3, 1.0)
            x = df[(df.config == m) & (df.dur == d) & (df.score == "asnorm")].iloc[0]
            r["minDCF(0.01) AS-norm"] = ci(x, "mindcf", "mindcf_lo", "mindcf_hi", 3, 1.0)
            x = df[(df.config == m) & (df.dur == d) & (df.score == "cos")].iloc[0]
            r["EER cosine, SD over seeds (pp)"] = f"{100 * x.seed_sd:.2f}"
            rows.append(r)
    write(df, "eer_all", "EER and minDCF per configuration, duration and scoring (95% CI)", pd.DataFrame(rows))
    # score distributions (why fixed thresholds do not transfer)
    rows = []
    for m in present(ALL):
        for d in DURS:
            x = df[(df.config == m) & (df.dur == d) & (df.score == "cos")].iloc[0]
            rows.append({"model": LABEL[m], "test": f"{d} s", "target cos q05": x["target_q0.05"],
                         "target cos median": x["target_q0.5"], "non-target cos median": x["nontarget_q0.5"],
                         "non-target cos q95": x["nontarget_q0.95"], "non-target cos q99": x["nontarget_q0.99"],
                         "non-target cos q99.9": x["nontarget_q0.999"]})
    s = pd.DataFrame(rows).round(4)
    write(s, "score_distributions", "Cosine score quantiles (target / non-target), pooled eval trials", s,
          "Bin upper edges of 0.0005-wide histograms, pooled over seeds and eval speakers.")


def t_closed(mainr) -> None:
    cs = mainr[mainr.rule == "closed_set"].copy()
    rows = []
    for m in present(ALL):
        for n in NS:
            r = {"model": LABEL[m], "N": n}
            for d in DURS:
                r[f"{d} s"] = ci(get(cs[cs.config == m], d, n, "closed_set", "acc_cos"))
            rows.append(r)
    write(cs, "closed_set_all", "Closed-set identification accuracy, all N, durations and scores (95% CI)",
          pd.DataFrame(rows), "Closed-set accuracy (%), cosine to centroid, eval speakers, 5 seeds, 95% CI.")
    rows = []
    for m in present(ALL):
        for d in DURS:
            r = {"model": LABEL[m], "test": f"{d} s"}
            for sc, lab in (("acc_cos", "cosine"), ("acc_blend", "NexaRoom blend"), ("acc_asnorm", "AS-norm")):
                for n in (5, 20):
                    r[f"{lab}, N={n} %"] = ci(get(cs[cs.config == m], d, n, "closed_set", sc))
            rows.append(r)
    x = pd.DataFrame(rows)
    write(x, "closed_set_by_score", "Closed-set accuracy by scoring function, N = 5 and 20", x)


def t_rules(mainr) -> None:
    keep = ["nexaroom_default", "cos@0.50", "ladder_a", "ladder_b", "ladder_c", "ladder_d", "ladder_e",
            "cos@dev_matched"]
    nx = mainr[mainr.rule.isin(keep)]
    rows = []
    for m in present(ALL):
        mm = nx[nx.config == m]
        for d in DURS:
            for rule, lab in (("nexaroom_default", "NexaRoom rule (defaults)"), ("cos@0.50", "cosine >= 0.50"),
                              ("cos@dev_matched", "cosine, threshold matched to NexaRoom's dev FA")):
                r = {"model": LABEL[m], "test": f"{d} s", "rule": lab}
                for k, l2, dd in (("member_correct", "member named correctly %", 1),
                                  ("member_wrong", "wrong member named %", 2),
                                  ("member_unknown", "member -> unknown %", 1),
                                  ("member_ambiguous", "member -> ambiguous %", 1),
                                  ("member_too_short", "member -> too_short %", 2),
                                  ("stranger_fa", "stranger accepted %", 2),
                                  ("stranger_ambiguous", "stranger -> ambiguous %", 2)):
                    x = get(mm, d, 5, rule, k)
                    r[l2] = "" if x is None else (ci(x, d=dd) if k in ("member_correct", "stranger_fa")
                                                  else f"{100 * x.est:.{dd}f}")
                rows.append(r)
    write(nx, "rules_all", "Decision-rule outcomes (NexaRoom default, cosine 0.50, ladder), all N", pd.DataFrame(rows),
          "Open set, household of N = 5. Member wakes: true speaker enrolled; stranger wakes: not enrolled. "
          "Rows sum to 100% over correct + wrong + unknown + ambiguous + too_short for members.")
    # NexaRoom default vs N
    rows = []
    for m in present(ALL):
        mm = nx[(nx.config == m) & (nx.rule == "nexaroom_default")]
        for d in DURS:
            r = {"model": LABEL[m], "test": f"{d} s"}
            for n in NS:
                c, w, f = (get(mm, d, n, "nexaroom_default", k) for k in ("member_correct", "member_wrong",
                                                                          "stranger_fa"))
                r[f"N={n}: correct / wrong / stranger %"] = f"{100 * c.est:.1f} / {100 * w.est:.2f} / {100 * f.est:.2f}"
            rows.append(r)
    x = pd.DataFrame(rows)
    write(x, "nexaroom_default_vs_N", "NexaRoom default rule vs household size", x)
    # ladder (decomposition of NexaRoom's rule)
    steps = {"ladder_a": "a: cosine to centroid >= 0.50", "ladder_b": "b: blend score >= 0.50",
             "ladder_c": "c: blend >= calibrated per-profile threshold",
             "ladder_d": "d: c + 0.05 penalty for audio < 2 s",
             "ladder_e": "e: d + runner-up margin 0.06 (= full NexaRoom rule)"}
    rows = []
    for m in present(ALL):
        mm = nx[nx.config == m]
        for d in DURS:
            for st, lab in steps.items():
                c, w, f = (get(mm, d, 5, st, k) for k in ("member_correct", "member_wrong", "stranger_fa"))
                rows.append({"model": LABEL[m], "test": f"{d} s", "step": lab, "member correct %": round(100 * c.est, 2),
                             "wrong member %": round(100 * w.est, 3), "stranger accepted %": round(100 * f.est, 2)})
    x = pd.DataFrame(rows)
    write(x, "rule_ladder_N5", "Step-by-step decomposition of NexaRoom's rule, N = 5", x,
          "Each step adds one element of NexaRoom's rule (all at N = 5, eval speakers, 5 seeds). "
          "Step e equals nexaroom_default exactly (checked in 05_report.py).")
    for m in present(ALL):
        a = nx[(nx.config == m) & (nx.rule == "ladder_e")].set_index(["dur", "N", "metric"]).est
        b = nx[(nx.config == m) & (nx.rule == "nexaroom_default")].set_index(["dur", "N", "metric"]).est
        assert float((a - b.reindex(a.index)).abs().max()) < 1e-12, m


def t_open_tuned(mainr) -> None:
    th = load("tuned_thresholds.csv")
    tun = mainr[mainr.rule.str.contains("@dev_fa|@pooled_fa")].copy()
    rows = []
    for m in present(ALL):
        mm = tun[tun.config == m]
        for tgt in (1, 5):
            for d in DURS:
                r = {"model": LABEL[m], "dev FA target": f"{tgt}%", "test": f"{d} s"}
                for fam, lab in (("cos", "cosine"), ("nexaroom", "NexaRoom rule"), ("asnorm", "AS-norm")):
                    c = get(mm, d, 5, f"{fam}@dev_fa{tgt}", "member_correct")
                    f = get(mm, d, 5, f"{fam}@dev_fa{tgt}", "stranger_fa")
                    w = get(mm, d, 5, f"{fam}@dev_fa{tgt}", "member_wrong")
                    r[f"{lab}: correct %"] = ci(c)
                    r[f"{lab}: wrong %"] = f"{100 * w.est:.2f}"
                    r[f"{lab}: stranger FA on eval %"] = ci(f, d=2)
                rows.append(r)
    write(tun, "open_set_tuned_all", "Open set with dev-tuned thresholds (1%/5% stranger FA), all N",
          pd.DataFrame(rows), "Thresholds tuned on dev speakers (per seed, duration and N) for the stated stranger "
          "acceptance, applied to eval speakers. N = 5.")
    rows = []
    for m in present(ALL):
        mm = tun[tun.config == m]
        for tgt in (1, 5):
            for d in DURS:
                r = {"model": LABEL[m], "dev FA target": f"{tgt}%", "test": f"{d} s"}
                for fam, lab in (("cos", "cosine"), ("nexaroom", "NexaRoom rule"), ("asnorm", "AS-norm")):
                    c = get(mm, d, 5, f"{fam}@pooled_fa{tgt}", "member_correct")
                    f = get(mm, d, 5, f"{fam}@pooled_fa{tgt}", "stranger_fa")
                    r[f"{lab}: correct %"] = ci(c)
                    r[f"{lab}: stranger FA on eval %"] = ci(f, d=2)
                rows.append(r)
    x = pd.DataFrame(rows)
    write(x, "open_set_pooled_threshold_N5", "One threshold for all durations (dev-tuned), N = 5", x,
          "One threshold per seed, tuned on dev trials pooled over 1, 2 and 3 s, applied to each duration.")
    # tuned threshold values
    g = th[th.N.astype(str).isin(["5"])].groupby(["config", "dur", "label"]).threshold.agg(["mean", "min", "max"])
    g = g.reset_index()
    g["model"] = g.config.map(LABEL)
    write(g, "tuned_threshold_values_N5", "Dev-tuned threshold values (mean, min, max over seeds), N = 5",
          g[["model", "dur", "label", "mean", "min", "max"]].round(3))


def t_gender() -> None:
    ge = load("gender_eer.csv")
    gc = load("gender.csv")
    rows = []
    for m in present(ALL):
        for d in DURS:
            s = ge[(ge.config == m) & (ge.dur == d) & (ge.nontarget_gender == "same")].iloc[0]
            c = ge[(ge.config == m) & (ge.dur == d) & (ge.nontarget_gender == "cross")].iloc[0]
            rows.append({"model": LABEL[m], "test": f"{d} s", "EER same-gender non-targets %":
                         ci(s, "eer", "eer_lo", "eer_hi", 2), "EER cross-gender non-targets %":
                         ci(c, "eer", "eer_lo", "eer_hi", 2), "ratio same/cross": f"{s.eer / c.eer:.2f}",
                         "labelled speakers": int(s.n_spk)})
    write(ge, "gender_eer_all", "EER with same- vs cross-gender non-targets", pd.DataFrame(rows),
          "Speakers with a male or female label only (non-target trials restricted to labelled speakers).")
    rows = []
    for m in present(ALL):
        mm = gc[(gc.config == m) & (gc.target_gender == "ALL")]
        for d in DURS:
            r = {"model": LABEL[m], "test": f"{d} s"}
            for rel, lab in (("same", "same-gender household"), ("any", "random household"),
                             ("cross", "cross-gender household")):
                x = mm[(mm.dur == d) & (mm.household_gender == rel) & (mm.metric == "acc_cos")].iloc[0]
                r[f"closed-set acc. N=5, {lab} %"] = ci(x)
            for rel in ("same", "cross"):
                x = mm[(mm.dur == d) & (mm.household_gender == rel) & (mm.metric == "nexaroom_stranger_fa")].iloc[0]
                r[f"NexaRoom rule stranger FA, {rel}-gender %"] = ci(x, d=2)
            rows.append(r)
    write(gc, "gender_all", "Closed-set accuracy and NexaRoom stranger FA by household gender composition",
          pd.DataFrame(rows), "N = 5. Same-gender household: the 4 competitors (members) or 5 enrolled "
          "speakers (strangers) share the test speaker's gender label; cross: all have the other label.")
    rows = []
    for m in present(ALL):
        mm = gc[(gc.config == m) & (gc.household_gender == "any")]
        for d in DURS:
            r = {"model": LABEL[m], "test": f"{d} s"}
            for tg in ("MALE", "FEMALE"):
                x = mm[(mm.dur == d) & (mm.target_gender == tg) & (mm.metric == "acc_cos")].iloc[0]
                r[f"closed-set acc. N=5, {tg.lower()} test speakers %"] = ci(x)
                r[f"n {tg.lower()}"] = int(x.n_spk)
            rows.append(r)
    x = pd.DataFrame(rows)
    write(x, "gender_by_target", "Closed-set accuracy by the test speaker's gender label, N = 5", x)


def t_timing(timing) -> pd.DataFrame:
    host = json.loads((R / "timing_host.json").read_text()) if (R / "timing_host.json").exists() else {}
    env = json.loads((R / "timing_env.json").read_text())
    rep = R / "timing_rep2.csv"
    t2 = pd.read_csv(rep) if rep.exists() else None
    rows = []
    for _, r in timing.iterrows():
        x = {"timing key": r.model, "utterance": r.utterance, "audio s": r.audio_s,
             "wall ms median": round(r.wall_ms_median, 1), "wall ms p10": round(r.wall_ms_p10, 1),
             "wall ms p90": round(r.wall_ms_p90, 1), "CPU ms median": round(r.cpu_ms_median, 1),
             "real-time factor": round(r.rtf_median, 4)}
        if t2 is not None:
            y = t2[(t2.model == r.model) & (t2.utterance == r.utterance)]
            if len(y):
                x["repeat run: CPU ms median"] = round(float(y.cpu_ms_median.iloc[0]), 1)
                x["repeat vs first, %"] = round(100 * (float(y.cpu_ms_median.iloc[0]) / r.cpu_ms_median - 1), 1)
        x.update({"model load s": round(r.load_s, 2), "RSS increase MB": round(r.rss_increase_mb, 1),
                  "NexaRoom scoring, 5 profiles, ms": round(r.scoring_ms_5_profiles, 3), "dim": int(r.embedding_dim),
                  "file MB": round(r.file_mb, 1)})
        rows.append(x)
    df = pd.DataFrame(rows)
    note = (f"Single-thread time per utterance (create_stream + accept_waveform + compute), "
            f"{env['repeats']} timed runs after {env['warmup']} warm-up runs, one pinned vCPU, num_threads = 1. "
            f"Host: {host.get('model_name', env['cpu'])}, {host.get('hypervisor_vendor', '?')} guest with "
            f"{host.get('vcpus', '?')} vCPUs; load average before/after the first run "
            f"{env['loadavg_before'][0]:.1f}/{env['loadavg_after'][0]:.1f}. Raspberry Pi / ARM timing NOT measured.")
    write(df, "timing", "Single-thread CPU time per utterance and model (x86 VM only)", df, note)
    if "repeat vs first, %" in df:
        d = df["repeat vs first, %"].abs()
        rr = pd.DataFrame([{"rows compared": int(d.notna().sum()), "median abs. difference, repeat vs first %": round(float(d.median()), 1),
                            "max abs. difference, repeat vs first %": round(float(d.max()), 1),
                            "row with max": df.loc[d.idxmax(), "timing key"] + ", " + df.loc[d.idxmax(), "utterance"]}])
        write(rr, "timing_repeatability", "Run-to-run spread of the CPU-time medians (two full runs)", rr)
    # compact
    rows = []
    for m in present(ALL):
        k = TIME_KEY[m]
        g = timing[timing.model == k].set_index("utterance")
        r = {"model": LABEL[m], "timing key": k}
        for u, lab in (("1 s (1 clip)", "1 s"), ("2 s (2 clips)", "2 s"), ("3 s (3 clips)", "3 s"),
                       ("4.3 s enrollment sentence (4 clips)", "4.3 s sentence"),
                       ("8.7 s enrollment sentence (8 clips)", "8.7 s sentence")):
            r[f"CPU ms, {lab}"] = round(float(g.loc[u, "cpu_ms_median"]), 1)
        r["file MB"] = round(float(g.file_mb.iloc[0]), 1)
        r["dim"] = int(g.embedding_dim.iloc[0])
        rows.append(r)
    x = pd.DataFrame(rows)
    write(x, "timing_compact", "CPU ms per utterance for each evaluated configuration", x,
          "The 1-LSB noise-floor configuration uses the default model's graph and front end; its timing row "
          "is the default model's (adding the noise was not timed separately).")
    return df


def t_ablations() -> None:
    m = "campplus_zhen_adv"
    ab = pd.read_csv(R / m / "ablations.csv")
    ae = pd.read_csv(R / m / "ablations_eer.csv")
    rows = []
    for cfg in ("3x4", "5x4", "3x8", "5x8"):
        for d in DURS:
            g = ab[(ab.ablation == "enroll") & (ab.config == cfg) & (ab.dur == d)].set_index("metric")
            e = ae[(ae.ablation == "enroll") & (ae.config == cfg) & (ae.dur == d)].iloc[0]
            rows.append({"enrollment (sentences x clips)": cfg, "test": f"{d} s",
                         "EER cosine %": ci(e, "eer", "eer_lo", "eer_hi", 2),
                         "closed-set acc. N=5 %": ci(g.loc["acc_cos"]),
                         "NexaRoom rule: correct %": ci(g.loc["nexaroom_member_correct"]),
                         "NexaRoom rule: wrong %": f"{100 * g.loc['nexaroom_member_wrong'].est:.2f}",
                         "NexaRoom rule: stranger FA %": ci(g.loc["nexaroom_stranger_fa"], d=2),
                         "mean calibrated threshold": f"{g.loc['calibrated_threshold'].est:.3f}",
                         "speakers": int(g.loc["acc_cos"].n_spk)})
    x = pd.DataFrame(rows)
    write(pd.concat([ab.assign(kind="metric"), ae.assign(kind="eer")]), "ablation_enrollment",
          "Enrollment size ablation (NexaRoom default model)", x[x.columns],
          "3D-Speaker CAM++ (NexaRoom default), eval speakers with >= 40 enrollment clips, N = 5.")
    rows = []
    for cfg in ("enroll_raw+test_raw", "enroll_trim+test_raw", "enroll_raw+test_trim", "enroll_trim+test_trim"):
        g = ab[(ab.ablation == "trim") & (ab.config == cfg)].set_index("metric")
        e = ae[(ae.ablation == "trim") & (ae.config == cfg)].iloc[0]
        rows.append({"condition": cfg, "EER cosine 1 s %": ci(e, "eer", "eer_lo", "eer_hi", 2),
                     "closed-set acc. N=5 %": ci(g.loc["acc_cos"])})
    x = pd.DataFrame(rows)
    write(x, "ablation_trim", "Silence trimming of enrollment / test audio (NexaRoom default model, 1 s)", x)
    fw = pd.read_csv(R / m / "fixed_word.csv")
    fe = pd.read_csv(R / m / "fixed_word_eer.csv")
    cs = pd.read_csv(R / "clip_stats.csv").set_index("group")
    rows = []
    for w in fe.word.unique():
        r = {"word": w, "active speech median s": cs.loc[f"fixed_word:{w}", "active_s_median"]}
        for cond in ("fixed_word", "random_word"):
            e = fe[(fe.word == w) & (fe.condition == cond)].iloc[0]
            g = fw[(fw.word == w) & (fw.condition == cond)].set_index("metric")
            r[f"{cond}: EER %"] = ci(e, "eer", "eer_lo", "eer_hi", 2)
            r[f"{cond}: closed-set acc. N=5 %"] = ci(g.loc["acc_cos"])
            r[f"{cond}: NexaRoom correct %"] = ci(g.loc["nexaroom_member_correct"])
            r["speakers"] = int(e.n_spk)
        rows.append(r)
    x = pd.DataFrame(rows)
    write(x, "fixed_word", "Fixed wake word vs random word, same speakers (NexaRoom default model, 1 s)", x,
          "Test = one 1 s clip of the given word vs a random-word 1 s test clip of the same speakers.")


def t_diagnostics() -> None:
    rows = []

    def add(model, cfg, eer, acc, src):
        rows.append({"model": model, "configuration": cfg, "EER cosine 1 s %": round(100 * eer, 1),
                     "closed-set acc. N=5 1 s %": "" if acc is None else round(100 * acc, 1), "source": src})

    for k in ("wespeaker_campplus_vox", "wespeaker_resnet34_vox", "wespeaker_resnet34_cnc", "campplus_zhen_adv",
              "eres2net_base_zh", "titanet_small"):
        p = R / "dither_check" / f"{k}.json"
        if p.exists():
            j = json.loads(p.read_text())
            add(k, "sherpa-onnx as shipped", j["raw"]["eer_cos_1s"], j["raw"]["closed_set_acc_N5_1s"], p.name)
            add(k, "sherpa-onnx + 1-LSB noise floor", j["dither_1lsb"]["eer_cos_1s"],
                j["dither_1lsb"]["closed_set_acc_N5_1s"], p.name)
        p = R / "dither_check" / f"{k}_cmn.json"
        if p.exists():
            j = json.loads(p.read_text())
            add(k, "sherpa-onnx + CMN metadata", j["raw"]["eer_cos_1s"], j["raw"]["closed_set_acc_N5_1s"], p.name)
            add(k, "sherpa-onnx + CMN metadata + 1-LSB noise floor", j["dither_1lsb"]["eer_cos_1s"],
                j["dither_1lsb"]["closed_set_acc_N5_1s"], p.name)
        p = R / "wespeaker_ort_check" / f"{k}.json"
        if p.exists():
            j = json.loads(p.read_text())
            for kk, lab in (("no_dither", "WeSpeaker front end, onnxruntime 1.31.1, no noise floor"),
                            ("no_cmn", "WeSpeaker front end, onnxruntime 1.31.1, no CMN"),
                            ("wespeaker_frontend", "WeSpeaker front end, onnxruntime 1.31.1 (noise floor + CMN)")):
                add(k, lab, j[kk]["eer_cos_1s"], j[kk]["closed_set_acc_N5_1s"], p.name)
    p = R / "runtime_effect_subset" / "campplus_zhen_adv.json"
    if p.exists():
        j = json.loads(p.read_text())
        for kk, lab in (("sherpa_onnx_2.0.1", "sherpa-onnx 2.0.1 (onnxruntime 1.28.2), own front end"),
                        ("onnxruntime_node_1.27.0", "onnxruntime-node 1.27.0, numpy front end"),
                        ("onnxruntime_1.31.1_python", "onnxruntime 1.31.1, numpy front end")):
            add("campplus_zhen_adv", lab, j[kk]["eer_cos_1s"], j[kk]["closed_set_acc_N5_1s"], "runtime_effect_subset")
    df = pd.DataFrame(rows)
    write(df, "diagnostics_subset", "Front-end / runtime diagnostics on a 100-speaker subset (seed 0, 1 s)", df,
          "100 eval speakers of seed 0, 600 test clips of 1 s, cosine scoring.")
    j = json.loads((R / "ort_version_check.json").read_text())
    rows = []
    for k, v in j["models"].items():
        rows.append({"model": k, "ort 1.27.0 vs 1.29.0": round(v["node_ort_1.27.0 vs node_ort_1.29.0"], 3),
                     "ort 1.29.0 vs 1.31.1": round(v["node_ort_1.29.0 vs node_ort_1.31.1"], 3),
                     "ort 1.27.0 vs sherpa-onnx 2.0.1": round(v["node_ort_1.27.0 vs sherpa_onnx"], 3),
                     "ort 1.31.1 vs sherpa-onnx 2.0.1": round(v["node_ort_1.31.1 vs sherpa_onnx"], 3)})
    df = pd.DataFrame(rows)
    write(df, "ort_version_agreement", "Embedding agreement across onnxruntime versions (20 clips)", df,
          f"Cosine between embeddings of the same {j['n_clips']} clips from identical input features "
          "(sherpa-onnx column: sherpa-onnx's own front end).")


def t_paired() -> None:
    p = TAB / "paired_differences.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    INDEX.append(("paired_differences", "Paired (same-speaker) bootstrap differences between configurations and rules"))
    keep = df[(df.comparison == "config") & (df.B == "campplus_zhen_adv")
              & df.metric.isin(["EER (cos)", "closed-set acc. N=5 (cosine)", "closed-set acc. N=20 (cosine)",
                                "NexaRoom rule: member named correctly", "NexaRoom rule: stranger accepted",
                                "AS-norm @ dev 1% FA: member named correctly"])].copy()
    keep["A"] = keep.A.map(lambda a: LABEL.get(a, a))
    keep["diff pp [95% CI]"] = keep.apply(lambda r: f"{100 * r.diff_A_minus_B:+.2f} [{100 * r.lo:+.2f}, "
                                                    f"{100 * r.hi:+.2f}]", axis=1)
    keep["p"] = keep.p_boot.map(lambda v: "<0.001" if v < 0.001 else f"{v:.3f}")
    x = keep[["A", "metric", "dur", "diff pp [95% CI]", "p"]].rename(columns={"A": "configuration minus default"})
    write(keep, "paired_vs_default", "Paired differences of each configuration vs the NexaRoom default model", x,
          "A minus 3D-Speaker CAM++ (NexaRoom default, sherpa-onnx), same eval speakers, 1,000 paired "
          "speaker-bootstrap replicates; percentage points.")


def t_concentration() -> None:
    conc = []
    for m in present(ALL):
        ps = pd.read_csv(R / m / "per_speaker_main.csv.gz")
        ps = ps[(ps.rule == "closed_set") & (ps.metric == "acc_cos") & (ps.N == 5)]
        for d in DURS:
            v = ps[ps.dur == d].groupby("spk").value.mean().to_numpy()
            err = np.sort(1 - v)[::-1]
            top10 = err[: max(1, int(round(0.1 * err.size)))].sum() / max(err.sum(), 1e-12)
            conc.append({"model": LABEL[m], "test": f"{d} s", "speakers": int(v.size),
                         "median speaker acc. %": round(100 * float(np.median(v)), 1),
                         "speakers with acc. < 80% (%)": round(100 * float(np.mean(v < 0.8)), 1),
                         "speakers with acc. = 100% (%)": round(100 * float(np.mean(v >= 0.99999)), 1),
                         "share of errors from worst 10% of speakers (%)": round(100 * float(top10), 1)})
    x = pd.DataFrame(conc)
    write(x, "error_concentration_N5", "How closed-set errors concentrate on few speakers (N = 5, cosine)", x)


def t_headline(mainr, eer, timing) -> None:
    rows = []
    for m in present(ALL):
        mm = mainr[mainr.config == m]
        for d in DURS:
            e = eer[(eer.config == m) & (eer.dur == d)]
            r = {"model": LABEL[m], "test": f"{d} s",
                 "EER cosine %": f"{100 * e[e.score == 'cos'].eer.iloc[0]:.2f}",
                 "EER AS-norm %": f"{100 * e[e.score == 'asnorm'].eer.iloc[0]:.2f}",
                 "closed-set N=5 %": f"{100 * get(mm, d, 5, 'closed_set', 'acc_cos').est:.1f}",
                 "closed-set N=20 %": f"{100 * get(mm, d, 20, 'closed_set', 'acc_cos').est:.1f}",
                 "NexaRoom rule N=5: correct / wrong / stranger %":
                     f"{100 * get(mm, d, 5, 'nexaroom_default', 'member_correct').est:.1f} / "
                     f"{100 * get(mm, d, 5, 'nexaroom_default', 'member_wrong').est:.2f} / "
                     f"{100 * get(mm, d, 5, 'nexaroom_default', 'stranger_fa').est:.2f}",
                 "AS-norm @ dev 1% FA, N=5: correct %": f"{100 * get(mm, d, 5, 'asnorm@dev_fa1', 'member_correct').est:.1f}"}
            t = timing[(timing.model == TIME_KEY[m]) & (timing.audio_s == float(d))]
            r["CPU ms"] = round(float(t.cpu_ms_median.iloc[0]), 1) if len(t) else ""
            rows.append(r)
    x = pd.DataFrame(rows)
    write(x, "headline", "One-line-per-condition summary of every configuration", x)


# ---------------------------------------------------------------- figures
def legend_below(fig, ax, ncol=3, y=-0.04):
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, y))


def f_closed(mainr) -> None:
    cs = mainr[(mainr.rule == "closed_set") & (mainr.metric == "acc_cos")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, d in zip(axes, DURS):
        for m in present(MAIN):
            g = cs[(cs.config == m) & (cs.dur == d)].sort_values("N")
            ax.plot(g.N, 100 * g.est, marker=MARKERS[m], color=COLORS[m], label=SHORT[m], ms=4.5)
            ax.fill_between(g.N, 100 * g.lo, 100 * g.hi, color=COLORS[m], alpha=0.15, linewidth=0)
        logx(ax)
        ax.set_title(f"{d} s test utterance")
        ax.set_xlabel("Enrolled speakers N")
    axes[0].set_ylabel("Closed-set accuracy (%)")
    fig.tight_layout()
    legend_below(fig, axes[0])
    save(fig, "fig_closed_set_vs_N")


def f_eer(eer) -> None:
    ms = present(MAIN)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=False)
    for ax, d in zip(axes, DURS):
        for i, m in enumerate(ms):
            for j, (sc, fill) in enumerate((("cos", True), ("asnorm", False))):
                g = eer[(eer.config == m) & (eer.score == sc) & (eer.dur == d)].iloc[0]
                y = i + (j - 0.5) * 0.32
                ax.errorbar(100 * g.eer, y, xerr=[[100 * (g.eer - g.eer_lo)], [100 * (g.eer_hi - g.eer)]],
                            fmt=MARKERS[m], color=COLORS[m], mfc=COLORS[m] if fill else "white", mew=1.2, ms=5,
                            elinewidth=1, capsize=0)
        ax.set_yticks(range(len(ms)))
        ax.set_yticklabels([SHORT[m] for m in ms] if d == 1 else [""] * len(ms))
        ax.invert_yaxis()
        ax.set_title(f"{d} s test utterance")
        ax.set_xlabel("EER (%)")
        ax.set_xlim(left=0)
        ax.grid(axis="y", visible=False)
    axes[2].plot([], [], "o", color=INK2, label="cosine to centroid (filled)")
    axes[2].plot([], [], "o", color=INK2, mfc="white", mew=1.2, label="AS-norm (open)")
    fig.tight_layout()
    legend_below(fig, axes[2], ncol=2, y=0.0)
    save(fig, "fig_eer")


def f_tradeoff(mainr) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
    for ax, d in zip(axes, DURS):
        for m in present(MAIN):
            cur = pd.read_csv(R / m / "curves.csv")
            g = cur[(cur.dur == d) & (cur.N == 5) & (cur.family == "cos")].sort_values("param")
            g = g[g.stranger_fa > 0]
            ax.plot(100 * g.stranger_fa, 100 * g.member_correct, color=COLORS[m], label=SHORT[m], lw=1.3)
            mm = mainr[(mainr.config == m)]
            fa = get(mm, d, 5, "nexaroom_default", "stranger_fa").est
            ca = get(mm, d, 5, "nexaroom_default", "member_correct").est
            ax.plot(100 * fa, 100 * ca, MARKERS[m], color=COLORS[m], mec="white", mew=0.8, ms=6.5, zorder=5)
        ax.set_xscale("log")
        ax.set_xlim(0.05, 60)
        plain_log(ax)
        ax.set_ylim(0, 100)
        ax.set_title(f"{d} s test utterance")
        ax.set_xlabel("Stranger wakes accepted (%)")
    axes[0].set_ylabel("Member wakes named correctly (%)")
    axes[0].plot([], [], "-", color=INK2, label="line: cosine, threshold swept")
    axes[0].plot([], [], "o", color=INK2, label="marker: NexaRoom rule, defaults")
    fig.tight_layout()
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    save(fig, "fig_open_set_tradeoff")
    # families for the default model
    m0 = "campplus_zhen_adv"
    cur = pd.read_csv(R / m0 / "curves.csv")
    mm = mainr[mainr.config == m0]
    fam_col = {"cos": "#2a78d6", "nexaroom": "#eb6834", "asnorm": "#1baf7a"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
    for ax, d in zip(axes, DURS):
        for fam, lab in (("cos", "cosine to centroid, threshold swept"),
                         ("nexaroom", "NexaRoom rule, base threshold swept"),
                         ("asnorm", "AS-norm cosine, threshold swept")):
            g = cur[(cur.dur == d) & (cur.N == 5) & (cur.family == fam)].sort_values("param")
            g = g[g.stranger_fa > 0]
            ax.plot(100 * g.stranger_fa, 100 * g.member_correct, color=fam_col[fam], label=lab)
        for rule, mk, lab in (("nexaroom_default", "o", "NexaRoom rule, defaults"),
                              ("cos@0.50", "s", "cosine >= 0.50")):
            ax.plot(100 * get(mm, d, 5, rule, "stranger_fa").est, 100 * get(mm, d, 5, rule, "member_correct").est,
                    mk, color=INK, mfc="white", mew=1.2, ms=6, label=lab, zorder=5)
        ax.axvline(1.0, color=INK2, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(0.05, 60)
        plain_log(ax)
        ax.set_ylim(0, 100)
        ax.set_title(f"{d} s test utterance")
        ax.set_xlabel("Stranger wakes accepted (%)")
    axes[0].set_ylabel("Member wakes named correctly (%)")
    fig.tight_layout()
    legend_below(fig, axes[0], ncol=3, y=-0.02)
    save(fig, "fig_open_set_tradeoff_campplus")


def f_nexaroom_vs_N(mainr) -> None:
    r = mainr[mainr.rule == "nexaroom_default"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True)
    for j, d in enumerate(DURS):
        for m in present(MAIN):
            for i, met in enumerate(("member_correct", "stranger_fa")):
                g = r[(r.config == m) & (r.dur == d) & (r.metric == met)].sort_values("N")
                axes[i, j].plot(g.N, 100 * g.est, marker=MARKERS[m], color=COLORS[m], label=SHORT[m], ms=4.5)
        axes[0, j].set_title(f"{d} s test utterance")
        axes[1, j].set_xlabel("Enrolled speakers N")
        for i in range(2):
            logx(axes[i, j])
        axes[1, j].set_yscale("log")
    for j in range(3):
        log_y_ticks(axes[1, j])
    axes[0, 0].set_ylabel("Member wakes named\ncorrectly (%)")
    axes[1, 0].set_ylabel("Stranger wakes named\nas a member (%)")
    fig.tight_layout()
    legend_below(fig, axes[0, 0], y=-0.01)
    save(fig, "fig_nexaroom_default_vs_N")


def f_eer_cpu(eer, timing) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for m in present(MAIN + ["campplus_zhen_adv_ort131"]):
        tm = timing[(timing.model == TIME_KEY[m]) & timing.utterance.str.startswith("1 s")].cpu_ms_median
        ee = eer[(eer.config == m) & (eer.score == "cos") & (eer.dur == 1)].iloc[0]
        ax.errorbar(tm.iloc[0], 100 * ee.eer, yerr=[[100 * (ee.eer - ee.eer_lo)], [100 * (ee.eer_hi - ee.eer)]],
                    fmt=MARKERS[m], color=COLORS[m], ms=7, elinewidth=1, label=SHORT[m])
    ax.set_xlabel("CPU time per 1 s utterance, one vCPU (ms)")
    ax.set_ylabel("EER at 1 s, cosine (%)")
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 12)
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    save(fig, "fig_eer_vs_cpu")


def f_gender() -> None:
    ge = load("gender_eer.csv", MAIN)
    ms = present(MAIN)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, d in zip(axes, DURS):
        for i, m in enumerate(ms):
            s = ge[(ge.config == m) & (ge.dur == d) & (ge.nontarget_gender == "same")].iloc[0]
            c = ge[(ge.config == m) & (ge.dur == d) & (ge.nontarget_gender == "cross")].iloc[0]
            ax.plot([100 * c.eer, 100 * s.eer], [i, i], color=GRID, lw=2.5, zorder=1, solid_capstyle="round")
            ax.errorbar(100 * c.eer, i, xerr=[[100 * (c.eer - c.eer_lo)], [100 * (c.eer_hi - c.eer)]], fmt=MARKERS[m],
                        color=COLORS[m], mfc="white", mew=1.2, ms=5.5, elinewidth=1, zorder=3)
            ax.errorbar(100 * s.eer, i, xerr=[[100 * (s.eer - s.eer_lo)], [100 * (s.eer_hi - s.eer)]], fmt=MARKERS[m],
                        color=COLORS[m], ms=5.5, elinewidth=1, zorder=3)
        ax.set_yticks(range(len(ms)))
        ax.set_yticklabels([SHORT[m] for m in ms])
        ax.invert_yaxis()
        ax.set_xlim(left=0)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{d} s test utterance")
        ax.set_xlabel("EER (%)")
    axes[2].plot([], [], "o", color=INK2, label="same-gender non-targets (filled)")
    axes[2].plot([], [], "o", color=INK2, mfc="white", mew=1.2, label="cross-gender non-targets (open)")
    fig.tight_layout()
    legend_below(fig, axes[2], ncol=2, y=0.0)
    save(fig, "fig_gender")


def f_enrollment() -> None:
    ab = pd.read_csv(R / "campplus_zhen_adv" / "ablations.csv")
    en = ab[ab.ablation == "enroll"]
    cfgs = ["3x4", "5x4", "3x8", "5x8"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    shades = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
    for d in DURS:
        for ax, met, lab in zip(axes, ("acc_cos", "nexaroom_member_correct", "nexaroom_stranger_fa"),
                                ("Closed-set accuracy, N = 5", "Members named correctly\n(NexaRoom rule)",
                                 "Strangers accepted\n(NexaRoom rule)")):
            g = en[(en.dur == d) & (en.metric == met)].set_index("config").reindex(cfgs)
            ax.errorbar(range(4), 100 * g.est, yerr=[100 * (g.est - g.lo), 100 * (g.hi - g.est)],
                        marker="o", color=shades[d], ms=4.5, elinewidth=1, capsize=0, label=f"{d} s test")
            ax.set_title(lab)
            ax.set_ylabel("%")
            ax.set_xticks(range(4))
            ax.set_xticklabels(cfgs)
            ax.set_xlabel("Enrollment: sentences x clips")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig_enrollment_size")


def f_ladder(mainr) -> None:
    m0 = "campplus_zhen_adv"
    mm = mainr[mainr.config == m0]
    steps = ["ladder_a", "ladder_b", "ladder_c", "ladder_d", "ladder_e"]
    labs = ["a: cos\n>= 0.50", "b: blend\n>= 0.50", "c: calibrated\nthreshold", "d: + short-\naudio penalty",
            "e: + margin\n(NexaRoom)"]
    shades = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for d in DURS:
        c = [100 * get(mm, d, 5, s, "member_correct").est for s in steps]
        f = [100 * get(mm, d, 5, s, "stranger_fa").est for s in steps]
        axes[0].plot(range(5), c, marker="o", color=shades[d], label=f"{d} s test")
        axes[1].plot(range(5), f, marker="o", color=shades[d], label=f"{d} s test")
    for ax, lab in zip(axes, ("Member wakes named correctly (%)", "Stranger wakes accepted (%)")):
        ax.set_xticks(range(5))
        ax.set_xticklabels(labs, fontsize=6.5)
        ax.set_ylabel(lab)
    axes[1].set_yscale("log")
    log_y_ticks(axes[1])
    axes[0].legend(frameon=False)
    fig.suptitle("Adding NexaRoom's rule elements one at a time (3D-Speaker CAM++, N = 5)", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig_rule_ladder")


def f_variants(eer) -> None:
    ks = ["campplus_zhen_adv", "campplus_zhen_adv_d1", "campplus_zhen_adv_ort131"]
    labs = {"campplus_zhen_adv": "sherpa-onnx 2.0.1 (NexaRoom)", "campplus_zhen_adv_d1": "+ 1-LSB noise floor",
            "campplus_zhen_adv_ort131": "onnxruntime 1.31.1"}
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    for i, m in enumerate(present(ks)):
        g = eer[(eer.config == m) & (eer.score == "cos")].sort_values("dur")
        x = np.arange(3) + (i - 1) * 0.18
        ax.errorbar(x, 100 * g.eer, yerr=[100 * (g.eer - g.eer_lo), 100 * (g.eer_hi - g.eer)], fmt=MARKERS[m],
                    color=COLORS[m], ms=6, elinewidth=1, capsize=0, label=labs[m], ls="none")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["1 s", "2 s", "3 s"])
    ax.set_ylabel("EER, cosine (%)")
    ax.set_ylim(bottom=0)
    ax.set_title("3D-Speaker CAM++: same weights, different runtime path")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig_runtime_variants")


def f_diagnostics() -> None:
    p = TAB / "diagnostics_subset.csv"
    df = pd.read_csv(p)
    order = ["sherpa-onnx as shipped", "sherpa-onnx + 1-LSB noise floor", "sherpa-onnx + CMN metadata",
             "sherpa-onnx + CMN metadata + 1-LSB noise floor", "WeSpeaker front end, onnxruntime 1.31.1, no noise floor",
             "WeSpeaker front end, onnxruntime 1.31.1, no CMN",
             "WeSpeaker front end, onnxruntime 1.31.1 (noise floor + CMN)"]
    ms = [("wespeaker_campplus_vox", "WeSpeaker CAM++ VoxCeleb", "#eda100", "D"),
          ("wespeaker_resnet34_vox", "WeSpeaker ResNet34 VoxCeleb", "#e87ba4", "v")]
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    for k, (m, lab, col, mk) in enumerate(ms):
        g = df[df.model == m].set_index("configuration").reindex(order)
        ax.plot(g["EER cosine 1 s %"], np.arange(len(order)) + (k - 0.5) * 0.3, mk, color=col, ms=6, label=lab,
                ls="none")
    ref = df[(df.model == "campplus_zhen_adv") & (df.configuration == "sherpa-onnx as shipped")]
    if len(ref):
        ax.axvline(ref["EER cosine 1 s %"].iloc[0], color=INK2, ls=":", lw=1)
        ax.text(ref["EER cosine 1 s %"].iloc[0] + 0.6, -0.45, "3D-Speaker CAM++ via sherpa-onnx",
                fontsize=6.5, color=INK2, va="center")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 55)
    ax.axvline(50, color=GRID, lw=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("EER at 1 s, cosine (%), 100-speaker subset")
    ax.set_ylim(len(order) - 0.5, -0.8)
    fig.tight_layout()
    legend_below(fig, ax, ncol=2, y=0.0)
    save(fig, "fig_wespeaker_diagnostics")


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    mainr = load("main.csv")
    eer = load("eer.csv")
    timing = pd.read_csv(R / "timing.csv")
    t_data(mainr, eer)
    t_models(timing)
    t_eer(eer)
    t_closed(mainr)
    t_rules(mainr)
    t_open_tuned(mainr)
    t_gender()
    t_timing(timing)
    t_ablations()
    t_diagnostics()
    t_paired()
    t_concentration()
    t_headline(mainr, eer, timing)
    f_closed(mainr)
    f_eer(eer)
    f_tradeoff(mainr)
    f_nexaroom_vs_N(mainr)
    f_eer_cpu(eer, timing)
    f_gender()
    f_enrollment()
    f_ladder(mainr)
    f_variants(eer)
    f_diagnostics()
    lines = ["# results/tables index", "", "Written by experiments/05_report.py (and 19_paired.py, 14_summary_tables.py)"
             " from the per-configuration result files. Percentages unless stated; brackets are 95% "
             "speaker-level bootstrap CIs.", "", "| file | content |", "|---|---|"]
    lines += [f"| `{n}.csv` / `{n}.md` | {d} |" for n, d in INDEX]
    lines += ["| `summary.md` | Compact cross-model summary (14_summary_tables.py) |"]
    (TAB / "INDEX.md").write_text("\n".join(lines) + "\n")
    print("configs:", present(ALL))
    print("tables:", len(INDEX))


if __name__ == "__main__":
    main()
