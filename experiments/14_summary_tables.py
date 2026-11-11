"""Compact cross-model summary tables (Markdown) for README.md.

Reads results/<model>/{eer,main,gender_eer}.csv for every evaluated model and
writes results/tables/summary.md. Percentages, 95% speaker-bootstrap CIs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swk import data  # noqa: E402

ORDER = ["campplus_zhen_adv", "campplus_zhen_adv_d1", "campplus_zhen_adv_ort131", "eres2net_base_zh",
         "titanet_small", "wespeaker_campplus_vox_wsfe", "wespeaker_resnet34_vox_wsfe",
         "wespeaker_resnet34_cnc_wsfe", "wespeaker_campplus_vox"]
SHORT = {"campplus_zhen_adv": "3D-Speaker CAM++ (NexaRoom default)",
         "campplus_zhen_adv_d1": "3D-Speaker CAM++, 1-LSB noise floor",
         "campplus_zhen_adv_ort131": "3D-Speaker CAM++, onnxruntime 1.31.1",
         "eres2net_base_zh": "3D-Speaker ERes2Net-base",
         "titanet_small": "NeMo TitaNet-small",
         "wespeaker_campplus_vox_wsfe": "WeSpeaker CAM++ VoxCeleb (WeSpeaker front end)",
         "wespeaker_resnet34_vox_wsfe": "WeSpeaker ResNet34 VoxCeleb (WeSpeaker front end)",
         "wespeaker_resnet34_cnc_wsfe": "WeSpeaker ResNet34 CN-Celeb (WeSpeaker front end)",
         "wespeaker_campplus_vox": "WeSpeaker CAM++ VoxCeleb, sherpa-onnx as shipped"}


def ci(r, k="est", lo="lo", hi="hi", d=1) -> str:
    return f"{100 * r[k]:.{d}f} [{100 * r[lo]:.{d}f}, {100 * r[hi]:.{d}f}]"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join(lines)


def main() -> None:
    models = [m for m in ORDER if (data.RESULTS / m / "eval.done").exists()]
    parts = []
    # T1: EER and closed-set accuracy
    rows = []
    for m in models:
        e = pd.read_csv(data.RESULTS / m / "eer.csv")
        mm = pd.read_csv(data.RESULTS / m / "main.csv")
        for d in (1, 2, 3):
            ec = e[(e.dur == d) & (e.score == "cos")].iloc[0]
            ea = e[(e.dur == d) & (e.score == "asnorm")].iloc[0]
            a5 = mm[(mm.rule == "closed_set") & (mm.metric == "acc_cos") & (mm.N == 5) & (mm.dur == d)].iloc[0]
            a20 = mm[(mm.rule == "closed_set") & (mm.metric == "acc_cos") & (mm.N == 20) & (mm.dur == d)].iloc[0]
            rows.append({"model": SHORT[m], "test": f"{d} s", "EER cosine %": ci(ec, "eer", "eer_lo", "eer_hi", 2),
                         "EER AS-norm %": ci(ea, "eer", "eer_lo", "eer_hi", 2),
                         "closed-set acc. N=5 %": ci(a5), "closed-set acc. N=20 %": ci(a20)})
    parts.append("### EER and closed-set identification accuracy (cosine to centroid)\n\n" + md(pd.DataFrame(rows)))
    # T2: open set, N = 5
    rows = []
    for m in models:
        mm = pd.read_csv(data.RESULTS / m / "main.csv")
        x = mm[mm.N == 5]
        for d in (1, 2, 3):
            def g(rule, metric):
                r = x[(x.rule == rule) & (x.metric == metric) & (x.dur == d)]
                return r.iloc[0] if len(r) else None
            row = {"model": SHORT[m], "test": f"{d} s"}
            for rule, lab in (("nexaroom_default", "NexaRoom rule"), ("cos@0.50", "cosine >= 0.50")):
                row[f"{lab}: correct %"] = f"{100 * g(rule, 'member_correct').est:.1f}"
                row[f"{lab}: wrong member %"] = f"{100 * g(rule, 'member_wrong').est:.2f}"
                row[f"{lab}: stranger accepted %"] = f"{100 * g(rule, 'stranger_fa').est:.2f}"
            for rule, lab in (("cos@dev_fa1", "cosine"), ("nexaroom@dev_fa1", "NexaRoom rule"),
                              ("asnorm@dev_fa1", "AS-norm")):
                r = g(rule, "member_correct")
                row[f"1% stranger FA (dev-tuned), {lab}: correct %"] = ci(r)
            rows.append(row)
    parts.append("### Open set, household of N = 5 (member wakes and stranger wakes)\n\n"
                 "Dev-tuned thresholds aim at 1% stranger acceptance on dev speakers; the realised value on "
                 "eval speakers is in main.csv (`stranger_fa`).\n\n" + md(pd.DataFrame(rows)))
    # T3: gender
    rows = []
    for m in models:
        g = pd.read_csv(data.RESULTS / m / "gender_eer.csv")
        for d in (1, 3):
            r = {"model": SHORT[m], "test": f"{d} s"}
            for rel in ("same", "cross"):
                x = g[(g.dur == d) & (g.nontarget_gender == rel)].iloc[0]
                r[f"EER, non-targets of {rel} gender %"] = ci(x, "eer", "eer_lo", "eer_hi", 2)
            rows.append(r)
    parts.append("### Gender (speakers with a male/female label only)\n\n" + md(pd.DataFrame(rows)))
    out = data.RESULTS / "tables" / "summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join(parts) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
