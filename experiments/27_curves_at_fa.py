"""Member wakes named at a fixed EVAL stranger acceptance, read from the trade-off curves (revision).

With the dev-trained back end (results/<model>__cw), thresholds tuned on dev
speakers do not transfer (the dev speakers trained the transform, so their
non-target scores are lower and the realized eval stranger acceptance overshoots
the 1% target). To compare score quality at an equal operating point, this script
interpolates each configuration's eval trade-off curve (curves.csv: thresholds
swept on eval trials, averaged over seeds; N = 5) at a stranger acceptance of 1%
(linear in log stranger acceptance between the two neighbouring grid points).
This is an eval-set operating point, not a deployable threshold; it is used only
to compare raw and back-end scores, like the EER.

Output: results/curves_at_fa1.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swk import data  # noqa: E402

MAIN = ["campplus_zhen_adv", "eres2net_base_zh", "titanet_small", "wespeaker_campplus_vox_wsfe",
        "wespeaker_resnet34_vox_wsfe", "wespeaker_resnet34_cnc_wsfe"]
TARGET = 0.01


def at_fa(cur: pd.DataFrame, target: float) -> tuple[float, float]:
    c = cur.sort_values("param")
    fa, mc, mw = c.stranger_fa.to_numpy(), c.member_correct.to_numpy(), c.member_wrong.to_numpy()
    # stranger_fa decreases with the threshold; find the bracketing grid points
    i = np.flatnonzero((fa[:-1] >= target) & (fa[1:] <= target))
    if not i.size:
        return float("nan"), float("nan")
    i = i[0]
    x0, x1 = np.log(max(fa[i], 1e-9)), np.log(max(fa[i + 1], 1e-9))
    w = 0.0 if x0 == x1 else (np.log(target) - x0) / (x1 - x0)
    return float(mc[i] + w * (mc[i + 1] - mc[i])), float(mw[i] + w * (mw[i + 1] - mw[i]))


def main() -> None:
    rows = []
    for m in MAIN:
        for cfg in (m, f"{m}__cw"):
            f = data.RESULTS / cfg / "curves.csv"
            if not f.exists():
                continue
            cur = pd.read_csv(f)
            for dur in (1, 2, 3):
                for fam in ("cos", "asnorm"):
                    c = cur[(cur.dur == dur) & (cur.N == 5) & (cur.family == fam)]
                    mc, mw = at_fa(c, TARGET)
                    rows.append({"model": m, "config": cfg, "backend": "cw" if cfg.endswith("__cw") else "none",
                                 "dur": dur, "family": fam, "eval_stranger_fa": TARGET,
                                 "member_correct": mc, "member_wrong": mw})
    df = pd.DataFrame(rows)
    df.to_csv(data.RESULTS / "curves_at_fa1.csv", index=False)
    print(df.pivot_table(index=["model", "dur"], columns=["family", "backend"], values="member_correct")
          .mul(100).round(1).to_string())


if __name__ == "__main__":
    main()
