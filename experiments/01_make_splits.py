"""Build the speaker pool, per-seed splits and the table of utterances to embed.

Outputs
  results/speakers.csv            pool speakers (spk id, hashed client id, gender, clip counts)
  results/splits/seed{s}.json     enrollment / test clip ids per speaker, dev flag, fixed-word clips
  results/split_summary.json      counts
  cache/specs.csv                 one row per unique utterance to embed
  cache/roles.csv                 (seed, spk, role, index) -> spec_id
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swk import data, splits  # noqa: E402

FW_WORDS = ["این", "برای", "خیلی", "باید", "اینجا"]
ENROLL_GAP_S = 0.10


def main() -> None:
    meta = data.load_meta()
    spk, sp = splits.make_splits(meta)
    (data.RESULTS / "splits").mkdir(parents=True, exist_ok=True)
    data.CACHE.mkdir(parents=True, exist_ok=True)
    spk.to_csv(data.RESULTS / "speakers.csv", index=False)

    clip_src = meta.set_index("clip").src
    clip_word = meta.set_index("clip").word
    hash_of = dict(zip(spk.spk, spk.speaker_hash))
    by_spk_word = meta[meta.speaker_hash.isin(spk.speaker_hash)].groupby(["speaker_hash", "word"]).clip.apply(list)

    specs: dict[tuple, int] = {}
    spec_rows: list[dict] = []
    roles: list[tuple] = []

    def add(clips: tuple[int, ...], gap: float, trim: bool, kind: str, sset: str, spk_id: int) -> int:
        key = (clips, gap, trim)
        if key not in specs:
            specs[key] = len(spec_rows)
            spec_rows.append({"spec_id": len(spec_rows), "spk": spk_id, "clips": " ".join(map(str, clips)),
                              "n_clips": len(clips), "gap": gap, "trim": trim, "kind": kind, "set": sset})
        elif sset == "main" and spec_rows[specs[key]]["set"] != "main":
            spec_rows[specs[key]]["set"] = "main"
        return specs[key]

    summary = {"n_speakers": int(len(spk)), "gender": spk.gender.value_counts().to_dict(), "seeds": {}}
    for seed, table in sp.items():
        n_rich = 0
        fw_counts = {w: 0 for w in FW_WORDS}
        for k, r in table.items():
            e, t = r["enroll"], r["test"]
            # enrollment sentences
            for cfg, (n_sent, per) in splits.ENROLL_CONFIGS.items():
                if n_sent * per > len(e):
                    continue
                sset = "main" if cfg == "3x4" else "ablation"
                for j in range(n_sent):
                    sid = add(tuple(e[j * per:(j + 1) * per]), ENROLL_GAP_S, False, "enroll", sset, k)
                    roles.append((seed, k, f"enroll_{cfg}", j, sid))
            if len(e) >= splits.RICH_ENROLL:
                n_rich += 1
            for j in range(3):
                sid = add(tuple(e[j * 4:(j + 1) * 4]), ENROLL_GAP_S, True, "enroll", "ablation", k)
                roles.append((seed, k, "enroll_3x4_trim", j, sid))
            for dur, groups in splits.TEST_GROUPS.items():
                for j, g in enumerate(groups):
                    sid = add(tuple(t[i] for i in g), 0.0, False, "test", "main", k)
                    roles.append((seed, k, f"test{dur}", j, sid))
            for j in range(splits.N_TEST):
                sid = add((t[j],), 0.0, True, "test", "ablation", k)
                roles.append((seed, k, "test1_trim", j, sid))
            # fixed-word test clips: one clip of word W from a source not used by the 3x4 enrollment
            enroll_src = {clip_src[c] for c in e[:12]}
            rng = np.random.default_rng(int(seed) * 1_000_003 + int(k))
            r["fw"] = {}
            for w in FW_WORDS:
                cand = [c for c in by_spk_word.get((hash_of[k], w), []) if clip_src[c] not in enroll_src]
                if cand:
                    c = int(cand[int(rng.integers(len(cand)))])
                    r["fw"][w] = c
                    fw_counts[w] += 1
                    sid = add((c,), 0.0, False, "test", "ablation", k)
                    roles.append((seed, k, f"fw_{w}", 0, sid))
        with open(data.RESULTS / "splits" / f"seed{seed}.json", "w") as f:
            json.dump({str(k): v for k, v in table.items()}, f)
        summary["seeds"][int(seed)] = {"n": len(table), "dev": int(sum(v["dev"] for v in table.values())),
                                       "rich_enroll_5x8": n_rich, "fixed_word": fw_counts}
        # sanity: enrollment and test never share a source recording
        for k, r in table.items():
            es = {clip_src[c] for c in r["enroll"]}
            ts = {clip_src[c] for c in r["test"]}
            assert not (es & ts), (seed, k)
            assert len(set(r["enroll"]) | set(r["test"])) == len(r["enroll"]) + len(r["test"])
    specs_df = pd.DataFrame(spec_rows)
    specs_df.to_csv(data.CACHE / "specs.csv", index=False)
    pd.DataFrame(roles, columns=["seed", "spk", "role", "index", "spec_id"]).to_csv(data.CACHE / "roles.csv", index=False)
    summary["n_specs"] = specs_df.set.value_counts().to_dict()
    summary["enroll_gap_s"] = ENROLL_GAP_S
    summary["fw_words"] = FW_WORDS
    summary["words_in_test_clips"] = int(clip_word[[c for t in sp[0].values() for c in t["test"]]].nunique())
    with open(data.RESULTS / "split_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
