"""Split the expert survey labels by respondent and draw the LLM test sample and few-shot pool.

data/expert/split.json        respondent ids in train / test (80/20, grouped so nobody is in both)
data/expert/test_items.jsonl  test sample for every model: PSF-stratified, natural label distribution
data/expert/shots.jsonl       few-shot examples from TRAIN respondents only, balanced per level
"""
from __future__ import annotations

import argparse
import collections
import json
import random

from bench.spec import LEVELS, ROOT


def load_responses() -> list[dict]:
    return [json.loads(x) for x in open(ROOT / "data/survey/responses.jsonl", encoding="utf-8")]


def split_ids(rows: list[dict], test_share: float, seed: int) -> tuple[set, set]:
    rids = sorted({r["rid"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(rids)
    n_test = round(len(rids) * test_share)
    return set(rids[n_test:]), set(rids[:n_test])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-share", type=float, default=0.2)
    ap.add_argument("--per-psf", type=int, default=120)
    ap.add_argument("--shots-per-level", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261118)
    a = ap.parse_args()
    rows = load_responses()
    train, test = split_ids(rows, a.test_share, a.seed)
    rng = random.Random(a.seed)
    out = ROOT / "data/expert"
    out.mkdir(parents=True, exist_ok=True)
    (out / "split.json").write_text(json.dumps({"seed": a.seed, "train": sorted(train), "test": sorted(test)},
                                               ensure_ascii=False, indent=0), encoding="utf-8")
    test_rows = [r for r in rows if r["rid"] in test]
    train_rows = [r for r in rows if r["rid"] in train]
    sample, shots = [], []
    for pid in range(1, 9):
        pool = [r for r in test_rows if r["psf"] == pid]
        rng.shuffle(pool)
        for r in pool[: a.per_psf]:
            sample.append({"id": r["id"], "psf": pid, "choice": r["choice"], "label": r["answer"], "rid": r["rid"]})
        tr = [r for r in train_rows if r["psf"] == pid]
        rng.shuffle(tr)
        for lv in LEVELS[pid]:
            for r in [x for x in tr if x["answer"] == lv][: a.shots_per_level]:
                shots.append({"psf": pid, "choice": r["choice"], "label": lv})
    rng.shuffle(shots)  # mix levels so the prompt does not list them in a fixed order
    shots.sort(key=lambda s: s["psf"])
    for name, data in [("test_items", sample), ("shots", shots)]:
        with open(out / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"respondents train={len(train)} test={len(test)}; labelled rows train={len(train_rows)} test={len(test_rows)}")
    print("LLM test sample:", len(sample), dict(collections.Counter(s["psf"] for s in sample)))
    for pid in range(1, 9):
        print(f"  F.{pid}", dict(collections.Counter(s["label"] for s in sample if s["psf"] == pid)))
    print("shots:", len(shots))
    # consensus set: unique scenarios judged by >= 3 TEST respondents, labelled by their strict majority
    by = collections.defaultdict(list)
    for r in test_rows:
        by[(r["psf"], tuple(sorted(r["choice"].items())))].append(r["answer"])
    cons = []
    for (pid, ch), answers in sorted(by.items()):
        if len(answers) < 3:
            continue
        top = collections.Counter(answers).most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            continue  # tie: no consensus
        cons.append({"id": f"C{pid}-{len(cons):04d}", "psf": pid, "choice": dict(ch), "label": top[0][0],
                     "n_raters": len(answers), "dist": dict(collections.Counter(answers))})
    with open(out / "consensus_items.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in cons:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "llm_items.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in sample + cons:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("consensus scenarios:", len(cons), dict(collections.Counter(c["psf"] for c in cons)),
          "-> llm_items.jsonl", len(sample) + len(cons))
    # natural-distribution few-shot pool: 24 random train answers per PSF (own RNG keeps the files above stable)
    rng2 = random.Random(a.seed + 1)
    nat = []
    for pid in range(1, 9):
        tr = [r for r in train_rows if r["psf"] == pid]
        nat += [{"psf": pid, "choice": r["choice"], "label": r["answer"]} for r in rng2.sample(tr, 24)]
    with open(out / "shots_natural.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in nat:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("natural shots:", len(nat))


if __name__ == "__main__":
    main()
