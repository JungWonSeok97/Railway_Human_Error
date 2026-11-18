"""Build the evaluation set: random survey answers per PSF, labelled by the CatBoost reference.

For each PSF scored by CatBoost in the default path (F.1-3, F.5-8) we draw random answer combinations,
label them with CatBoost, then take a class-balanced test set and a disjoint few-shot pool.
HEP profiles combine one test item per PSF with a random role and training-elapsed months.
"""
from __future__ import annotations

import argparse
import collections
import json
import random

from bench.reference import load_spec, predict
from bench.spec import LEVELS, ROLE_BASE_HEP, ROOT

PSF_IDS = [1, 2, 3, 5, 6, 7, 8]


def draw(spec_psf: dict, rng: random.Random) -> dict:
    return {it["key"]: rng.randrange(len(it["options"])) for it in spec_psf["items"]}


def to_values(spec_psf: dict, choice: dict) -> dict:
    return {it["key"]: it["options"][choice[it["key"]]]["value"] for it in spec_psf["items"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-psf", type=int, default=90)
    ap.add_argument("--shots", type=int, default=9)
    ap.add_argument("--pool", type=int, default=20000)
    ap.add_argument("--profiles", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261117)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    spec = load_spec()
    items, shots, natural = [], [], {}
    for pid in PSF_IDS:
        s = spec[str(pid)]
        seen, pool = set(), []
        for _ in range(a.pool):
            c = draw(s, rng)
            key = tuple(sorted(c.items()))
            if key not in seen:
                seen.add(key)
                pool.append(c)
        labels, proba = predict(pid, [to_values(s, c) for c in pool])
        natural[pid] = dict(collections.Counter(labels))
        by_class = collections.defaultdict(list)
        for c, lb, pr in zip(pool, labels, proba):
            by_class[lb].append((c, lb, [round(float(x), 4) for x in pr]))
        for lst in by_class.values():
            rng.shuffle(lst)
        n_cls = len(LEVELS[pid])
        # few-shot pool first (balanced), then the test set (balanced as far as each class allows)
        for k in range(a.shots):
            lv = LEVELS[pid][k % n_cls]
            if by_class[lv]:
                c, lb, pr = by_class[lv].pop()
                shots.append({"psf": pid, "choice": c, "label": lb})
        want = a.per_psf // n_cls
        taken = []
        for lv in LEVELS[pid]:
            taken += by_class[lv][:want]
            by_class[lv] = by_class[lv][want:]
        rest = [x for lv in LEVELS[pid] for x in by_class[lv]]
        rng.shuffle(rest)
        taken += rest[: a.per_psf - len(taken)]
        for i, (c, lb, pr) in enumerate(taken):
            items.append({"id": f"F{pid}-{i:03d}", "psf": pid, "choice": c, "label": lb, "proba": pr})
        print(pid, s["name"], "unique pool", len(pool), "natural", natural[pid],
              "test", dict(collections.Counter(t[1] for t in taken)))
    by_psf = collections.defaultdict(list)
    for it in items:
        by_psf[it["psf"]].append(it["id"])
    roles = list(ROLE_BASE_HEP)
    profiles = [{"id": f"P{j:03d}", "role": rng.choice(roles), "months": round(rng.uniform(0, 48), 1),
                 "items": {str(pid): rng.choice(by_psf[pid]) for pid in PSF_IDS}} for j in range(a.profiles)]
    out = ROOT / "data"
    for name, rows in [("items", items), ("shots", shots), ("profiles", profiles)]:
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "natural_distribution.json").write_text(json.dumps(natural, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(items), "items,", len(shots), "shots,", len(profiles), "profiles")


if __name__ == "__main__":
    main()
