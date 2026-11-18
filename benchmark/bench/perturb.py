"""Monotonicity probe: make one answer strictly worse and check the predicted level never improves.

Writes data/items_perturbed.jsonl (ids "<item id>~<item key>") labelled by CatBoost, so the same LLM runner
can be used. score_monotonic() compares each run on the perturbed set with the same model on the base set.
"""
from __future__ import annotations

import json
import random

from bench.reference import load_spec, predict
from bench.spec import LEVELS, ROOT


def main(n_per_psf: int = 30, seed: int = 7) -> None:
    rng = random.Random(seed)
    spec = load_spec()
    items = [json.loads(x) for x in open(ROOT / "data/items.jsonl", encoding="utf-8")]
    out = []
    for pid in sorted({i["psf"] for i in items}):
        s = spec[str(pid)]
        pool = [i for i in items if i["psf"] == pid]
        rng.shuffle(pool)
        for it in pool[:n_per_psf]:
            cands = []
            for x in s["items"]:
                cur = x["options"][it["choice"][x["key"]]]["value"]
                worse = [k for k, o in enumerate(x["options"]) if o["value"] > cur]
                if worse:
                    cands.append((x["key"], max(worse, key=lambda k: x["options"][k]["value"])))
            if not cands:
                continue
            key, opt = rng.choice(cands)
            ch = dict(it["choice"], **{key: opt})
            vals = {x["key"]: x["options"][ch[x["key"]]]["value"] for x in s["items"]}
            lab = predict(pid, [vals])[0][0]
            out.append({"id": f"{it['id']}~{key}", "base": it["id"], "psf": pid, "choice": ch, "label": lab})
    with open(ROOT / "data/items_perturbed.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    items_by = {i["id"]: i for i in items}
    viol = sum(LEVELS[r["psf"]].index(r["label"]) < LEVELS[r["psf"]].index(items_by[r["base"]]["label"]) for r in out)
    print(len(out), "perturbed items; CatBoost violations:", viol)


if __name__ == "__main__":
    main()
