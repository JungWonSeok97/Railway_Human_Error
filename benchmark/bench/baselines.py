"""Non-LLM baselines written in the same raw format as the LLM runs.

majority       always the most frequent CatBoost level over the whole answer space
item-mean      mean of the item values (0 good .. 2 bad) cut into equal-width bins; no data needed
logreg-9shot   multinomial logistic regression trained on the same 9 few-shot examples the LLM sees
"""
from __future__ import annotations

import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from bench.reference import load_spec
from bench.spec import LEVELS, ROOT


def vec(psf: dict, choice: dict) -> list[int]:
    return [it["options"][choice[it["key"]]]["value"] for it in psf["items"]]


def main() -> None:
    spec = load_spec()
    items = [json.loads(x) for x in open(ROOT / "data/items.jsonl", encoding="utf-8")]
    shots = [json.loads(x) for x in open(ROOT / "data/shots.jsonl", encoding="utf-8")]
    natural = json.loads((ROOT / "data/natural_distribution.json").read_text(encoding="utf-8"))
    out = {"baseline-majority": [], "baseline-item-mean": [], "baseline-logreg-9shot": []}
    models = {}
    for pid in {i["psf"] for i in items}:
        s = spec[str(pid)]
        sh = [x for x in shots if x["psf"] == pid]
        X = np.array([vec(s, x["choice"]) for x in sh])
        y = [x["label"] for x in sh]
        models[pid] = LogisticRegression(max_iter=2000).fit(X, y) if len(set(y)) > 1 else y[0]
    for it in items:
        pid, s, lv = it["psf"], spec[str(it["psf"])], LEVELS[it["psf"]]
        v = vec(s, it["choice"])
        t0 = time.perf_counter()
        base = {"id": it["id"], "psf": pid, "label": it["label"]}
        maj = max(natural[str(pid)].items(), key=lambda kv: kv[1])[0]
        out["baseline-majority"].append({**base, "pred": maj})
        k = len(lv)
        out["baseline-item-mean"].append({**base, "pred": lv[min(int(np.mean(v) / 2 * k), k - 1)]})
        m = models[pid]
        pred = m if isinstance(m, str) else str(m.predict([v])[0])
        out["baseline-logreg-9shot"].append({**base, "pred": pred,
                                             "latency_s": round(time.perf_counter() - t0, 6)})
    for name, rows in out.items():
        with open(ROOT / "results/raw" / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({**r, "run": name}, ensure_ascii=False) + "\n")
    print("wrote", ", ".join(out))


if __name__ == "__main__":
    main()
