"""Turn the fine-tuning job's scores into run files for bench.score_expert.

  uv run python -m bench.ft_import results/expert/ft/out-<job>
writes results/expert/raw/openjev-base-logprob.jsonl (before training) and openjev-lora-ft.jsonl (after).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from bench.spec import ROOT


def main() -> None:
    out_dir = Path(sys.argv[1])
    items = {json.loads(x)["id"]: json.loads(x) for x in open(ROOT / "data/expert/llm_items.jsonl", encoding="utf-8")}
    for src, run in [("scores_base.json", "openjev-base-logprob"), ("scores_lora.json", "openjev-lora-ft")]:
        p = out_dir / src
        if not p.exists():
            print("missing", p)
            continue
        scores = json.loads(p.read_text(encoding="utf-8"))
        with open(ROOT / "results/expert/raw" / f"{run}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for i, s in scores.items():
                it = items[i]
                f.write(json.dumps({"id": i, "psf": it["psf"], "label": it["label"], "pred": s["pred"],
                                    "logp": s["logp"], "run": run}, ensure_ascii=False) + "\n")
        print(run, len(scores))
    meta = out_dir / "train_meta.json"
    if meta.exists():
        print("train meta:", meta.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
