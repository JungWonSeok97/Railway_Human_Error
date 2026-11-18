"""Write the LoRA fine-tuning set for OpenJev: every training-respondent answer as (prompt -> answer).

ft/train.jsonl  {"messages": [...], "target": "{\"level\": \"<label>\"}"}         21,380 rows
ft/eval.jsonl   {"id", "psf", "label", "messages", "candidates": [<target per level>]}  1,288 rows
The prompt is the zero-shot prompt the chat runs use (bench.llm_eval.build_messages, no examples).
"""
from __future__ import annotations

import json

from bench.llm_eval import build_messages
from bench.reference import load_spec
from bench.spec import LEVELS, ROOT


def target(label: str) -> str:
    return json.dumps({"level": label}, ensure_ascii=False)


def main() -> None:
    spec = load_spec()
    split = json.loads((ROOT / "data/expert/split.json").read_text(encoding="utf-8"))
    train_ids = set(split["train"])
    out = ROOT / "ft"
    out.mkdir(exist_ok=True)
    n = 0
    with open(out / "train.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in map(json.loads, open(ROOT / "data/survey/responses.jsonl", encoding="utf-8")):
            if r["rid"] not in train_ids:
                continue
            msgs = build_messages(spec[str(r["psf"])], r, [])
            f.write(json.dumps({"messages": msgs, "target": target(r["answer"])}, ensure_ascii=False) + "\n")
            n += 1
    m = 0
    with open(out / "eval.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for it in map(json.loads, open(ROOT / "data/expert/llm_items.jsonl", encoding="utf-8")):
            msgs = build_messages(spec[str(it["psf"])], it, [])
            f.write(json.dumps({"id": it["id"], "psf": it["psf"], "label": it["label"], "messages": msgs,
                                "candidates": [target(lv) for lv in LEVELS[it["psf"]]]}, ensure_ascii=False) + "\n")
            m += 1
    print(f"train {n}, eval {m} -> {out}")


if __name__ == "__main__":
    main()
