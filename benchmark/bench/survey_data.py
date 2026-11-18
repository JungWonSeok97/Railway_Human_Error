"""Convert the Korail worker survey export into a de-identified label file.

The raw export (data/raw/*.txt, git-ignored) holds names, employee numbers, contact details, birth dates,
health-check values and employee card numbers. Only the fields the benchmark needs are kept:

  data/survey/responses.jsonl   one line per answered PSF question
    {"id", "rid" (respondent pseudonym), "job", "years_band", "round", "psf", "choice", "answer"}

`choice` maps each item key (F.p.S.k) to the index of the chosen option in data/survey_spec.json, so the
condition sentences can be rebuilt without storing them per row.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from bench.spec import LEVELS, ROOT

# survey export label -> label used in Railway_hep input.html (same item, longer wording in the export)
LABEL_ALIAS = {
    "(직무)절차의 적절성(완전성 포함)": "(직무)절차의 적절성",
    "(직무)절차 기술(description)의 복잡성": "(직무)절차 기술의 복잡성",
    "시스템 지식(원리적 지식)의 숙련도": "시스템 지식의 숙련도",
    "상황에 대한 경험 정도(자주 또는 드물게 겪는지)": "상황에 대한 경험 정도",
}
TEXT_ALIAS = {"10m^3": "10m³"}

KEEP_JOBS = {"기관사", "관제사", "작업자", "승무원"}


def years_band(raw: str) -> str:
    try:
        y = float(str(raw).strip())
    except ValueError:
        return "unknown"
    for hi, name in [(5, "0-4"), (10, "5-9"), (20, "10-19")]:
        if y < hi:
            return name
    return "20+"


def convert(raw_path: Path) -> list[dict]:
    spec = json.loads((ROOT / "data/survey_spec.json").read_text(encoding="utf-8"))
    recs = json.loads(raw_path.read_text(encoding="utf-8"))
    rid_of: dict[tuple, str] = {}
    rows, problems = [], collections.Counter()
    for n, r in enumerate(recs):
        person = (r.get("name"), r.get("employee_id"))
        rid = rid_of.setdefault(person, f"R{len(rid_of) + 1:04d}")
        job = r.get("job") if r.get("job") in KEEP_JOBS else "기타"
        for j, q in enumerate(r.get("questions") or []):
            pid = int(q["id"])
            items = spec[str(pid)]["items"]
            by_label = {it["label"]: it for it in items}
            choice = {}
            for label, text in q["conditions"].items():
                it = by_label.get(LABEL_ALIAS.get(label, label))
                if it is None:
                    problems["unknown label"] += 1
                    continue
                for a, b in TEXT_ALIAS.items():
                    text = text.replace(a, b)
                texts = [o["text"] for o in it["options"]]
                if text not in texts:
                    problems["unknown option text"] += 1
                    continue
                choice[it["key"]] = texts.index(text)
            if len(choice) != len(items):
                problems["incomplete conditions"] += 1
                continue
            if q.get("answer") not in LEVELS[pid]:
                problems["unknown answer"] += 1
                continue
            rows.append({"id": f"S{n:05d}-{j}", "rid": rid, "job": job, "years_band": years_band(r.get("years")),
                         "round": r.get("round"), "psf": pid, "choice": choice, "answer": q["answer"]})
    print(f"{len(recs)} records, {len(rid_of)} respondents -> {len(rows)} labelled questions; skipped: {dict(problems)}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="?", default=None, help="raw export; default: the only .txt in data/raw/")
    a = ap.parse_args()
    raw = Path(a.raw) if a.raw else next((ROOT / "data/raw").glob("*.txt"))
    rows = convert(raw)
    out = ROOT / "data/survey/responses.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
