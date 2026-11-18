"""End-to-end check of the fixed Railway_hep: form codes -> app predictor == benchmark prediction.

Sends every test-respondent scenario (up to --per-psf per PSF) the way the fixed input.html would
(option-code table + code_to_value), runs the app's own `predictors.catboost.predict_psf_catboost` in a
separate interpreter with the fixed checkout on sys.path, and compares the predicted level with the
benchmark's per-item prediction from the same model files.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from catboost import CatBoostClassifier

from bench.spec import ROOT, option_code_table
from bench.train_catboost import load_app_config, option_values, per_item_frame

APP_SIDE = r"""
import contextlib, io, json, sys
sys.path.insert(0, sys.argv[1])
from config.mappings import mappings
from predictors.catboost import predict_psf_catboost
out = []
for line in sys.stdin:
    q = json.loads(line)
    with contextlib.redirect_stdout(io.StringIO()):
        r = predict_psf_catboost(q["psf"], q["conditions"])
    out.append(mappings[q["psf"]]["label_map"][int(r["predicted_label"])])
print(json.dumps(out, ensure_ascii=False))
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", default="vendor/Railway_hep-fixed")
    ap.add_argument("--per-psf", type=int, default=150)
    a = ap.parse_args()
    app = (ROOT / a.app).resolve()
    spec = json.loads((ROOT / "data/survey_spec.json").read_text(encoding="utf-8"))
    table = option_code_table((app / "input.html").read_text(encoding="utf-8-sig"))
    cfg = load_app_config(app)
    split = json.loads((ROOT / "data/expert/split.json").read_text(encoding="utf-8"))
    test = set(split["test"])
    rows = [r for r in map(json.loads, open(ROOT / "data/survey/responses.jsonl", encoding="utf-8"))
            if r["rid"] in test]
    queries, ours = [], []
    for pid in range(1, 9):
        rr = [r for r in rows if r["psf"] == pid][: a.per_psf]
        m = CatBoostClassifier()
        m.load_model(str(app / "models" / cfg[pid]["filename"]))
        X = per_item_frame(cfg[pid]["features"], [option_values(spec, pid, r["choice"]) for r in rr])
        ours += [str(p) for p in m.predict(X).ravel()]
        for r in rr:
            conds = {}
            for it in spec[str(pid)]["items"]:
                i = r["choice"][it["key"]]
                n = table[it["key"]][i] if it["key"] in table else i + 1
                conds[it["key"]] = f"{it['key']}.{n}"
            queries.append({"psf": pid, "conditions": conds})
    res = subprocess.run([sys.executable, "-c", APP_SIDE, str(app)], input="\n".join(map(json.dumps, queries)),
                         capture_output=True, text=True, encoding="utf-8", cwd=app)
    if res.returncode:
        raise SystemExit(res.stderr[-2000:])
    theirs = json.loads(res.stdout.strip().splitlines()[-1])
    diff = [(q["psf"], a_, b_) for q, a_, b_ in zip(queries, ours, theirs) if a_ != b_]
    print(f"{len(queries)} scenarios through the fixed app; mismatches with the benchmark: {len(diff)}")
    for d in diff[:10]:
        print("  ", d)
    if diff:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
