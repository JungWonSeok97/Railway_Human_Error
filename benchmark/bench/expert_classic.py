"""Predict the expert test set with the non-LLM models; one run file per model in results/expert/raw/.

Runs (all on every test-respondent row, data/expert/split.json):
  baseline-majority            most frequent level per PSF among train respondents
  baseline-item-mean           mean item value (0 good .. 2 bad) cut into equal bins; uses no data
  baseline-logreg              multinomial logistic regression on one-hot items, train respondents
  catboost-original-deployed   Railway_hep models as the web app runs them today (reversed options included)
  catboost-original-inputfix   same models after the input-order fix only (binary items still sent as 2)
  catboost-original-as-trained same models with binary items sent as 1, the value they were trained with
  catboost-retrained           GMM + CatBoost retrained on train respondents with bench.train_catboost
  catboost-retrained-raw       CatBoost on the raw items (no GMM composites), train respondents
  catboost-fixed-app           the per-item models the fix branch ships, trained on train respondents only
Also writes results/expert/consensus.json: how often other experts agree on an identical scenario.
"""
from __future__ import annotations

import collections
import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from bench.spec import LEVELS, ROOT, VENDOR

warnings.filterwarnings("ignore")
sys.path.insert(0, str(VENDOR))


def load():
    spec = json.loads((ROOT / "data/survey_spec.json").read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in open(ROOT / "data/survey/responses.jsonl", encoding="utf-8")]
    split = json.loads((ROOT / "data/expert/split.json").read_text(encoding="utf-8"))
    train, test = set(split["train"]), set(split["test"])
    return spec, [r for r in rows if r["rid"] in train], [r for r in rows if r["rid"] in test], rows


def values(spec, pid, choice, field="value", binary_as_one=False):
    out = {}
    for it in spec[str(pid)]["items"]:
        v = it["options"][choice[it["key"]]][field]
        if binary_as_one and len(it["options"]) == 2 and v == 2:
            v = 1
        out[it["key"]] = v
    return out


def catboost_predict(pid, cond_rows, model_path, bundle):
    import preprocessing.catboost_features as cf
    from catboost import CatBoostClassifier
    from config.catboost_config import CATBOOST_CONFIG

    if bundle is not None:
        cf.gmm_bundles[pid] = bundle
    names = [n for n, _ in CATBOOST_CONFIG[pid]["features"]]
    with open(os.devnull, "w") as devnull:
        old, sys.stdout = sys.stdout, devnull
        try:
            X = pd.DataFrame([cf.build_catboost_features(pid, c) for c in cond_rows], columns=names)
        finally:
            sys.stdout = old
    m = CatBoostClassifier()
    m.load_model(str(model_path))
    proba = m.predict_proba(X)
    classes = [str(c) for c in m.classes_]
    return [classes[i] for i in proba.argmax(1)]


def write_run(name, rows, preds):
    out = ROOT / "results/expert/raw"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r, p in zip(rows, preds):
            f.write(json.dumps({"id": r["id"], "psf": r["psf"], "label": r["answer"], "pred": p, "run": name},
                               ensure_ascii=False) + "\n")


def consensus(test_rows, all_rows):
    """Leave-the-respondent-out majority of OTHER experts who judged the identical scenario."""
    by_scen = collections.defaultdict(list)
    for r in all_rows:
        by_scen[(r["psf"], tuple(sorted(r["choice"].items())))].append((r["rid"], r["answer"]))
    res, agree_pairs = {}, collections.Counter()
    for r in test_rows:
        others = [a for rid, a in by_scen[(r["psf"], tuple(sorted(r["choice"].items())))] if rid != r["rid"]]
        if len(others) >= 3:
            c = collections.Counter(others).most_common()
            top = [lv for lv, n in c if n == c[0][1]]
            pred = top[0] if len(top) == 1 else sorted(top, key=LEVELS[r["psf"]].index)[len(top) // 2]
            res[r["id"]] = pred
        if others:
            agree_pairs[r["psf"]] += 0  # make sure the key exists
    # pairwise agreement between two different experts on the same scenario
    pair = {}
    for pid in range(1, 9):
        same = tot = 0
        for (p, _), lst in by_scen.items():
            if p != pid or len(lst) < 2:
                continue
            labels = [a for _, a in lst]
            cnt = collections.Counter(labels)
            n = len(labels)
            same += sum(k * (k - 1) for k in cnt.values())
            tot += n * (n - 1)
        pair[pid] = round(same / tot, 3) if tot else None
    return res, pair


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder
    from config.catboost_config import CATBOOST_CONFIG

    spec, train, test, all_rows = load()
    cons_rows = [{"id": c["id"], "psf": c["psf"], "choice": c["choice"], "answer": c["label"], "rid": None}
                 for c in map(json.loads, open(ROOT / "data/expert/consensus_items.jsonl", encoding="utf-8"))]
    targets = test + cons_rows  # individual test answers + consensus scenarios
    preds = collections.defaultdict(dict)
    for pid in range(1, 9):
        tr = [r for r in train if r["psf"] == pid]
        te = [r for r in targets if r["psf"] == pid]
        lv = LEVELS[pid]
        k = len(lv)
        maj = collections.Counter(r["answer"] for r in tr).most_common(1)[0][0]
        preds["baseline-majority"].update({r["id"]: maj for r in te})
        for r in te:
            v = list(values(spec, pid, r["choice"]).values())
            preds["baseline-item-mean"][r["id"]] = lv[min(int(np.mean(v) / 2 * k), k - 1)]
        keys = [it["key"] for it in spec[str(pid)]["items"]]
        enc = OneHotEncoder(handle_unknown="ignore").fit([[r["choice"][q] for q in keys] for r in tr])
        lr = LogisticRegression(max_iter=3000).fit(enc.transform([[r["choice"][q] for q in keys] for r in tr]),
                                                   [r["answer"] for r in tr])
        for r, p in zip(te, lr.predict(enc.transform([[r["choice"][q] for q in keys] for r in te]))):
            preds["baseline-logreg"][r["id"]] = str(p)
        orig_bundle = joblib.load(VENDOR / "models/gmm" / f"psf{pid}_gmm.joblib") if pid in (2, 4, 6, 8) else None
        orig_model = VENDOR / "models" / CATBOOST_CONFIG[pid]["filename"]
        for name, kw in [("catboost-original-deployed", {"field": "value_web"}),
                         ("catboost-original-inputfix", {"field": "value"}),
                         ("catboost-original-as-trained", {"field": "value", "binary_as_one": True})]:
            conds = [values(spec, pid, r["choice"], **kw) for r in te]
            for r, p in zip(te, catboost_predict(pid, conds, orig_model, orig_bundle)):
                preds[name][r["id"]] = p
        mdir = ROOT / "results/expert/models_train"
        new_bundle = joblib.load(mdir / "gmm" / f"psf{pid}_gmm.joblib") if pid in (2, 4, 6, 8) else None
        conds = [values(spec, pid, r["choice"]) for r in te]
        for r, p in zip(te, catboost_predict(pid, conds, mdir / CATBOOST_CONFIG[pid]["filename"], new_bundle)):
            preds["catboost-retrained"][r["id"]] = p
        from catboost import CatBoostClassifier
        raw = CatBoostClassifier()
        raw.load_model(str(mdir / f"raw_items_psf{pid}.cbm"))
        rk = sorted(keys, key=lambda q: int(q.split(".")[-1]))
        Xr = pd.DataFrame([[str(values(spec, pid, r["choice"])[q]) for q in rk] for r in te], columns=rk)
        for r, p in zip(te, raw.predict(Xr).ravel()):
            preds["catboost-retrained-raw"][r["id"]] = str(p)
        app_models = ROOT / "results/expert/models_train_app"  # what the fix branch ships, train split only
        if app_models.exists():
            from bench.train_catboost import load_app_config, per_item_frame
            cfg = load_app_config(ROOT / "vendor/Railway_hep-fixed")
            ma = CatBoostClassifier()
            ma.load_model(str(app_models / cfg[pid]["filename"]))
            Xa = per_item_frame(cfg[pid]["features"], [values(spec, pid, r["choice"]) for r in te])
            for r, p in zip(te, ma.predict(Xa).ravel()):
                preds["catboost-fixed-app"][r["id"]] = str(p)
        print(f"F.{pid}: {len(te)} test rows predicted")
    for name, d in preds.items():
        write_run(name, targets, [d[r["id"]] for r in targets])
    cons, pair = consensus(test, all_rows)
    write_run("ceiling-other-experts", [r for r in test if r["id"] in cons], [cons[r["id"]] for r in test if r["id"] in cons])
    (ROOT / "results/expert/consensus.json").write_text(json.dumps(
        {"covered_test_rows": len(cons), "pairwise_agreement_same_scenario": pair}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("consensus coverage", len(cons), "pairwise agreement", pair)


if __name__ == "__main__":
    main()
