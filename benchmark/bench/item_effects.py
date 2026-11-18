"""How much each survey item moves a model's PSF judgement, next to how much it moves the experts.

For every item: change only that item over its options on 300 random backgrounds and average the model's
expected level (0 = best). `range` = max - min of those averages; 0 means the model ignores the item.
Expert effect: mean expert level per option in data/survey/responses.jsonl (options are randomised, so the
difference is the item's own effect).
"""
from __future__ import annotations

import json
import os
import random
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

from bench.spec import LEVELS, ROOT, VENDOR

warnings.filterwarnings("ignore")
sys.path.insert(0, str(VENDOR))


def expected_levels(pid, cond_rows, model_path, bundle):
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
    order = [classes.index(lv) for lv in LEVELS[pid]]
    return proba[:, order] @ np.arange(len(order))


def expected_levels_raw(pid, cond_rows, model_path):
    from catboost import CatBoostClassifier

    m = CatBoostClassifier()
    m.load_model(str(model_path))
    keys = m.feature_names_
    proba = m.predict_proba(pd.DataFrame([[str(c[k]) for k in keys] for c in cond_rows], columns=keys))
    classes = [str(c) for c in m.classes_]
    order = [classes.index(lv) for lv in LEVELS[pid]]
    return proba[:, order] @ np.arange(len(order))


def fidelity(out: dict, name: str, min_expert=0.10, flat=0.03) -> dict:
    """How faithfully a model's item effects follow the experts' (items the experts clearly react to)."""
    ignored, reversed_, devs = [], [], []
    for key, rec in out.items():
        exp = [e for e in rec["expert_mean_level"] if e is not None]
        mod = [m for m, e in zip(rec[name], rec["expert_mean_level"]) if e is not None]
        if len(exp) < 2:
            continue
        e_rng, m_rng = max(exp) - min(exp), max(mod) - min(mod)
        if e_rng >= min_expert:
            if m_rng < flat:
                ignored.append(key)
            elif (mod[-1] - mod[0]) * (exp[-1] - exp[0]) < 0:
                reversed_.append(key)
        ec, mc = np.array(exp) - np.mean(exp), np.array(mod) - np.mean(mod)
        devs.append(float(np.abs(ec - mc).mean()))
    return {"ignored": ignored, "reversed": reversed_, "mean_abs_dev": round(float(np.mean(devs)), 3)}


def main() -> None:
    from config.catboost_config import CATBOOST_CONFIG

    spec = json.loads((ROOT / "data/survey_spec.json").read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in open(ROOT / "data/survey/responses.jsonl", encoding="utf-8")]
    models = {"original": (VENDOR / "models", "value_web", False),
              "retrained": (ROOT / "results/expert/models_all", "value", False)}
    if not (ROOT / "results/expert/models_all" / CATBOOST_CONFIG[1]["filename"]).exists():
        models["retrained"] = (ROOT / "results/expert/models_train", "value", False)
    rng = random.Random(5)
    out = {}
    for pid in range(1, 9):
        items = spec[str(pid)]["items"]
        bg = [{it["key"]: rng.randrange(len(it["options"])) for it in items} for _ in range(300)]
        rr = [r for r in rows if r["psf"] == pid]
        y = np.array([LEVELS[pid].index(r["answer"]) for r in rr])
        for it in items:
            idx = np.array([r["choice"][it["key"]] for r in rr])
            exp = [float(y[idx == k].mean()) if (idx == k).any() else None for k in range(len(it["options"]))]
            rec = {"label": it["label"], "expert_mean_level": [round(e, 3) if e is not None else None for e in exp]}
            for name, (mdir, field, _) in models.items():
                gmm = mdir / "gmm" / f"psf{pid}_gmm.joblib"
                bundle = joblib.load(gmm) if pid in (2, 4, 6, 8) else None
                means = []
                for k in range(len(it["options"])):
                    conds = [{q["key"]: q["options"][(k if q["key"] == it["key"] else b[q["key"]])][field]
                              for q in items} for b in bg]
                    means.append(float(expected_levels(pid, conds, mdir / CATBOOST_CONFIG[pid]["filename"],
                                                       bundle).mean()))
                rec[name] = [round(v, 3) for v in means]
            raw_path = models["retrained"][0] / f"raw_items_psf{pid}.cbm"
            rec["retrained_raw"] = [round(float(expected_levels_raw(pid, [
                {q["key"]: q["options"][(k if q["key"] == it["key"] else b[q["key"]])]["value"] for q in items}
                for b in bg], raw_path).mean()), 3) for k in range(len(it["options"]))]
            app_dir = ROOT / "results/expert/models_all_app"  # the models the fix branch ships
            if (app_dir / "작업부하_catboost.cbm").exists():
                from bench.train_catboost import load_app_config, per_item_frame
                from catboost import CatBoostClassifier
                cfg = load_app_config(ROOT / "vendor/Railway_hep-fixed")
                ma = CatBoostClassifier()
                ma.load_model(str(app_dir / cfg[pid]["filename"]))
                classes = [str(c) for c in ma.classes_]
                order = [classes.index(lv) for lv in LEVELS[pid]]
                vals = []
                for k in range(len(it["options"])):
                    conds = [{q["key"]: q["options"][(k if q["key"] == it["key"] else b[q["key"]])]["value"]
                              for q in items} for b in bg]
                    pr = ma.predict_proba(per_item_frame(cfg[pid]["features"], conds))[:, order]
                    vals.append(round(float((pr @ np.arange(len(order))).mean()), 3))
                rec["fixed_app"] = vals
            out[it["key"]] = rec
            fmt = lambda v: " ".join(f"{x:.2f}" if x is not None else " -- " for x in v)
            print(f"{it['key']:9s} {it['label'][:14]:14s} expert[{fmt(rec['expert_mean_level'])}] "
                  f"original[{fmt(rec['original'])}] retrained[{fmt(rec['retrained'])}] "
                  f"raw[{fmt(rec['retrained_raw'])}]")
    names = ["original", "retrained", "retrained_raw"] + (["fixed_app"] if "fixed_app" in next(iter(out.values())) else [])
    summary = {name: fidelity(out, name) for name in names}
    for name, f in summary.items():
        print(f"{name:14s} ignored={len(f['ignored'])} {f['ignored']} reversed={len(f['reversed'])} "
              f"{f['reversed']} mean_abs_dev={f['mean_abs_dev']}")
    (ROOT / "results/expert/item_effects.json").write_text(
        json.dumps({"items": out, "fidelity": summary}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
