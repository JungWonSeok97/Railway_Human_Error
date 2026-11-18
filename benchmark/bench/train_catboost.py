"""Retrain Railway_hep's PSF models from the expert survey labels, in Railway_hep's own model format.

Fixes two causes of the original models' problems:
  * train/serve skew: features are built with Railway_hep's own `build_catboost_features`, from the same
    0/1/2 values the (fixed) web form sends, so binary items are 0/2 in training too (the original
    models were trained with 0/1 and never saw the 2 the app sends);
  * the GMM bundles are refit on the same data with the notebook's settings (env_preprocess_v1.ipynb).

Hyper-parameters copied from the original .cbm files: 500 iterations, depth 6, learning rate 0.05, MultiClass.

  uv run python -m bench.train_catboost --rows train --out results/expert/models_train   # evaluation
  uv run python -m bench.train_catboost --rows all   --out results/expert/models_all     # deployment
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from bench.spec import ROOT, VENDOR

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, str(VENDOR))

GMM_PSF_IDS = (2, 4, 6, 8)


def option_values(spec: dict, pid: int, choice: dict) -> dict[str, int]:
    """Values the fixed web form sends (2 = worst, binary items 0/2)."""
    return {it["key"]: it["options"][choice[it["key"]]]["value"] for it in spec[str(pid)]["items"]}


def fit_gmm_bundle(pid: int, cond_rows: list[dict], seed: int = 42) -> dict:
    """Refit the composite-feature GMMs exactly as env_preprocess_v1.ipynb does."""
    from sklearn.mixture import GaussianMixture
    from config.catboost_config import CATBOOST_CONFIG

    original = joblib.load(VENDOR / "models" / "gmm" / f"psf{pid}_gmm.joblib")
    bundle = {}
    for name, keys in CATBOOST_CONFIG[pid]["features"]:
        if len(keys) < 2:
            continue
        X = np.array([[r[k] for k in keys] for r in cond_rows], dtype=float)
        gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=seed, n_init=20).fit(X)
        order = np.argsort(gmm.means_.mean(axis=1))  # cluster with the lowest mean value -> state 0
        label_map = {int(c): int(i) for i, c in enumerate(order)}
        fill = {col: float(pd.Series(X[:, j]).mode().iloc[0]) for j, col in enumerate(keys)}
        group_cols = original.get(name, {}).get("group_cols") or list(keys)
        bundle[name] = {"gmm": gmm, "label_map": label_map, "group_cols": list(group_cols), "fill_values": fill}
    return bundle


def build_features(pid: int, cond_rows: list[dict], bundle: dict | None) -> pd.DataFrame:
    """Run Railway_hep's own feature builder with the given GMM bundle injected into its cache."""
    import preprocessing.catboost_features as cf
    from config.catboost_config import CATBOOST_CONFIG

    if pid in GMM_PSF_IDS:
        cf.gmm_bundles[pid] = bundle
    names = [n for n, _ in CATBOOST_CONFIG[pid]["features"]]
    with open(os.devnull, "w") as devnull:  # the builder prints per call
        old, sys.stdout = sys.stdout, devnull
        try:
            rows = [cf.build_catboost_features(pid, c) for c in cond_rows]
        finally:
            sys.stdout = old
    return pd.DataFrame(rows, columns=names)


def train_psf(pid: int, cond_rows: list[dict], labels: list[str], seed: int = 42):
    from catboost import CatBoostClassifier

    bundle = fit_gmm_bundle(pid, cond_rows, seed) if pid in GMM_PSF_IDS else None
    X = build_features(pid, cond_rows, bundle)
    model = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, loss_function="MultiClass",
                               cat_features=list(range(X.shape[1])), random_seed=seed, verbose=False,
                               allow_writing_files=False)
    model.fit(X, labels)
    return model, bundle


def train_raw_items(pid: int, cond_rows: list[dict], labels: list[str], seed: int = 42):
    """Variant without GMM composites: every survey item is its own categorical feature."""
    from catboost import CatBoostClassifier

    keys = sorted(cond_rows[0], key=lambda k: int(k.split(".")[-1]))
    X = pd.DataFrame([[str(r[k]) for k in keys] for r in cond_rows], columns=keys)
    model = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, loss_function="MultiClass",
                               cat_features=list(range(len(keys))), random_seed=seed, verbose=False,
                               allow_writing_files=False)
    model.fit(X, labels)
    return model, keys


def load_app_config(app_dir: Path) -> dict:
    """CATBOOST_CONFIG of another Railway_hep checkout, loaded by file path (no sys.path clash)."""
    import importlib.util

    s = importlib.util.spec_from_file_location("app_catboost_config", app_dir / "config" / "catboost_config.py")
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return mod.CATBOOST_CONFIG


def per_item_frame(features, cond_rows) -> pd.DataFrame:
    """Same strings preprocessing.catboost_features.build_catboost_features makes for single-item features."""
    return pd.DataFrame([[str(c[keys[0]]) for _, keys in features] for c in cond_rows],
                        columns=[n for n, _ in features])


def train_app(a, spec, rows) -> None:
    """Train models for an app whose config uses one survey item per feature (the fixed Railway_hep)."""
    from catboost import CatBoostClassifier

    cfg = load_app_config(Path(a.app))
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    for pid in range(1, 9):
        feats = cfg[pid]["features"]
        if any(len(k) != 1 for _, k in feats):
            raise SystemExit(f"F.{pid}: --app expects one survey item per feature")
        rr = [r for r in rows if r["psf"] == pid]
        X = per_item_frame(feats, [option_values(spec, pid, r["choice"]) for r in rr])
        model = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, loss_function="MultiClass",
                                   cat_features=list(range(X.shape[1])), random_seed=42, verbose=False,
                                   allow_writing_files=False)
        model.fit(X, [r["answer"] for r in rr])
        model.save_model(str(out / cfg[pid]["filename"]))
        print(f"F.{pid}: trained on {len(rr)} rows -> {cfg[pid]['filename']}", flush=True)


def main() -> None:
    from config.catboost_config import CATBOOST_CONFIG

    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", choices=["train", "all"], default="train")
    ap.add_argument("--out", default="results/expert/models_train")
    ap.add_argument("--app", default="", help="Railway_hep checkout whose config to train for (per-item)")
    a = ap.parse_args()
    spec = json.loads((ROOT / "data/survey_spec.json").read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in open(ROOT / "data/survey/responses.jsonl", encoding="utf-8")]
    if a.rows == "train":
        train_ids = set(json.loads((ROOT / "data/expert/split.json").read_text(encoding="utf-8"))["train"])
        rows = [r for r in rows if r["rid"] in train_ids]
    if a.app:
        train_app(a, spec, rows)
        return
    out = ROOT / a.out
    (out / "gmm").mkdir(parents=True, exist_ok=True)
    for pid in range(1, 9):
        rr = [r for r in rows if r["psf"] == pid]
        conds = [option_values(spec, pid, r["choice"]) for r in rr]
        labels = [r["answer"] for r in rr]
        model, bundle = train_psf(pid, conds, labels)
        model.save_model(str(out / CATBOOST_CONFIG[pid]["filename"]))
        if bundle is not None:
            joblib.dump(bundle, out / "gmm" / f"psf{pid}_gmm.joblib")
        raw_model, keys = train_raw_items(pid, conds, labels)
        raw_model.save_model(str(out / f"raw_items_psf{pid}.cbm"))
        print(f"F.{pid}: trained on {len(rr)} rows -> {CATBOOST_CONFIG[pid]['filename']}")


if __name__ == "__main__":
    main()
