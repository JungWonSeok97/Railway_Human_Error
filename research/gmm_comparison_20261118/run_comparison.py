"""Controlled PSF 2/4/6/8 feature-representation comparison.

Read-only access to Railway_hep; outputs go only to this research directory.
No respondent identifiers or survey-level records are exported.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import platform
import time
import warnings
from pathlib import Path

import catboost
import joblib
import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1] / "Railway_hep"
NAMES = {2: "장비 및 기기", 4: "교육 및 훈련", 6: "환경", 8: "복잡성"}
GROUPS = {
    2: {
        "Equipment_Accessibility": ["장비 가용성(접근성)"],
        "Equipment_Quality": ["작업 도구의 적절성", "장비의 사용성"],
        "System_Support": ["컴퓨터 지원 시스템의 적절성", "타 기기와의 연계성"],
        "Documentation_Quality": ["장비 사용 매뉴얼의 질"],
        "Maintenance_Support": ["유지보수의 적절성"],
    },
    4: {
        "Training_Program": ["교육/훈련 프로그램 적절성"],
        "Training_Frequency": ["교육 주기(빈도)", "직무능력 평가 빈도"],
        "Experience_Level": ["상황에 대한 경험 정도(자주 또는 드물게 겪는지)"],
        "Technical_Knowledge": ["시스템 지식(원리적 지식)의 숙련도", "장비 사용 지식의 숙련도"],
        "Procedural_Knowledge": ["절차적 지식의 숙련도"],
    },
    6: {
        "Thermal_Comfort": ["실내 온도", "실내 습도", "통풍"],
        "Visual_Environment": ["업무공간 가시성", "안개", "황사"],
        "Surface_Safety": ["마찰계수", "비"],
        "Workspace": ["작업공간 크기"],
        "Acoustic_Environment": ["소음"],
        "Outdoor_Climate": ["실외 온도", "실외 습도", "풍속", "낙뢰", "비"],
    },
    8: {
        "Goal_Clarity": ["의사결정 목표의 명확성"],
        "Decision_Complexity": ["의사결정 대안의 수", "의사결정 판단 기준의 수"],
        "Concurrent_Decision_Load": ["(동시에 처리 되어야 하는)의사결정 수"],
        "Information_Complexity": ["관측되는 정보의 양", "(관측이 불가능하여)추론해야하는 정보의 양"],
        "Information_Uncertainty": ["정보의 불확실성"],
    },
}
PARAMS = dict(
    iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=20,
    grow_policy="Lossguide", max_leaves=32, min_data_in_leaf=2,
    random_strength=10, bagging_temperature=2, loss_function="MultiClass",
    eval_metric="TotalF1", random_seed=42, verbose=False,
    thread_count=2, allow_writing_files=False,
)
GMM_PARAMS = dict(n_components=3, covariance_type="full", random_state=42, n_init=20)
VARIANTS = ["gmm", "raw_numeric", "raw_text", "raw_text_plus_gmm"]
NB_SHA256 = "e4b8fd330b1cc193656e535e0fc2146763d831256e11fe39f04d0cd545461d48"
PREPROCESS_CELLS = [4, 5, 11, 13, 16, 18, 23, 30, 35, 50, 51, 52, 53, 54, 55, 56]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def protected_manifest():
    paths = [
        PROJECT / "철도_코레일 종사자 설문조사 결과.txt",
        PROJECT / "processed_data/subpsf_level_mapping.xlsx",
        PROJECT / "CatBoost_Result/preprocess_v1.4.ipynb",
        PROJECT / "CatBoost_Result/analysis_v1.5.ipynb",
    ]
    paths += sorted((PROJECT / "models").rglob("*.cbm"))
    paths += sorted((PROJECT / "models").rglob("*.joblib"))
    return {str(p.relative_to(PROJECT)).replace("\\", "/"): sha256(p) for p in paths}


def load_answers():
    """Reuse the original quality/identity rules without copying personal data.

    Only the pinned, reviewed, non-writing cells below are executed. Notebook
    export/training cells are never run. Identity fields remain in memory only.
    """
    nb_path = PROJECT / "CatBoost_Result/preprocess_v1.4.ipynb"
    if sha256(nb_path) != NB_SHA256:
        raise RuntimeError("Preprocessing notebook changed; review cells before reproducing.")
    notebook = json.loads(nb_path.read_text(encoding="utf-8"))
    records = json.loads((PROJECT / "철도_코레일 종사자 설문조사 결과.txt").read_text(encoding="utf-8"))
    df = pd.DataFrame(records)
    # The TXT contains both already-parsed lists and JSON-encoded lists.
    df["questions"] = df.questions.map(
        lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
    )
    namespace = {"pd": pd, "json": json, "df": df}
    with contextlib.redirect_stdout(io.StringIO()):
        for index in PREPROCESS_CELLS:
            if index == 30:
                namespace["answer_df"] = namespace["df_answer"].copy()
            code = "".join(notebook["cells"][index]["source"])
            exec(compile(code, f"original-preprocessing-cell-{index}", "exec"), namespace)
    answer = namespace["answer_df_new_quality"]
    # Discard identifying columns before returning to the experimental code.
    return answer[["PSF", "PSF_name", "answer", "conditions", "quality_95"]].copy()


def make_data(answers, psf):
    all_rows = answers.loc[answers.PSF == psf]
    filtered = all_rows.loc[(all_rows.quality_95 == "성실") & all_rows.answer.notna()]
    raw = pd.DataFrame(filtered.conditions.tolist())
    raw = raw[sorted(raw.columns)].reset_index(drop=True)
    if raw.isna().any().any():
        raise ValueError("Unexpected missing raw inputs; specify fold-local handling first.")
    raw = raw.astype(str)
    y = filtered.answer.reset_index(drop=True).astype(str)
    mapping = pd.read_excel(PROJECT / "processed_data/subpsf_level_mapping.xlsx")
    mapping = mapping.loc[mapping.PSF_name == NAMES[psf]]
    numeric = pd.DataFrame(index=raw.index)
    for col in raw:
        subset = mapping.loc[mapping["Sub-PSF"] == col]
        level_map = subset.set_index("Sub-PSF LV.")["level"].to_dict()
        numeric[col] = raw[col].map(level_map)
        if numeric[col].isna().any():
            raise ValueError(f"Unmapped values in PSF{psf} {col}")
        numeric[col] = numeric[col].astype(int)
    grouped_columns = set(sum(GROUPS[psf].values(), []))
    assert set(raw.columns) == grouped_columns
    scenario_text = raw.agg("|".join, axis=1)
    assert not raw.map(lambda v: "|" in v).any().any()
    groups = pd.factorize(scenario_text, sort=False)[0]
    meta = {
        "name": NAMES[psf], "n_raw_full": len(all_rows), "n_quality95": len(y),
        "full_label_counts": {k: int(v) for k, v in all_rows.answer.value_counts().items()},
        "label_counts": {k: int(v) for k, v in y.value_counts().items()},
        "raw_features": list(raw.columns), "n_raw_features": raw.shape[1],
        "n_gmm_features": len(GROUPS[psf]), "n_original_scenarios": len(set(groups)),
        "n_numeric_scenarios": len(numeric.drop_duplicates()),
        "constant_raw_features": [c for c in raw if raw[c].nunique() == 1],
        "raw_levels": {c: int(raw[c].nunique()) for c in raw},
        "numeric_levels": {c: int(numeric[c].nunique()) for c in numeric},
        "dataset_fingerprint": hashlib.sha256(
            json.dumps({"x": raw.values.tolist(), "y": y.tolist()}, ensure_ascii=False).encode()
        ).hexdigest(),
    }
    return raw, numeric, y, groups, meta


def transform_gmm(numeric, psf, fitted=None):
    """Fit on this input if fitted is None; otherwise only transform it."""
    training = fitted is None
    fitted = {} if training else fitted
    result = pd.DataFrame(index=numeric.index)
    for feature, cols in GROUPS[psf].items():
        x = numeric[cols].to_numpy(dtype=float)
        if len(cols) == 1:
            result[feature] = x[:, 0].astype(int).astype(str)
            continue
        if training:
            model = GaussianMixture(**GMM_PARAMS).fit(x)
            ordered = pd.DataFrame(model.means_).mean(axis=1).sort_values().index.tolist()
            fitted[feature] = (model, {cluster: state for state, cluster in enumerate(ordered)})
        model, label_map = fitted[feature]
        result[feature] = [str(label_map[int(c)]) for c in model.predict(x)]
    return result, fitted


def stored_gmm_features(numeric, psf):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sklearn.exceptions.InconsistentVersionWarning)
        bundle = joblib.load(PROJECT / f"models/gmm/psf{psf}_gmm.joblib")
    result = pd.DataFrame(index=numeric.index)
    for feature, cols in GROUPS[psf].items():
        if len(cols) == 1:
            result[feature] = numeric[cols[0]].astype(str)
        else:
            entry = bundle[feature]
            assert entry["group_cols"] == cols
            model = entry["gmm"]
            x = numeric[cols] if hasattr(model, "feature_names_in_") else numeric[cols].to_numpy()
            result[feature] = [str(entry["label_map"][int(c)]) for c in model.predict(x)]
    return result


def aligned_probabilities(model, x, labels):
    probability = model.predict_proba(x)
    order = [list(model.classes_).index(label) for label in labels]
    return probability[:, order]


def metrics(y, probabilities, labels, groups=None):
    predicted = np.asarray(labels)[probabilities.argmax(axis=1)]
    cm = confusion_matrix(y, predicted, labels=labels)
    support = cm.sum(axis=1)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(len(labels)), where=support > 0)
    predicted_support = cm.sum(axis=0)
    precision = np.divide(cm.diagonal(), predicted_support, out=np.zeros(len(labels)), where=predicted_support > 0)
    result = {
        "n": len(y), "labels": labels, "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1_score(y, predicted, labels=labels, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, probabilities, labels=labels)),
        "confusion_matrix": cm.tolist(),
        "recall": dict(zip(labels, recall.tolist())),
        "precision": dict(zip(labels, precision.tolist())),
        "prediction_counts": dict(zip(labels, predicted_support.tolist())),
        "prediction_ratios": dict(zip(labels, (predicted_support / len(y)).tolist())),
    }
    if groups is not None:
        first = np.unique(groups, return_index=True)[1]
        result["scenario_prediction_counts"] = {
            label: int((predicted[first] == label).sum()) for label in labels
        }
    return result


def check_saved_model(numeric, y, psf):
    x = stored_gmm_features(numeric, psf)
    stored = CatBoostClassifier()
    filename = NAMES[psf].replace(" ", "_") + "_catboost.cbm"
    stored.load_model(str(PROJECT / "models" / filename))
    assert list(x.columns) == stored.feature_names_
    for key, value in PARAMS.items():
        if key not in {"thread_count", "allow_writing_files", "verbose"}:
            actual = stored.get_all_params().get(key)
            if isinstance(value, float):
                assert np.isclose(actual, value), (key, actual, value)
            else:
                assert actual == value, (key, actual, value)
    recreated = CatBoostClassifier(**PARAMS)
    recreated.fit(x, y, cat_features=list(x.columns))
    labels = sorted(y.unique().tolist())
    expected = aligned_probabilities(stored, x, labels)
    actual = aligned_probabilities(recreated, x, labels)
    delta = float(np.max(np.abs(expected - actual)))
    if delta > 1e-10:
        raise AssertionError(f"PSF{psf} saved-model reproduction failed: {delta}")
    return {
        "maximum_probability_difference": delta,
        "n_stored_gmm_scenarios": len(x.drop_duplicates()),
        "in_sample_only": metrics(y, expected, labels),
    }


def compare_psf(answers, psf, out, signature, reference_cache=None):
    started = time.time()
    raw, numeric, y, groups, meta = make_data(answers, psf)
    if reference_cache and str(psf) in reference_cache["psfs"]:
        cached = reference_cache["psfs"][str(psf)]
        assert raw.to_dict("records") == cached["raw_conditions"]
        assert numeric.to_dict("records") == cached["raw_mapped"]
        assert y.tolist() == cached["y"]
        assert stored_gmm_features(numeric, psf).values.tolist() == cached["X"]
        meta["matches_prior_analysis_data"] = True
    labels = sorted(y.unique().tolist())
    print(f"PSF{psf}: {len(y)} responses, {len(set(groups))} original scenarios", flush=True)
    reproduction = check_saved_model(numeric, y, psf)
    print(f"PSF{psf}: saved model reproduced, maximum probability difference = 0", flush=True)
    result = {
        "signature": signature, "psf": psf, "data": meta, "reproduction": reproduction,
        "variants": {}, "folds": [],
    }
    all_prob = {variant: np.full((len(y), len(labels)), np.nan) for variant in VARIANTS}
    prior_prob = np.full((len(y), len(labels)), np.nan)
    fold_index = np.full(len(y), -1, dtype=int)
    folds = list(GroupKFold(n_splits=5).split(raw, y, groups))
    for fold, (train, test) in enumerate(folds, 1):
        assert not (set(groups[train]) & set(groups[test]))
        assert fold_index[test].max() == -1
        fold_index[test] = fold
        assert set(y.iloc[train]) == set(labels)
        gmm_train, fitted = transform_gmm(numeric.iloc[train], psf)
        gmm_test, _ = transform_gmm(numeric.iloc[test], psf, fitted)
        representations = {
            "gmm": (gmm_train, gmm_test),
            "raw_numeric": (numeric.iloc[train].astype(str), numeric.iloc[test].astype(str)),
            "raw_text": (raw.iloc[train], raw.iloc[test]),
            "raw_text_plus_gmm": (
                pd.concat([raw.iloc[train], gmm_train], axis=1),
                pd.concat([raw.iloc[test], gmm_test], axis=1),
            ),
        }
        priors = y.iloc[train].value_counts(normalize=True).reindex(labels).to_numpy()
        prior_prob[test] = priors
        fold_result = {
            "fold": fold, "n_train": len(train), "n_test": len(test),
            "n_train_scenarios": len(set(groups[train])),
            "n_test_scenarios": len(set(groups[test])), "scenario_overlap": 0,
            "gmm_converged": {name: bool(model.converged_) for name, (model, _) in fitted.items()},
            "variants": {},
        }
        for variant, (x_train, x_test) in representations.items():
            model = CatBoostClassifier(**PARAMS)
            model.fit(x_train, y.iloc[train], cat_features=list(x_train.columns))
            probabilities = aligned_probabilities(model, x_test, labels)
            all_prob[variant][test] = probabilities
            fold_result["variants"][variant] = metrics(y.iloc[test], probabilities, labels)
            print(
                f"PSF{psf} fold {fold}/5 {variant}: "
                f"balanced accuracy={fold_result['variants'][variant]['balanced_accuracy']:.4f}",
                flush=True,
            )
        result["folds"].append(fold_result)
        write_json(out / f"psf{psf}_partial.json", result)
    assert (fold_index > 0).all()
    for variant, probabilities in all_prob.items():
        assert np.isfinite(probabilities).all()
        assert np.allclose(probabilities.sum(axis=1), 1)
        # Every identical original scenario must have exactly the same OOF output.
        check = pd.DataFrame(probabilities).groupby(groups).nunique()
        assert (check <= 1).all().all()
        result["variants"][variant] = metrics(y, probabilities, labels, groups)
    result["training_prior_baseline"] = metrics(y, prior_prob, labels, groups)
    result["elapsed_seconds"] = time.time() - started
    # Anonymous arrays permit checking counts/confusion matrices without exporting
    # respondent identities, condition texts, or source record identifiers.
    np.savez_compressed(
        out / f"psf{psf}_oof.npz", labels=np.asarray(labels),
        y=np.asarray([labels.index(v) for v in y]), groups=groups, fold=fold_index,
        prior=prior_prob, **all_prob,
    )
    write_json(out / f"psf{psf}.json", result)
    (out / f"psf{psf}_partial.json").unlink(missing_ok=True)
    return result


def export_tables(results, out):
    overall, per_class, per_fold = [], [], []
    for result in results:
        psf = result["psf"]
        variants = {"training_prior_baseline": result["training_prior_baseline"], **result["variants"]}
        for variant, metric in variants.items():
            overall.append({
                "psf": psf, "name": NAMES[psf], "variant": variant,
                **{key: metric[key] for key in ["n", "accuracy", "balanced_accuracy", "macro_f1", "log_loss"]},
            })
            for label in metric["labels"]:
                per_class.append({
                    "psf": psf, "variant": variant, "label": label,
                    "actual_count": result["data"]["label_counts"][label],
                    **{key: metric[key][label] for key in [
                        "prediction_counts", "prediction_ratios", "recall", "precision", "scenario_prediction_counts",
                    ]},
                })
        for fold in result["folds"]:
            for variant, metric in fold["variants"].items():
                per_fold.append({
                    "psf": psf, "fold": fold["fold"], "variant": variant,
                    **{key: metric[key] for key in ["n", "accuracy", "balanced_accuracy", "macro_f1", "log_loss"]},
                })
    pd.DataFrame(overall).to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_class).to_csv(out / "class_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_fold).to_csv(out / "fold_metrics.csv", index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--psfs", nargs="+", type=int, default=[2, 4, 6, 8], choices=NAMES)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reference-cache", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    # Refuse an output location inside the original application/data directory.
    if out == PROJECT.resolve() or PROJECT.resolve() in out.parents:
        raise ValueError("Outputs must be outside Railway_hep.")
    out.mkdir(parents=True, exist_ok=True)
    before = protected_manifest()
    protocol = {
        "date": "2026-11-18", "quality": "original quality_95 == 성실",
        "split": "GroupKFold(5, shuffle=False), original text scenario, shared folds",
        "weighting": "each survey response has weight 1; no class weights",
        "prediction": "argmax; no threshold tuning or probability calibration",
        "catboost_params": PARAMS, "gmm_params": GMM_PARAMS, "groups": GROUPS,
        "variants": VARIANTS, "preprocessing_cells": PREPROCESS_CELLS,
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__, "catboost": catboost.__version__,
        },
        "protected_before": before, "runner_sha256": sha256(__file__),
    }
    signature = hashlib.sha256(json.dumps(protocol, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    protocol["signature"] = signature
    write_json(out / "protocol.json", protocol)
    answers = load_answers()
    reference = json.loads(args.reference_cache.read_text(encoding="utf-8")) if args.reference_cache else None
    results = []
    with threadpool_limits(limits=2):
        for psf in args.psfs:
            checkpoint = out / f"psf{psf}.json"
            if args.resume and checkpoint.exists():
                result = json.loads(checkpoint.read_text(encoding="utf-8"))
                assert result["signature"] == signature, "Changed protocol; use a new output directory."
            else:
                result = compare_psf(answers, psf, out, signature, reference)
            results.append(result)
            export_tables(results, out)
    after = protected_manifest()
    assert before == after, "A protected source/model file changed during the experiment."
    protocol["protected_after"] = after
    protocol["protected_files_unchanged"] = True
    write_json(out / "protocol.json", protocol)
    write_json(out / "comparison.json", {"protocol": protocol, "results": results})
    print("COMPLETE: all protected inputs/models unchanged; results in " + str(out), flush=True)


if __name__ == "__main__":
    main()
