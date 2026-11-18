"""Read-only audit of the deployed PSF models and request contracts.

Run: python -B research/code_review_20261118/run_review.py
Writes aggregate results here; does not train models or modify source data.
"""

import contextlib
import hashlib
import io
import itertools
import json
import platform
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import catboost
import sklearn
from catboost import CatBoostClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "Railway_hep"
sys.path.insert(0, str(ROOT))

from app import app
from config.catboost_config import CATBOOST_CONFIG
from config.catboost_mappings import CATBOOST_OPTION_VALUES
from config.environment_schema import ENVIRONMENT_FIELDS
from predictors.catboost import load_catboost_model
from preprocessing.catboost_features import build_catboost_features, predict_composite_feature
from training.train_environment_raw14 import load_training_data, evaluate


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(predictions):
    labels, numbers = np.unique(predictions, return_counts=True)
    return dict(zip(labels.tolist(), numbers.astype(int).tolist()))


def enumerate_psf(psf):
    fields = [field for field in CATBOOST_OPTION_VALUES if field.startswith(f"F.{psf}.")]
    sizes = [len(CATBOOST_OPTION_VALUES[field]) for field in fields]
    selections = list(itertools.product(*(range(1, size + 1) for size in sizes)))
    names = [name for name, _ in CATBOOST_CONFIG[psf]["features"]]
    inputs = [
        build_catboost_features(psf, {field: f"{field}.{choice}" for field, choice in zip(fields, row)})
        for row in selections
    ]
    frame = pd.DataFrame(inputs, columns=names)
    model = load_catboost_model(psf)
    probabilities = model.predict_proba(frame, thread_count=2)
    predicted_indices = probabilities.argmax(axis=1)
    predicted = np.asarray(model.classes_)[predicted_indices]
    probability_tensor = probabilities.reshape(*sizes, probabilities.shape[1])
    label_tensor = predicted_indices.reshape(sizes)
    input_tensor = frame.to_numpy().reshape(*sizes, len(names))
    sensitivity = {}
    for axis, field in enumerate(fields):
        # Hold every other raw response fixed and vary only this question.
        spread = np.ptp(probability_tensor, axis=axis)
        changed_labels = np.ptp(label_tensor, axis=axis) > 0
        varied = np.moveaxis(input_tensor, axis, -2).reshape(-1, sizes[axis], len(names))
        changed_inputs = np.any(varied != varied[:, :1, :], axis=(0, 1))
        sensitivity[field] = {
            "other_question_contexts": int(changed_labels.size),
            "label_changing_contexts": int(changed_labels.sum()),
            "max_probability_delta": float(spread.max()),
            "probability_changing_contexts": int((spread.max(axis=-1) > 1e-12).sum()),
            "transformed_features_that_change": [name for name, changed in zip(names, changed_inputs) if changed],
        }
    examples = {}
    for label in model.classes_:
        index = int(np.flatnonzero(predicted == label)[0]) if np.any(predicted == label) else None
        if index is not None:
            examples[str(label)] = {
                "conditions": {field: f"{field}.{choice}" for field, choice in zip(fields, selections[index])},
                "probabilities": dict(zip(model.classes_.tolist(), probabilities[index].tolist())),
            }
    report = {
        "scope": "Every Cartesian combination of the currently defined choice codes; equal scenario weighting, not real-world frequency or held-out accuracy",
        "n_scenarios": len(frame),
        "prediction_counts": counts(predicted),
        "feature_importance": dict(zip(names, model.get_feature_importance().tolist())),
        "raw_question_sensitivity": sensitivity,
        "examples": examples,
    }
    if psf == 6:
        observed = frame["안개"].eq("안개가 발생할 경우") & frame["낙뢰"].eq("낙뢰가 발생할 경우")
        report["training_supported_category_domain"] = {
            "n_scenarios": int(observed.sum()),
            "prediction_counts": counts(predicted[observed]),
            "scope": "Restricts fog and lightning to their only observed categories; still includes unseen joint scenarios",
        }
        numeric = pd.DataFrame({
            ENVIRONMENT_FIELDS[field][0]: [CATBOOST_OPTION_VALUES[field][row[index] - 1] for row in selections]
            for index, field in enumerate(fields)
        })
        bundle = joblib.load(ROOT / "models/gmm/psf6_gmm.joblib")
        previous = CatBoostClassifier().load_model(str(ROOT / "models/환경_catboost.cbm"))
        legacy = {}
        for name in previous.feature_names_:
            if name in bundle:
                config = bundle[name]
                clusters = config["gmm"].predict(numeric[config["group_cols"]])
                legacy[name] = [str(config["label_map"][int(cluster)]) for cluster in clusters]
            else:
                column = {"Workspace": "작업공간 크기", "Acoustic_Environment": "소음"}[name]
                legacy[name] = numeric[column].astype(str)
        legacy = pd.DataFrame(legacy)
        old_probs = previous.predict_proba(legacy, thread_count=2)
        report["previous_gmm_model_same_raw_domain"] = {
            "n_raw_scenarios": len(legacy),
            "n_distinct_composite_inputs": len(legacy.drop_duplicates()),
            "prediction_counts": counts(np.asarray(previous.classes_)[old_probs.argmax(axis=1)]),
            "scope": "Old saved model with workbook-corrected numeric encoding; not the old UI bug",
        }
    return report


def request_checks():
    client = app.test_client()
    cases = [
        ("missing_workload_single_route", "/api/predict/1", {}),
        ("incomplete_workload_batch", "/api/predict", {"F.1": {"F.1.S.1": "F.1.S.1.1"}}),
        ("missing_workload_questions", "/api/test/predict/1", {}),
        ("unknown_workload_question", "/api/test/predict/1", {"F.1.S.3_typo": "F.1.S.3.3"}),
        ("conflicting_workload_aliases", "/api/test/predict/1", {"F.1.S.3": "F.1.S.3.3", "F.1.3": "F.1.3.1"}),
        ("missing_environment_questions", "/api/predict/6", {}),
    ]
    results = {}
    for name, route, body in cases:
        response = client.post(route, json=body)
        results[name] = {"route": route, "request": body, "status": response.status_code, "response": response.get_json()}
    response = client.post("/api/predict", data="{bad", content_type="application/json")
    results["malformed_json"] = {"route": "/api/predict", "status": response.status_code, "response": response.get_json()}
    results["education_same_months"] = []
    for choose_last in (False, True):
        conditions = {field: f"{field}.{len(values) if choose_last else 1}" for field, values in CATBOOST_OPTION_VALUES.items() if field.startswith("F.4.")}
        conditions["training_elapsed_months"] = 12
        response = client.post("/api/predict", json={"F.4": conditions})
        results["education_same_months"].append({"request": conditions, "response": response.get_json()})
    results["route_inventory"] = sorted(str(rule) for rule in app.url_map.iter_rules())
    return results


def gmm_tables():
    tables = []
    for psf, feature in [(2, "System_Support"), (8, "Decision_Complexity"),
                         (8, "Information_Complexity"), (4, "Training_Frequency"),
                         (4, "Technical_Knowledge")]:
        tables.append({
            "psf": psf, "composite_feature": feature,
            "row_field_then_column_field": dict(CATBOOST_CONFIG[psf]["features"])[feature],
            "row_and_column_numeric_levels": [0, 1, 2],
            "output_levels": [[predict_composite_feature(psf, feature, [a, b]) for b in range(3)] for a in range(3)],
        })
    return tables


def main():
    warnings.filterwarnings("once")
    # Production supplies arrays in the checked feature order. Keep version
    # warnings visible, but avoid repeating this known name warning per row.
    warnings.filterwarnings("ignore", message="X does not have valid feature names, but GaussianMixture was fitted with feature names")
    protected_paths = [p for p in ROOT.rglob("*") if p.is_file() and (
        p.suffix in {".cbm", ".joblib", ".ipynb", ".xlsx", ".pdf"}
        or (p.suffix == ".txt" and p.name != "requirements.txt")
    )]
    before = {str(p.relative_to(ROOT)): sha256(p) for p in protected_paths}
    report = {"psfs": {}, "versions": {"python": platform.python_version(),
        "catboost": catboost.__version__, "scikit_learn": sklearn.__version__,
        "numpy": np.__version__, "pandas": pd.__version__}}
    for psf in range(1, 9):
        with contextlib.redirect_stdout(io.StringIO()):
            report["psfs"][str(psf)] = enumerate_psf(psf)
        print(f'PSF{psf}: {report["psfs"][str(psf)]["n_scenarios"]} scenarios, {report["psfs"][str(psf)]["prediction_counts"]}', flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        report["gmm_mapping_tables"] = gmm_tables()
        report["request_checks"] = request_checks()
        raw, labels, groups = load_training_data()
        model = load_catboost_model(6)
        report["psf6_training_data_reprediction"] = evaluate(labels, model.predict_proba(raw), list(model.classes_), groups)
    metadata = json.loads((ROOT / "models/environment_raw14_metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_sha256"] == sha256(ROOT / "철도_코레일 종사자 설문조사 결과.txt")
    assert metadata["model_sha256"] == sha256(ROOT / "models/환경_raw14_catboost.cbm")
    assert report["psf6_training_data_reprediction"] == metadata["full_fit_in_sample_only"]
    report["psf6_previous_oof_metadata_not_retrained"] = metadata["group_kfold_oof"]
    report["psf6_label_counts"] = metadata["label_counts"]
    report["psf6_majority_only_accuracy"] = float(labels.value_counts().max() / len(labels))
    assert all(sha256(ROOT / name) == digest for name, digest in before.items())
    report["protected_artifact_hashes"] = before
    (HERE / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f'Wrote aggregate review to {HERE / "results.json"}; {len(before)} original artifacts unchanged.', flush=True)


if __name__ == "__main__":
    main()
