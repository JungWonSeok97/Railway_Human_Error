"""Audit saved CBM artifacts and live Flask inference paths without training.

Run from the repository root. Only aggregate evidence is written beside this file.
"""
import contextlib
import hashlib
import importlib.util
import io
import itertools
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = REPO / "Railway_hep"
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("previous_runtime_audit", REPO / "research/code_review_20261118/run_review.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

from config.catboost_config import CATBOOST_CONFIG
from config.catboost_mappings import CATBOOST_OPTION_VALUES
from config.mappings import PSF_LABELS
from predictors import catboost as predictor
from predictors.psf_multipliers import multiplier_for_prediction
from preprocessing import catboost_features as preprocessing


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def psf4_class_selection_check():
    fields = {key: values for key, values in CATBOOST_OPTION_VALUES.items() if key.startswith("F.4.")}
    choices = list(itertools.product(*(range(1, len(values) + 1) for values in fields.values())))
    model = predictor.load_catboost_model(4)
    frame = pd.DataFrame([
        preprocessing.build_catboost_features(4, {field: f"{field}.{value}" for field, value in zip(fields, row)})
        for row in choices
    ], columns=model.feature_names_)
    probabilities = model.predict_proba(frame)
    classes = list(model.classes_)
    good = classes.index("좋음")
    index = int(probabilities[:, good].argmax())
    return {
        "classes": classes, "n_valid_raw_combinations": len(frame),
        "n_distinct_model_inputs": len(frame.drop_duplicates()),
        "max_probability_good": float(probabilities[:, good].max()),
        "best_good_probability_scenario": {field: f"{field}.{value}" for field, value in zip(fields, choices[index])},
        "probabilities_at_max_good": dict(zip(classes, probabilities[index].tolist())),
        "selected_label_at_max_good": classes[int(probabilities[index].argmax())],
    }


def main():
    warnings.filterwarnings("once")
    warnings.filterwarnings("ignore", message="X does not have valid feature names, but GaussianMixture was fitted with feature names")
    protected = {str(p.relative_to(ROOT)): sha(p) for p in ROOT.rglob("*")
                 if p.is_file() and p.suffix in {".cbm", ".joblib", ".ipynb", ".xlsx", ".txt", ".py", ".js", ".cjs", ".html", ".json"}
                 and "__pycache__" not in p.parts}
    report = {"scope": "Saved artifacts and current inference routes; exploratory notebook outputs are not deployment defects.",
              "inventory": [], "psfs": {}, "api_model_agreement": []}
    active = {v["filename"]: k for k, v in CATBOOST_CONFIG.items()}
    for path in sorted((ROOT / "models").glob("*.cbm")):
        model = CatBoostClassifier().load_model(str(path))
        psf = active.get(path.name)
        entry = {"filename": path.name, "sha256": sha(path), "configured_psf": psf,
                 "default_batch_active": psf is not None and psf != 4,
                 "features": list(model.feature_names_), "classes": model.classes_.tolist(),
                 "categorical_indices": model.get_cat_feature_indices(), "tree_count": model.tree_count_}
        if psf is not None:
            assert entry["features"] == [name for name, _ in CATBOOST_CONFIG[psf]["features"]]
            assert set(entry["classes"]) == set(PSF_LABELS[psf].values())
            assert entry["categorical_indices"] == list(range(len(entry["features"])))
            with contextlib.redirect_stdout(io.StringIO()):
                predictor.load_catboost_model(psf)
            entry["schema_and_labels_match"] = True
        report["inventory"].append(entry)

    client = audit.app.test_client()
    for psf in range(1, 9):
        with contextlib.redirect_stdout(io.StringIO()):
            result = audit.enumerate_psf(psf)
            model = predictor.load_catboost_model(psf)
            for label, example in result["examples"].items():
                frame = pd.DataFrame([preprocessing.build_catboost_features(psf, example["conditions"])], columns=model.feature_names_)
                probs = model.predict_proba(frame)[0]
                assert np.isfinite(probs).all() and (probs >= 0).all() and np.isclose(probs.sum(), 1)
                response = client.post(f"/api/predict/{psf}", json=example["conditions"])
                assert response.status_code == 200
                body = response.get_json()["result"]
                for i, text in PSF_LABELS[psf].items():
                    code = f"F.{psf}.{i + 1}"
                    assert body["probabilities"][code] == float(probs[list(model.classes_).index(text)])
                expected_index = list(PSF_LABELS[psf].values()).index(label)
                assert body["predicted_label"] == expected_index
                assert body["predicted_text"] == f"F.{psf}.{expected_index + 1}"
                assert body["psf_multiplier"] == multiplier_for_prediction(psf, body["predicted_text"])
                report["api_model_agreement"].append({"psf": psf, "label": label, "passed": True})
        report["psfs"][str(psf)] = result
        print(f"PSF{psf}: {result['n_scenarios']} scenarios; {result['prediction_counts']}", flush=True)

    # Trace actual artifacts loaded by the default route and the all-CatBoost route.
    inputs = {f"F.{psf}": {key: f"{key}.1" for key in CATBOOST_OPTION_VALUES if key.startswith(f"F.{psf}.")} for psf in range(1, 9)}
    inputs["F.4"]["training_elapsed_months"] = 12
    report["route_traces"] = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for route in ["/api/predict", "/api/catboost/predict"]:
            predictor.catboost_models.clear()
            preprocessing.gmm_bundles.clear()
            response = client.post(route, json=inputs)
            assert response.status_code == 200
            body = response.get_json()
            assert len(body["results"]) == 8 and body["missing_psfs"] == []
            assert math.isclose(body["psf_multiplier_product"], math.prod(x["psf_multiplier"] for x in body["results"]))
            assert not any("hep" in k.lower() for k in body)
            report["route_traces"][route] = {"loaded_catboost_psfs": sorted(predictor.catboost_models),
                "loaded_gmm_psfs": sorted(preprocessing.gmm_bundles), "response": body}
        report["request_checks"] = audit.request_checks()
        report["gmm_mapping_tables"] = audit.gmm_tables()
        report["psf4_class_selection_check"] = psf4_class_selection_check()
        raw, labels, groups = audit.load_training_data()
        model = predictor.load_catboost_model(6)
        report["environment_training_reprediction"] = audit.evaluate(labels, model.predict_proba(raw), list(model.classes_), groups)

    metadata = json.loads((ROOT / "models/environment_raw14_metadata.json").read_text(encoding="utf-8"))
    assert metadata["model_sha256"] == sha(ROOT / "models/환경_raw14_catboost.cbm")
    assert metadata["source_sha256"] == sha(ROOT / "철도_코레일 종사자 설문조사 결과.txt")
    assert metadata["dataset_fingerprint"] == hashlib.sha256(json.dumps({"x": raw.values.tolist(), "y": labels.tolist()}, ensure_ascii=False).encode("utf-8")).hexdigest()
    assert report["environment_training_reprediction"] == metadata["full_fit_in_sample_only"]
    report["environment_metadata_verified"] = True
    report["environment_existing_oof_not_rerun"] = metadata["group_kfold_oof"]
    report["environment_actual_label_counts"] = metadata["label_counts"]
    report["environment_unobserved_categories"] = metadata["unobserved_categories"]

    old = json.loads((REPO / "research/notebook_review_20261118/results.json").read_text(encoding="utf-8"))
    same = all((ROOT / name).exists() and sha(ROOT / name) == digest for name, digest in old["protected_hashes"].items())
    assert same
    report["previous_full_refit_evidence"] = {
        "source": "research/notebook_review_20261118/results.json", "source_sha256": sha(REPO / "research/notebook_review_20261118/results.json"),
        "all_protected_hashes_still_match": same,
        "scope": "Prior eight legacy full fits were not repeated; PSF6 there is the old GMM model. Equality is on training rows only.",
        "results": {key: {"n": value["n"], **value["in_memory_final_refit_comparison"]} for key, value in old["psfs"].items()}}
    assert all(sha(ROOT / name) == digest for name, digest in protected.items())
    report["protected_files_unchanged"] = len(protected)
    report["protected_hashes"] = protected
    (HERE / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Complete: 9 CBM artifacts, {sum(x['n_scenarios'] for x in report['psfs'].values())} scenarios, {len(report['api_model_agreement'])} API comparisons; {len(protected)} protected files unchanged.", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
