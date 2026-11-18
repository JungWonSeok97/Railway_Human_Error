"""Review notebook outputs, rebuild data and replay models in memory only.

Run from the workspace root with python -B. Original notebooks, model files,
and survey records are read-only. Only aggregate evidence is written here.
"""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import platform
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import catboost
import sklearn
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "Railway_hep"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from config.catboost_config import CATBOOST_CONFIG
from config.mappings import PSF_NAMES
from test_catboost_encoding import workbook_rows

spec = importlib.util.spec_from_file_location("prior_comparison", HERE.parent / "gmm_comparison_20261118/run_comparison.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(book, index):
    return "".join(book["cells"][index].get("source", []))


def output_text(book, index):
    parts = []
    for output in book["cells"][index].get("outputs", []):
        text = output.get("text", output.get("data", {}).get("text/plain", ""))
        parts.append("".join(text) if isinstance(text, list) else text)
    return "\n".join(parts)


def load_answers():
    path = ROOT / "CatBoost_Result/preprocess_v1.4.ipynb"
    assert sha(path) == comparison.NB_SHA256
    book = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(json.loads((ROOT / "철도_코레일 종사자 설문조사 결과.txt").read_text(encoding="utf-8")))
    frame["questions"] = frame.questions.map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v)
    ns = {"pd": pd, "json": json, "df": frame}
    with contextlib.redirect_stdout(io.StringIO()):
        for index in comparison.PREPROCESS_CELLS:
            if index == 30:
                ns["answer_df"] = ns["df_answer"].copy()
            exec(compile(source(book, index), f"preprocess-cell-{index}", "exec"), ns)
    # Identifiers are used only for overlap and distinct-person counts in memory.
    return ns["answer_df_new_quality"], ns["survey_quality_new"]


def model_factory(**params):
    return CatBoostClassifier(**{**params, "thread_count": 2, "allow_writing_files": False})


def metrics(y, predicted):
    labels = sorted(set(y))
    matrix = confusion_matrix(y, predicted, labels=labels)
    return {
        "classes": labels, "prediction_counts": {label: int(np.sum(predicted == label)) for label in labels},
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "recall": dict(zip(labels, (matrix.diagonal() / matrix.sum(axis=1)).tolist())),
        "confusion_matrix": matrix.tolist(),
    }


def split_diagnostics(splitter, frame, subjects):
    rows = []
    for fold, (train, test) in enumerate(splitter, 1):
        train_scenarios = set(frame.scenario_id.iloc[train])
        test_scenarios = set(frame.scenario_id.iloc[test])
        test_seen = frame.scenario_id.iloc[test].isin(train_scenarios)
        rows.append({
            "fold": fold, "n_train": len(train), "n_test": len(test),
            "scenario_overlap": len(train_scenarios & test_scenarios),
            "test_rows_with_seen_scenario": int(test_seen.sum()),
            "person_overlap": len(set(subjects.iloc[train]) & set(subjects.iloc[test])),
        })
    return rows


def supplementary_checks(report, answers):
    report["versions"] = {"python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "catboost": catboost.__version__, "scikit_learn": sklearn.__version__}
    for psf in range(1, 9):
        selected = answers.loc[(answers.PSF == psf) & (answers.quality_95 == "성실") & answers.answer.notna()].reset_index(drop=True)
        raw = pd.DataFrame(selected.conditions.tolist()).astype(str)
        groups = pd.factorize(pd.Series(list(raw.itertuples(index=False, name=None))))[0]
        raw_denominator = int(pd.crosstab(groups, selected.answer).max(axis=1).sum())
        stat = report["psfs"][str(psf)]
        model_denominator = stat["replayed_main_cv_fixed_saved_gmm"]["denominator"]
        stat["empirical_majority_ceiling"] = {
            "original_scenarios_numerator": raw_denominator,
            "original_scenarios_ceiling": raw_denominator / len(selected),
            "model_scenarios_numerator": model_denominator,
            "model_scenarios_ceiling": model_denominator / len(selected),
        }
    book = json.loads((ROOT / "CatBoost_Result/analysis_v1.5.ipynb").read_text(encoding="utf-8"))
    namespace = {}
    exec(compile(source(book, 73), "latest-analysis-function-definitions", "exec"), namespace)
    frame = pd.DataFrame(columns=["empid", "survey_id", "quality_95", "근무시간", "수면시간", "answer"])
    report["out_of_order_execution_probe"] = {
        "latest_default_feature_columns": namespace["get_feature_cols"](frame),
        "earlier_cell22_passes_drop_base_cols": "drop_base_cols=" in source(book, 22),
        "scope": "Demonstrates a rerun hazard; saved deployed models have the correct feature names",
    }
    report["stored_x2_metric_baselines"] = {
        "test_n": 664, "stored_accuracy": 369 / 664,
        "always_middle_adjacent_accuracy": 1.0,
        "stored_adjacent_accuracy": 635 / 664,
        "stored_top2_accuracy": 589 / 664,
        "constant_two_most_frequent_classes_top2_accuracy": 588 / 664,
    }


def main():
    protected = {str(p.relative_to(ROOT)): sha(p) for p in ROOT.rglob("*") if p.is_file() and p.suffix in {".ipynb", ".cbm", ".joblib", ".xlsx", ".txt"}}
    notebooks = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in (ROOT / "CatBoost_Result").glob("*.ipynb")}
    analysis = notebooks["analysis_v1.5.ipynb"]
    ns = {"pd": pd, "np": np, "GroupKFold": GroupKFold, "CatBoostClassifier": model_factory, "accuracy_score": accuracy_score}
    for index in [19, 20, 21, 64, 72]:
        exec(compile(source(analysis, index), f"analysis-cell-{index}", "exec"), ns)
    report = {"cell_numbering": "zero-based notebook cell index", "psfs": {}, "notebooks": {}}
    for name, book in notebooks.items():
        report["notebooks"][name] = {
            "sha256": sha(ROOT / "CatBoost_Result" / name), "cells": len(book["cells"]),
            "errors": [{"cell": i, "execution": cell.get("execution_count"), "type": out.get("ename"), "message": out.get("evalue")}
                       for i, cell in enumerate(book["cells"]) for out in cell.get("outputs", []) if out.get("output_type") == "error"],
        }
    # Only aggregate stored output is exported; never raw participant tables.
    report["stored_outputs"] = {str(i): output_text(analysis, i) for i in [22, 23, 25, 26, 27, 28, 40, 56, 62, 67, 75, 88, 90, 94, 96, 98, 99, 100, 101, 102]}
    report["stored_main_summary"] = []
    pattern = re.compile(r"\[([^\]]+)\] 모델 학습 시작.*?데이터 수: (\d+).*?시나리오 수: (\d+).*?변수 수: (\d+).*?일반 Accuracy 평균: ([0-9.]+).*?보정 Accuracy: ([0-9.]+).*?분자/분모: (\d+) / (\d+)", re.S)
    for match in pattern.finditer(output_text(analysis, 22)):
        name, n, scenarios, features, raw, adjusted, numerator, denominator = match.groups()
        report["stored_main_summary"].append({"name": name, "n": int(n), "scenarios": int(scenarios), "features": int(features), "raw_rounded": float(raw), "relative_rounded": float(adjusted), "numerator": int(numerator), "denominator": int(denominator)})
    assert len(report["stored_main_summary"]) == 8
    answers, quality = load_answers()
    report["quality95"] = {
        "surveys_before_year_exception": quality["quality_95_raw"].value_counts().to_dict(),
        "surveys_after_year_exception": quality["quality_95"].value_counts().to_dict(),
        "surveys_restored_by_year_exception": int(((quality.quality_95_raw == "불성실") & (quality.quality_95 == "성실")).sum()),
    }
    mapping = {(psf, column, text): value for psf, column, text, value in workbook_rows()}
    for psf in range(1, 9):
        selected = answers.loc[(answers.PSF == psf) & (answers.quality_95 == "성실") & answers.answer.notna()].reset_index(drop=True)
        raw = pd.DataFrame(selected.conditions.tolist()).astype(str)
        y = selected.answer.astype(str)
        numeric = pd.DataFrame({column: raw[column].map(lambda text: mapping[(PSF_NAMES[psf], column, text)]) for column in raw})
        filename = "환경_catboost.cbm" if psf == 6 else CATBOOST_CONFIG[psf]["filename"]
        saved = CatBoostClassifier().load_model(str(ROOT / "models" / filename))
        if psf in comparison.GROUPS:
            x = comparison.stored_gmm_features(numeric, psf)[saved.feature_names_]
        else:
            x = numeric[saved.feature_names_].astype(str)
        frame = x.copy()
        frame["answer"] = y
        original_groups = ns["make_scenario_id"](raw, list(raw.columns))
        composite_groups = ns["make_scenario_id"](x, list(x.columns))
        stat = {"name": PSF_NAMES[psf], "n": len(y), "raw_features": list(raw.columns), "model_features": list(x.columns),
                "n_original_scenarios": len(set(original_groups)), "n_model_scenarios": len(set(composite_groups)),
                "actual_label_counts": y.value_counts().to_dict(), "majority_accuracy": float(y.value_counts().max()/len(y))}
        group_frame = pd.DataFrame({"scenario_id": original_groups, "answer": y})
        stat["scenario_label_split_on_original_scenarios"] = split_diagnostics(ns["scenario_label_kfold_split"](group_frame), group_frame, selected.empid)
        gmm_frame = pd.DataFrame({"scenario_id": composite_groups, "answer": y})
        stat["proper_groupkfold_on_model_scenarios"] = split_diagnostics(GroupKFold(5).split(x, y, composite_groups), gmm_frame, selected.empid)
        person_counts = pd.DataFrame({"scenario": original_groups, "person": selected.empid}).groupby("scenario").agg(n_responses=("person", "size"), n_people=("person", "nunique"))
        stat["multi_response_not_multi_person"] = {
            "scenarios_two_or_more_responses": int((person_counts.n_responses >= 2).sum()),
            "scenarios_two_or_more_people": int((person_counts.n_people >= 2).sum()),
            "repeated_responses_from_same_person": int(len(selected) - len(selected.assign(scenario=original_groups)[["empid", "scenario"]].drop_duplicates())),
        }
        with contextlib.redirect_stdout(io.StringIO()):
            cv = ns["run_catboost_by_psf"](PSF_NAMES[psf], frame)
        stat["replayed_main_cv_fixed_saved_gmm"] = {
            "scope": "Replays notebook section 2 with the saved GMM transformation; GMM is not refitted within folds. Historical PSF6 uses six GMM features.",
            "raw_fold_mean": float(cv["raw_accuracy_mean"]), "relative_accuracy": float(cv["theoretical_accuracy_total"]),
            "numerator": int(cv["theoretical_numerator"]), "denominator": int(cv["theoretical_denominator"]),
            **metrics(cv["y_true"], cv["y_pred"]),
        }
        candidate = model_factory(**comparison.PARAMS)
        candidate.fit(x, y, cat_features=list(x.columns))
        saved_prob = saved.predict_proba(x, thread_count=2)
        candidate_prob = candidate.predict_proba(x, thread_count=2)
        assert list(candidate.classes_) == list(saved.classes_)
        stat["in_memory_final_refit_comparison"] = {
            "max_probability_difference": float(np.max(np.abs(candidate_prob - saved_prob))),
            "different_prediction_rows": int((candidate_prob.argmax(axis=1) != saved_prob.argmax(axis=1)).sum()),
            "scope": "Fresh fit with section-2 parameters, no artifact save; not a held-out evaluation",
        }
        report["psfs"][str(psf)] = stat
        (HERE / "results.partial.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f'PSF{psf}: n={len(y)}, raw/model scenarios={stat["n_original_scenarios"]}/{stat["n_model_scenarios"]}, CV={stat["replayed_main_cv_fixed_saved_gmm"]["accuracy"]:.4f}, refit label differences={stat["in_memory_final_refit_comparison"]["different_prediction_rows"]}', flush=True)
    demo = pd.DataFrame({"scenario_id": [0]*10, "answer": ["보통"]*7+["나쁨"]*3})
    report["split_example_same_scenario_7_to_3"] = split_diagnostics(ns["scenario_label_kfold_split"](demo), demo, pd.Series(range(10)))
    report["relative_metric_counterexample"] = ns["theoretical_max_accuracy"](np.array(["보통","보통","나쁨","나쁨"]), np.array(["보통","보통","나쁨","나쁨"]), np.array([0,0,0,0]))[0]
    overlap = ns["distribution_overlap_by_scenario"](np.array(["보통","보통","나쁨","나쁨"]), np.array(["나쁨","나쁨","보통","보통"]), np.array([0,0,0,0]))
    report["distribution_overlap_counterexample"] = {"raw_accuracy": 0.0, "scenario_overlap": overlap[0], "note": "Valid marginal histogram equality, but not evidence that individual responses or model probabilities are correct"}
    supplementary_checks(report, answers)
    assert all(sha(ROOT / name) == digest for name, digest in protected.items())
    report["protected_hashes"] = protected
    (HERE / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "results.partial.json").unlink()
    print('Completed; original files unchanged.', flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
