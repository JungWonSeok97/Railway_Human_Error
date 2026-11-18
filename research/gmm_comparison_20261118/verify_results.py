"""Independently verify exported OOF arrays, aggregate metrics, and source hashes."""
from pathlib import Path
import hashlib
import json

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1] / "Railway_hep"


def main():
    results = HERE / "results"
    report = json.loads((results / "comparison.json").read_text(encoding="utf-8"))
    protocol = report["protocol"]
    assert protocol["protected_files_unchanged"]
    assert protocol["protected_before"] == protocol["protected_after"]
    for relative, expected in protocol["protected_before"].items():
        assert hashlib.sha256((PROJECT / relative).read_bytes()).hexdigest() == expected
    assert hashlib.sha256((HERE / "run_comparison.py").read_bytes()).hexdigest() == protocol["runner_sha256"]
    checked = []
    for result in report["results"]:
        psf = result["psf"]
        arrays = np.load(results / f"psf{psf}_oof.npz", allow_pickle=False)
        y, groups, fold = arrays["y"], arrays["groups"], arrays["fold"]
        labels = arrays["labels"].tolist()
        n, k = len(y), len(labels)
        assert n == result["data"]["n_quality95"]
        assert set(fold) == {1, 2, 3, 4, 5}
        assert len(set(groups)) == result["data"]["n_original_scenarios"]
        for group in np.unique(groups):
            assert len(set(fold[groups == group])) == 1
        assert result["reproduction"]["maximum_probability_difference"] == 0
        for variant in protocol["variants"] + ["prior"]:
            probabilities = arrays[variant]
            assert probabilities.shape == (n, k)
            assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
            assert np.allclose(probabilities.sum(axis=1), 1)
            prediction = probabilities.argmax(axis=1)
            cm = np.zeros((k, k), dtype=int)
            np.add.at(cm, (y, prediction), 1)
            actual_counts, predicted_counts = cm.sum(axis=1), cm.sum(axis=0)
            recalls = cm.diagonal() / actual_counts
            precisions = np.divide(cm.diagonal(), predicted_counts, out=np.zeros(k), where=predicted_counts > 0)
            f1 = 2 * cm.diagonal() / (actual_counts + predicted_counts)
            expected = result["training_prior_baseline"] if variant == "prior" else result["variants"][variant]
            assert labels == expected["labels"]
            assert cm.tolist() == expected["confusion_matrix"]
            assert np.isclose((prediction == y).mean(), expected["accuracy"])
            assert np.isclose(recalls.mean(), expected["balanced_accuracy"])
            assert np.isclose(f1.mean(), expected["macro_f1"])
            assert np.isclose(-np.log(probabilities[np.arange(n), y]).mean(), expected["log_loss"])
            _, first = np.unique(groups, return_index=True)
            for index, label in enumerate(labels):
                assert actual_counts[index] == result["data"]["label_counts"][label]
                assert predicted_counts[index] == expected["prediction_counts"][label]
                assert np.isclose(recalls[index], expected["recall"][label])
                assert np.isclose(precisions[index], expected["precision"][label])
                assert np.isclose(predicted_counts[index] / n, expected["prediction_ratios"][label])
                assert (prediction[first] == index).sum() == expected["scenario_prediction_counts"][label]
            for f in range(1, 6):
                train = fold != f
                test = fold == f
                assert not set(groups[train]) & set(groups[test])
                if variant == "prior":
                    priors = np.bincount(y[train], minlength=k) / train.sum()
                    assert np.allclose(probabilities[test], priors)
                else:
                    fold_result = result["folds"][f - 1]
                    assert fold_result["scenario_overlap"] == 0
                    assert all(fold_result["gmm_converged"].values())
                    assert np.isclose(
                        (prediction[test] == y[test]).mean(),
                        fold_result["variants"][variant]["accuracy"],
                    )
        checked.append({"psf": psf, "n": n, "status": "passed"})
        print(f"PASS PSF{psf}: {n} OOF responses; probabilities, counts, metrics, splits", flush=True)
    verification = {
        "status": "passed", "psfs": checked,
        "protected_files_checked": len(protocol["protected_before"]),
        "protected_files_unchanged": True,
    }
    (results / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("PASS protected source files and models unchanged", flush=True)


if __name__ == "__main__":
    main()
