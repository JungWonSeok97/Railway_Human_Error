"""Inspect which inputs can affect each CURRENT saved GMM composite.

Enumerate products of levels observed in the survey, after existing numeric
mapping. These are structural transformation checks, not prediction scores.
"""
import itertools
import warnings

import joblib
import pandas as pd

import run_comparison as experiment


def main():
    before = experiment.protected_manifest()
    answers = experiment.load_answers()
    results = []
    for psf in [2, 4, 6, 8]:
        raw, numeric, _, _, _ = experiment.make_data(answers, psf)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", experiment.sklearn.exceptions.InconsistentVersionWarning)
            bundle = joblib.load(experiment.PROJECT / f"models/gmm/psf{psf}_gmm.joblib")
        item = {"psf": psf, "composites": {}, "raw_to_numeric": {}}
        for col in raw:
            pairs = pd.DataFrame({"raw": raw[col], "numeric": numeric[col]}).drop_duplicates()
            item["raw_to_numeric"][col] = {
                "n_raw_levels": int(raw[col].nunique()),
                "n_numeric_levels": int(numeric[col].nunique()),
                "mapping": pairs.to_dict("records"),
            }
        for name, entry in bundle.items():
            cols = entry["group_cols"]
            levels = [sorted(numeric[c].unique().tolist()) for c in cols]
            tuples = list(itertools.product(*levels))
            model = entry["gmm"]
            x = pd.DataFrame(tuples, columns=cols)
            clusters = model.predict(x if hasattr(model, "feature_names_in_") else x.to_numpy())
            states = [int(entry["label_map"][int(c)]) for c in clusters]
            lookup = dict(zip(tuples, states))
            effects = {}
            for index, col in enumerate(cols):
                changed = total = 0
                for low, high in itertools.combinations(levels[index], 2):
                    for row in tuples:
                        if row[index] != low:
                            continue
                        other = list(row)
                        other[index] = high
                        changed += int(lookup[row] != lookup[tuple(other)])
                        total += 1
                effects[col] = {
                    "changed_output_pairs": changed,
                    "all_pairs": total,
                    "status": "constant_input" if total == 0 else (
                        "no_effect_on_this_composite" if changed == 0 else "some_effect"
                    ),
                }
            item["composites"][name] = {
                "columns": cols, "n_input_combinations": len(tuples),
                "n_output_states": len(set(states)), "effects": effects,
                "mapping": [dict(zip(cols, row), output_state=state) for row, state in zip(tuples, states)],
            }
        results.append(item)
    assert before == experiment.protected_manifest()
    experiment.write_json(experiment.HERE / "results/compression_diagnostics.json", {
        "scope": "CURRENT stored GMM; Cartesian products of observed mapped levels; not OOF performance",
        "protected_files_unchanged": True,
        "results": results,
    })
    print("Saved structural GMM diagnostics; original files unchanged.")


if __name__ == "__main__":
    main()
