"""Score every run in results/expert/raw/ against the expert survey labels -> results/expert/summary.json.

Two evaluation sets:
  sample  the 960-question LLM test sample (120 per PSF, test respondents) -> every model
  full    every row of the 79 test respondents (5,036) -> the non-LLM models
Uncertainty: PSF-stratified bootstrap (2,000 resamples) for accuracy, and paired bootstrap differences
against `catboost-fixed-app` (the shipped model) on the same questions.
"""
from __future__ import annotations

import collections
import json
import math
import random
import statistics

import numpy as np

from bench.score import macro_f1, qwk
from bench.spec import LEVELS, MULTIPLIERS, ROOT

REF = "catboost-fixed-app"  # the per-item model the fix branch ships
RAW = ROOT / "results/expert/raw"


def load_runs() -> dict[str, dict[str, dict]]:
    runs = {}
    for f in sorted(RAW.glob("*.jsonl")):
        d = {}
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r["id"] not in d or "error" in d[r["id"]]:
                d[r["id"]] = r
        runs[f.stem] = d
    return runs


def metrics(items: list[dict], pred: dict[str, dict]) -> dict:
    per = {}
    for pid in range(1, 9):
        it = [i for i in items if i["psf"] == pid]
        if not it:
            continue
        lv = LEVELS[pid]
        y = [i["label"] for i in it]
        p = [(pred.get(i["id"]) or {}).get("pred") for i in it]
        idx = {c: n for n, c in enumerate(lv)}
        valid = [(idx[a], idx[b]) for a, b in zip(y, p) if b in idx]
        per[pid] = {"n": len(it), "accuracy": round(sum(a == b for a, b in zip(y, p)) / len(it), 3),
                    "macro_f1": round(macro_f1(y, p, lv), 3),
                    "qwk": round(qwk([v[0] for v in valid], [v[1] for v in valid], len(lv)), 3) if valid else None,
                    # severity bias: mean predicted level - mean expert level (levels 0 = best); > 0 = harsher
                    "severity_bias": round(float(np.mean([b - a for a, b in valid])), 3) if valid else None,
                    "pred_dist": dict(collections.Counter(str(x) for x in p))}
    n = sum(v["n"] for v in per.values())
    correct = [int((pred.get(i["id"]) or {}).get("pred") == i["label"]) for i in items]
    return {"n": n, "accuracy": round(sum(correct) / max(n, 1), 3),
            "accuracy_mean_psf": round(float(np.mean([v["accuracy"] for v in per.values()])), 3),
            "macro_f1_mean_psf": round(float(np.mean([v["macro_f1"] for v in per.values()])), 3),
            "qwk_mean_psf": round(float(np.nanmean([v["qwk"] if v["qwk"] is not None else np.nan
                                                     for v in per.values()])), 3),
            "missing": sum((pred.get(i["id"]) or {}).get("pred") not in LEVELS[i["psf"]] for i in items),
            "severity_bias_mean_psf": round(float(np.nanmean([v["severity_bias"] if v["severity_bias"] is not None
                                                              else np.nan for v in per.values()])), 3),
            "per_psf": per}


def bootstrap(items, pred, ref_pred=None, b=2000, seed=7):
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for i in items:
        by[i["psf"]].append(i)
    acc, diff = [], []
    for _ in range(b):
        s = [rng.choice(v) for v in by.values() for _ in v]
        c = [int((pred.get(i["id"]) or {}).get("pred") == i["label"]) for i in s]
        acc.append(sum(c) / len(c))
        if ref_pred is not None:
            r = [int((ref_pred.get(i["id"]) or {}).get("pred") == i["label"]) for i in s]
            diff.append((sum(c) - sum(r)) / len(c))
    out = {"acc_ci95": [round(float(np.percentile(acc, 2.5)), 3), round(float(np.percentile(acc, 97.5)), 3)]}
    if diff:
        out["diff_vs_ref_ci95"] = [round(float(np.percentile(diff, 2.5)), 3), round(float(np.percentile(diff, 97.5)), 3)]
    return out


def multiplier_agreement(items, pred, n_profiles=300, seed=11):
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for i in items:
        by[i["psf"]].append(i)
    errs, ok = [], 0
    for _ in range(n_profiles):
        prof = [rng.choice(by[p]) for p in range(1, 9)]
        preds = [(pred.get(i["id"]) or {}).get("pred") for i in prof]
        if any(p not in LEVELS[i["psf"]] for p, i in zip(preds, prof)):
            errs.append(math.inf)
            continue
        m_exp = math.prod(MULTIPLIERS[i["psf"]][i["label"]] for i in prof)
        m_mod = math.prod(MULTIPLIERS[i["psf"]][p] for p, i in zip(preds, prof))
        errs.append(abs(math.log10(m_exp) - math.log10(m_mod)))
        ok += 1
    fin = [e for e in errs if math.isfinite(e)]
    return {"profiles": n_profiles, "within_x2": round(sum(e <= math.log10(2) for e in errs) / n_profiles, 3),
            "within_x10": round(sum(e <= 1 for e in errs) / n_profiles, 3),
            "median_abs_log10": round(statistics.median(fin), 3) if fin else None}


def main() -> None:
    sample = [json.loads(x) for x in open(ROOT / "data/expert/test_items.jsonl", encoding="utf-8")]
    split = json.loads((ROOT / "data/expert/split.json").read_text(encoding="utf-8"))
    test_ids = set(split["test"])
    full = [{"id": r["id"], "psf": r["psf"], "label": r["answer"]}
            for r in map(json.loads, open(ROOT / "data/survey/responses.jsonl", encoding="utf-8"))
            if r["rid"] in test_ids]
    cons = [json.loads(x) for x in open(ROOT / "data/expert/consensus_items.jsonl", encoding="utf-8")]
    runs = load_runs()
    ref = runs.get(REF)
    summary = {"reference": REF, "sample_n": len(sample), "full_n": len(full), "consensus_n": len(cons), "runs": {}}
    for name, pred in runs.items():
        s = {}
        if cons and sum(i["id"] in pred for i in cons) / len(cons) >= 0.95:
            s["consensus"] = metrics(cons, pred) | bootstrap(cons, pred, ref if name != REF else None)
        cov_sample = sum(i["id"] in pred for i in sample) / len(sample)
        cov_full = sum(i["id"] in pred for i in full) / len(full)
        if cov_sample >= 0.95:
            s["sample"] = metrics(sample, pred) | bootstrap(sample, pred, ref if name != REF else None)
            s["sample"]["multiplier"] = multiplier_agreement(sample, pred)
        if cov_full >= 0.95:
            s["full"] = metrics(full, pred) | bootstrap(full, pred, ref if name != REF else None, b=500)
        if name == "ceiling-other-experts":
            covered = [i for i in full if i["id"] in pred]
            s["covered"] = {"n": len(covered), **{k: v for k, v in metrics(covered, pred).items() if k != "per_psf"},
                            "per_psf_accuracy": {p: v["accuracy"] for p, v in metrics(covered, pred)["per_psf"].items()}}
            s["covered_models"] = {m: metrics(covered, runs[m])["accuracy"] for m in runs
                                   if m.startswith(("catboost", "baseline")) and m != name}
        meta = RAW / f"{name}.meta.json"
        if meta.exists():
            mt = json.loads(meta.read_text(encoding="utf-8"))
            s["meta"] = mt
            lat = [r["latency_s"] for r in pred.values() if "latency_s" in r and "error" not in r]
            if lat:
                s["latency_p50_s"] = round(statistics.median(lat), 3)
            s["throughput_per_s"] = round(mt["n"] / max(mt["wall_s"], 1e-9), 2) if mt.get("n") else None
        summary["runs"][name] = s
    cons = ROOT / "results/expert/consensus.json"
    if cons.exists():
        summary["consensus"] = json.loads(cons.read_text(encoding="utf-8"))
    (ROOT / "results/expert/summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{'run':34s} {'sample acc':>10s} {'95% CI':>13s} {'vs ref':>15s} {'F1':>6s} {'QWK':>6s} {'M x2':>5s} "
          f"{'full acc':>8s} {'cons acc':>8s} {'cons QWK':>8s}")
    order = sorted(summary["runs"].items(), key=lambda kv: -(kv[1].get("sample", {}).get("accuracy") or 0))
    for name, s in order:
        sm, fl, cs = s.get("sample"), s.get("full"), s.get("consensus")
        if not sm and not fl:
            continue
        print(f"{name:34s} {sm['accuracy'] if sm else '':>10} {str(sm['acc_ci95']) if sm else '':>13s} "
              f"{str(sm.get('diff_vs_ref_ci95', '')) if sm else '':>15s} {sm['macro_f1_mean_psf'] if sm else '':>6} "
              f"{sm['qwk_mean_psf'] if sm else '':>6} {sm['multiplier']['within_x2'] if sm else '':>5} "
              f"{fl['accuracy'] if fl else '':>8} {cs['accuracy'] if cs else '':>8} {cs['qwk_mean_psf'] if cs else '':>8}")
    if "ceiling-other-experts" in summary["runs"]:
        c = summary["runs"]["ceiling-other-experts"]["covered"]
        print("ceiling (other experts, identical scenario):", c["n"], "rows, acc", c["accuracy"], c["per_psf_accuracy"])
        print("   models on the same rows:", summary["runs"]["ceiling-other-experts"]["covered_models"])
        print("pairwise expert agreement:", summary.get("consensus", {}).get("pairwise_agreement_same_scenario"))


if __name__ == "__main__":
    main()
