"""Score every results/raw/*.jsonl run against the CatBoost reference and write results/summary.json."""
from __future__ import annotations

import json
import statistics
import time
from collections import Counter, defaultdict

import numpy as np

from bench.reference import hep, load_spec, log10_ratio, multiplier, predict
from bench.spec import LEVELS, ROOT


def qwk(y: list[int], p: list[int], k: int) -> float:
    if k < 2:
        return float("nan")
    o = np.zeros((k, k))
    for a, b in zip(y, p):
        o[a, b] += 1
    w = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    e = np.outer(o.sum(1), o.sum(0)) / max(o.sum(), 1)
    den = (w * e).sum()
    return float(1 - (w * o).sum() / den) if den else float("nan")


def macro_f1(y: list[str], p: list[str | None], labels: list[str]) -> float:
    f = []
    for c in labels:
        tp = sum(a == c and b == c for a, b in zip(y, p))
        fp = sum(a != c and b == c for a, b in zip(y, p))
        fn = sum(a == c and b != c for a, b in zip(y, p))
        if tp + fp + fn:
            f.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f)) if f else float("nan")


def catboost_speed() -> dict:
    spec = load_spec()
    items = [json.loads(x) for x in open(ROOT / "data/items.jsonl", encoding="utf-8")]
    t = time.perf_counter()
    for it in items:
        s = spec[str(it["psf"])]
        predict(it["psf"], [{i["key"]: i["options"][it["choice"][i["key"]]]["value"] for i in s["items"]}])
    per = (time.perf_counter() - t) / len(items)
    return {"per_item_ms_single_cpu": round(per * 1000, 3), "items": len(items)}


def score_run(rows: list[dict], items: dict, profiles: list[dict]) -> dict:
    res = {"n": len(rows), "errors": sum("error" in r for r in rows),
           "unparsed": sum(r.get("pred") is None and "error" not in r for r in rows), "per_psf": {}}
    ys, ps = [], []
    for pid in sorted({r["psf"] for r in rows}):
        rr = [r for r in rows if r["psf"] == pid]
        lv = LEVELS[pid]
        y = [r["label"] for r in rr]
        p = [r.get("pred") for r in rr]
        idx = {c: i for i, c in enumerate(lv)}
        valid = [(idx[a], idx[b]) for a, b in zip(y, p) if b in idx]
        res["per_psf"][pid] = {
            "n": len(rr), "accuracy": round(sum(a == b for a, b in zip(y, p)) / len(rr), 3),
            "macro_f1": round(macro_f1(y, p, lv), 3),
            "qwk": round(qwk([v[0] for v in valid], [v[1] for v in valid], len(lv)), 3) if valid else None,
            "mean_abs_level_err": round(float(np.mean([abs(a - b) for a, b in valid])), 3) if valid else None,
            "pred_dist": dict(Counter(str(x) for x in p)),
            "confusion": {f"{a}->{b}": c for (a, b), c in Counter(zip(y, map(str, p))).items()},
        }
        ys += y
        ps += p
    res["accuracy_all"] = round(sum(a == b for a, b in zip(ys, ps)) / max(len(ys), 1), 3)
    psf_ids = [k for k in res["per_psf"] if k != 6]
    res["accuracy_excl_f6"] = round(float(np.mean([res["per_psf"][k]["accuracy"] for k in psf_ids])), 3)
    res["macro_f1_mean_excl_f6"] = round(float(np.mean([res["per_psf"][k]["macro_f1"] for k in psf_ids])), 3)
    lat = [r["latency_s"] for r in rows if "latency_s" in r and "error" not in r]
    if lat:
        res["latency_s"] = {"p50": round(statistics.median(lat), 3),
                            "p90": round(float(np.percentile(lat, 90)), 3), "max": round(max(lat), 3)}
    toks = [r["usage"].get("completion_tokens", 0) for r in rows if r.get("usage")]
    if toks:
        res["completion_tokens_mean"] = round(float(np.mean(toks)), 1)
    # HEP impact on full survey profiles
    pred = {r["id"]: r.get("pred") for r in rows}
    errs, merrs, same_bucket, n_ok = [], [], 0, 0
    for pr in profiles:
        ids = pr["items"]
        if not all(pred.get(i) for i in ids.values()):
            continue
        ref = hep(pr["role"], {int(k): items[i]["label"] for k, i in ids.items()}, pr["months"])
        llm = hep(pr["role"], {int(k): pred[i] for k, i in ids.items()}, pr["months"])
        d = log10_ratio(ref, llm)
        errs.append(d)
        merrs.append(log10_ratio(multiplier({int(k): items[i]["label"] for k, i in ids.items()}, pr["months"]),
                                 multiplier({int(k): pred[i] for k, i in ids.items()}, pr["months"])))
        same_bucket += int(np.floor(np.log10(ref)) == np.floor(np.log10(llm)))
        n_ok += 1
    if errs:
        res["hep"] = {"profiles": n_ok, "median_abs_log10_err": round(statistics.median(errs), 3),
                      "within_x2": round(sum(e <= np.log10(2) for e in errs) / n_ok, 3),
                      "within_x10": round(sum(e <= 1 for e in errs) / n_ok, 3),
                      "same_order_of_magnitude": round(same_bucket / n_ok, 3)}
        res["psf_multiplier"] = {"median_abs_log10_err": round(statistics.median(merrs), 3),
                                 "within_x2": round(sum(e <= np.log10(2) for e in merrs) / n_ok, 3),
                                 "within_x10": round(sum(e <= 1 for e in merrs) / n_ok, 3)}
    return res


def main() -> None:
    items = {json.loads(x)["id"]: json.loads(x) for x in open(ROOT / "data/items.jsonl", encoding="utf-8")}
    profiles = [json.loads(x) for x in open(ROOT / "data/profiles.jsonl", encoding="utf-8")]
    runs = defaultdict(list)
    for f in sorted((ROOT / "results/raw").glob("*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            runs[f.stem].append(r)
    summary = {"catboost": catboost_speed(), "runs": {}, "repeatability": {}, "monotonicity": {}}
    ref_hep = [hep(p["role"], {int(k): items[i]["label"] for k, i in p["items"].items()}, p["months"]) for p in profiles]
    summary["reference_hep"] = {"profiles": len(ref_hep), "share_above_1": round(float(np.mean(np.array(ref_hep) > 1)), 3),
                                "median": round(statistics.median(ref_hep), 3), "max": round(max(ref_hep), 3)}
    last_pred = {n: {r["id"]: r.get("pred") for r in rows if "error" not in r} for n, rows in runs.items()}
    pert = [json.loads(x) for x in open(ROOT / "data/items_perturbed.jsonl", encoding="utf-8")] \
        if (ROOT / "data/items_perturbed.jsonl").exists() else []
    if pert:
        viol = sum(LEVELS[p["psf"]].index(p["label"]) < LEVELS[p["psf"]].index(items[p["base"]]["label"]) for p in pert)
        summary["monotonicity"]["catboost"] = {"n": len(pert), "violations": viol}
    for name in list(runs):
        if name.startswith("perturb-"):
            base, pp = last_pred.get(name[len("perturb-"):], {}), last_pred[name]
            n = v = 0
            for p in pert:
                a, b = base.get(p["base"]), pp.get(p["id"])
                if a in LEVELS[p["psf"]] and b in LEVELS[p["psf"]]:
                    n += 1
                    v += LEVELS[p["psf"]].index(b) < LEVELS[p["psf"]].index(a)
            summary["monotonicity"][name[len("perturb-"):]] = {"n": n, "violations": v}
            del runs[name]
        elif name.endswith("-rep"):
            a, b = last_pred.get(name[:-4], {}), last_pred[name]
            common = [i for i in b if i in a]
            summary["repeatability"][name[:-4]] = {
                "n": len(common), "same_answer": round(sum(a[i] == b[i] for i in common) / max(len(common), 1), 3)}
            del runs[name]
    for name in ("nemotron-zeroshot-temp1",):
        if name in last_pred and "nemotron-zeroshot" in last_pred:
            a, b = last_pred["nemotron-zeroshot"], last_pred[name]
            common = [i for i in b if i in a]
            summary["repeatability"]["nemotron-zeroshot vs temperature 1.0"] = {
                "n": len(common), "same_answer": round(sum(a[i] == b[i] for i in common) / len(common), 3)}
    print("repeatability", summary["repeatability"])
    print("monotonicity", summary["monotonicity"])
    for name, rows in runs.items():
        last = {}
        for r in rows:  # keep the last attempt per item
            if r["id"] not in last or "error" in last[r["id"]]:
                last[r["id"]] = r
        meta_f = ROOT / "results/raw" / f"{name}.meta.json"
        s = score_run(list(last.values()), items, profiles)
        if meta_f.exists():
            s["meta"] = json.loads(meta_f.read_text(encoding="utf-8"))
            if s["meta"]["n"]:
                s["throughput_items_per_s"] = round(s["meta"]["n"] / max(s["meta"]["wall_s"], 1e-9), 2)
        summary["runs"][name] = s
        print(f"{name:28s} acc={s['accuracy_all']:.3f} acc(ex F6)={s['accuracy_excl_f6']:.3f} "
              f"F1={s['macro_f1_mean_excl_f6']:.3f} err={s['errors']} unparsed={s['unparsed']} "
              f"M x2={s.get('psf_multiplier', {}).get('within_x2')} M x10={s.get('psf_multiplier', {}).get('within_x10')} tput={s.get('throughput_items_per_s')}")
    (ROOT / "results/summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
