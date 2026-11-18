"""Print the markdown tables of docs/report.md from results/expert/summary.json (no hand-copied numbers)."""
from __future__ import annotations

import json
import sys

from bench.spec import ROOT

ROWS = [
    ("응답자 간 일치(같은 조건, 다른 사람)", None),
    ("기준선: 늘 가장 흔한 수준", "baseline-majority"),
    ("기준선: 문항 평균 규칙(데이터 불필요)", "baseline-item-mean"),
    ("기준선: 로지스틱 회귀", "baseline-logreg"),
    ("CatBoost 원래 모델, 웹앱 그대로", "catboost-original-deployed"),
    ("CatBoost 원래 모델, 보기 순서만 고침", "catboost-original-inputfix"),
    ("**CatBoost 재학습(고친 앱)**", "catboost-fixed-app"),
    ("CatBoost 재학습, GMM 묶음 유지", "catboost-retrained"),
    ("OpenJev systemone, 예시 없음", "openjev-systemone-zeroshot"),
    ("OpenJev systemone, 예시 24개(수준별 8)", "openjev-systemone-fewshot"),
    ("OpenJev chat, 예시 없음", "openjev-chat-zeroshot"),
    ("OpenJev chat, 예시 24개(수준별 8)", "openjev-chat-fewshot"),
    ("OpenJev chat, 실제 분포 예시 24개", "openjev-chat-fewshot-natural"),
    ("Nemotron, 예시 없음", "nemotron-zeroshot"),
    ("Nemotron, 예시 24개(수준별 8)", "nemotron-fewshot"),
    ("Nemotron, 실제 분포 예시 24개", "nemotron-fewshot-natural"),
    ("Kimi, 예시 없음", "kimi-zeroshot"),
    ("Kimi, 예시 24개(수준별 8)", "kimi-fewshot"),
    ("Kimi, 실제 분포 예시 24개", "kimi-fewshot-natural"),
]


def pct(x):
    return f"{100 * x:.1f}%"


def ci(c):
    return f"{100 * c[0]:.1f}~{100 * c[1]:.1f}"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    s = json.loads((ROOT / "results/expert/summary.json").read_text(encoding="utf-8"))
    runs = s["runs"]
    print("| 모델 | 정답률(960문항) | 95% 구간 | 고친 앱 CatBoost와 차이(95% 구간, %p) | macro-F1 | QWK | 판정 편향 | 다수 의견 시나리오(328) | HEP 배수 10배 이내 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, key in ROWS:
        if key is None:
            pair = s["consensus"]["pairwise_agreement_same_scenario"]
            vals = [v for v in pair.values() if v is not None]
            print(f"| {name} | {pct(min(vals))}~{pct(max(vals))} (PSF별) | | | | | | | |")
            continue
        r = runs.get(key)
        if not r or "sample" not in r:
            continue
        sm, cs = r["sample"], r.get("consensus")
        diff = "기준" if key == s["reference"] else ci(sm["diff_vs_ref_ci95"]).replace("~", " ~ ")
        print(f"| {name} | {pct(sm['accuracy'])} | {ci(sm['acc_ci95'])} | {diff} | {sm['macro_f1_mean_psf']:.3f} | "
              f"{sm['qwk_mean_psf']:.3f} | {sm['severity_bias_mean_psf']:+.2f} | "
              f"{pct(cs['accuracy']) if cs else ''} | {pct(sm['multiplier']['within_x10'])} |")
    print()
    print("| 모델 | " + " | ".join(f"F.{p}" for p in range(1, 9)) + " |")
    print("|---|" + "---|" * 8)
    for name, key in ROWS:
        r = runs.get(key or "", {}).get("sample")
        if not r:
            continue
        print(f"| {name} | " + " | ".join(pct(r["per_psf"][str(p)]["accuracy"]) if str(p) in r["per_psf"]
                                          else pct(r["per_psf"][p]["accuracy"]) for p in range(1, 9)) + " |")
    print()
    print("| 모델 | 지연 p50 | 처리량(문항/초) | 동시 요청 |")
    print("|---|---|---|---|")
    for name, key in ROWS:
        r = runs.get(key or "", {})
        if "meta" in r:
            print(f"| {name} | {r.get('latency_p50_s')}초 | {r.get('throughput_per_s')} | {r['meta'].get('concurrency')} |")


if __name__ == "__main__":
    main()
