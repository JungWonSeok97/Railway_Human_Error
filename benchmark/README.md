# railway-human-error-model-benchmark

철도 인적오류 확률(HEP) 예측에서 **CatBoost 모델**([Railway_hep](https://github.com/JungWonSeok97/Railway_hep))과 **KISTI 한강에서 직접 서빙한 로컬 LLM**(OpenJev(Qwen3.8-27B 기반), NVIDIA Nemotron 3.5 Super, Kimi K3.1 Pro)을 비교하고, 비교 중에 찾은 Railway_hep의 문제를 고칩니다.

- 결과와 해석: [docs/report.md](docs/report.md)
- 고친 내용과 패치: [fixes/README.md](fixes/README.md)
- 데이터와 개인 정보 처리: [data/README.md](data/README.md)

## 무엇을 비교하나
- Railway_hep는 철도 종사자가 고른 근무 조건(PSF 8개, 하위 문항 57개)으로 PSF 수준을 분류하고, 직종별 기본 HEP에 PSF 배수를 곱해 HEP를 계산합니다.
- **실제 설문 기준(주 결과, 2026-11-18)**: 코레일 종사자 396명이 무작위 조건마다 고른 PSF 수준 26,416건을 정답으로 씁니다. 응답자 단위로 학습 317명과 시험 79명을 나누고, 모든 모델을 시험 응답자의 질문으로 평가합니다.
- **CatBoost 판정 기준(첫 비교, 2026-11-17)**: 설문을 받기 전, 공개된 CatBoost 판정을 기준 라벨로 삼아 LLM이 얼마나 같은 판정을 내리는지 쟀습니다.

## 구성
| 경로 | 내용 |
|---|---|
| `scripts/fetch_source.sh` | Railway_hep를 고정 커밋(`f864cce`)으로 `vendor/`에 받습니다(git에 넣지 않음) |
| `bench/spec.py` | `input.html`에서 문항, 보기, 웹앱이 보내는 값을 뽑아 `data/survey_spec.json`을 만듭니다 |
| `bench/survey_data.py` | 설문 원본에서 개인 정보를 빼고 `data/survey/responses.jsonl`을 만듭니다 |
| `bench/expert_split.py` | 응답자 단위 분할, LLM 시험 표본 960개, 다수 의견 시나리오 328개, few-shot 예시 |
| `bench/train_catboost.py` | 설문으로 CatBoost를 다시 학습합니다(원래 형식 또는 고친 앱의 문항별 형식) |
| `bench/expert_classic.py` | 비LLM 모델(원래·재학습 CatBoost, 기준선)과 응답자 간 일치도 |
| `bench/llm_eval.py` | OpenAI 호환 chat(Nemotron, Kimi, OpenJev) 또는 OpenJev `/v1/systemone` 호출 |
| `bench/score_expert.py` | 정확도와 95% 구간, CatBoost와의 차이, macro-F1, QWK, 판정 편향 → `results/expert/summary.json` |
| `bench/item_effects.py` | 문항별로 모델과 응답자가 반응하는 정도 비교 → `results/expert/item_effects.json` |
| `bench/parity_check.py` | 고친 Railway_hep를 실제 코드 경로로 돌려 벤치마크 예측과 같은지 확인 |
| `bench/reference.py`, `build_dataset.py`, `baselines.py`, `perturb.py`, `score.py` | 첫 비교(CatBoost 판정 기준) |
| `fixes/` | Railway_hep와 survey 앱의 수정 패치 |

## 실행
```bash
bash scripts/fetch_source.sh
uv sync
uv run python bench/spec.py
uv run python -m bench.survey_data                 # data/raw/의 설문 원본이 있을 때만
uv run python -m bench.expert_split
uv run python -m bench.train_catboost --rows train --out results/expert/models_train
uv run python -m bench.train_catboost --rows train --app vendor/Railway_hep-fixed --out results/expert/models_train_app
uv run python -m bench.expert_classic
uv run python -m bench.llm_eval --run mock --backend mock --items expert/llm_items.jsonl --out-dir results/expert/raw
uv run python -m bench.score_expert
uv run --group dev pytest -q tests
```
`vendor/Railway_hep-fixed`는 수정 브랜치의 작업 트리입니다(`fixes/README.md`).

한강 모델은 상위 레포(`setup-ai-cloud-for-our-company`)의 서빙 작업과 터널을 씁니다.
```powershell
..\hangang run -Cmd 'cd /scratch/$USER/enu-ai && sbatch -t 01:00:00 cluster/slurm/jev.sbatch && sbatch -t 01:00:00 cluster/slurm/llm.sbatch'
..\hangang tunnel
uv run python -m bench.llm_eval --run nemotron-fewshot-natural --base-url http://127.0.0.1:8100 --key-file ../.tmp/llm-api-key --extra '{"chat_template_kwargs":{"enable_thinking":false}}' --items expert/llm_items.jsonl --shots-file expert/shots_natural.jsonl --shots --out-dir results/expert/raw
```
키는 파일로만 넘깁니다. 원시 응답과 학습한 모델 파일은 git에 넣지 않고, 요약(`results/**/summary.json`)만 넣습니다.

## 라이선스와 출처
- 기준 모델과 설문 문항: JungWonSeok97/Railway_hep(이 저장소에는 복사하지 않고 실행할 때 받습니다).
- 설문 원본: 코레일 종사자 설문(개인 정보 포함, git 제외). 비식별 파일만 넣었습니다.
- OpenJev 가중치는 CC BY-NC 4.0이므로 사내 평가 용도로만 썼습니다.
