# 데이터

> **이 저장소에는 설문에서 나온 파일을 넣지 않았습니다.** `data/survey/`와 `data/expert/`는 공개 저장소에 올리지 않았습니다. 설문 원본을 `data/raw/`에 둔 뒤 `uv run python -m bench.survey_data`와 `uv run python -m bench.expert_split`로 다시 만듭니다. 결과의 집계값(`results/expert/summary.json` 등)과 보고서는 들어 있습니다.

## 코레일 종사자 설문(전문가 라벨)
- 원본: `data/raw/철도_코레일 종사자 설문조사 결과.txt`(37MB, JSON). 2025-11-14~2026-04-17에 저장된 응답 13,208건(1건 = 한 회차, 질문 2개)이며 응답자는 396명입니다.
- **원본은 git에 넣지 않습니다**(`.gitignore`의 `data/raw/`). 이름, 사번, 직급, 부서, 성별, 생년월일, 전화, 회사 메일, 건강검진 값(체온, 혈압, 맥박), 사원증 번호, 근무 형태 같은 개인 정보가 들어 있습니다. 공유가 필요하면 사내 보안 절차를 따릅니다.
- `data/survey/responses.jsonl`(git에 포함): 비교에 필요한 필드만 남긴 비식별 파일입니다. 질문 하나가 한 줄이고 26,416줄입니다.

| 필드 | 내용 |
|---|---|
| `id` | `S<회차 순번>-<질문 순번>` |
| `rid` | 응답자 가명(`R0001`~`R0396`, 원본 등장 순서). 원본 없이는 실제 사람과 연결할 수 없습니다 |
| `job` | 기관사, 관제사, 작업자, 승무원 |
| `years_band` | 경력 구간(0-4, 5-9, 10-19, 20+) |
| `round` | 회차 |
| `psf` | 질문 번호 1~8(F.1~F.8) |
| `choice` | 문항 키(`F.p.S.k`) → `data/survey_spec.json`의 보기 순번. 조건 문장은 여기서 다시 만듭니다 |
| `answer` | 응답자가 고른 PSF 수준 |

- 다시 만들기: `uv run python -m bench.survey_data`(원본 경로를 인자로 줄 수 있음). 표기 차이 4개 문항 이름과 `10m^3` 단위는 `bench/survey_data.py`의 별칭 표로 맞춥니다. 변환에서 빠진 응답은 없습니다.

## 평가용 파일(`data/expert/`, `bench/expert_split.py`)
| 파일 | 내용 |
|---|---|
| `split.json` | 응답자 단위 80/20 분할(학습 317명, 시험 79명). 같은 사람이 양쪽에 들어가지 않습니다 |
| `test_items.jsonl` | 시험 응답자의 질문에서 PSF마다 120개씩 무작위로 뽑은 960개(실제 분포 그대로) |
| `consensus_items.jsonl` | 시험 응답자 3명 이상이 같은 조건을 판정했고 다수 의견이 갈리지 않은 328개 시나리오 |
| `llm_items.jsonl` | LLM에 보낸 1,288개(위 두 파일 합) |
| `shots.jsonl` | few-shot 예시: 학습 응답자 답 중 수준별 8개(PSF당 16~24개) |
| `shots_natural.jsonl` | few-shot 예시: 학습 응답자 답 24개를 실제 분포대로 무작위 추출 |

## 그 밖의 파일
- `survey_spec.json`: Railway_hep `input.html`에서 뽑은 문항과 보기. `value`는 모델이 학습한 방향(2 = 가장 나쁨), `value_web`은 원래 웹앱이 보내던 값입니다.
- `items.jsonl`, `shots.jsonl`, `profiles.jsonl`, `items_perturbed.jsonl`: 2026-11-17의 첫 비교(공개 CatBoost 판정을 기준 라벨로 쓴 합성 데이터).
