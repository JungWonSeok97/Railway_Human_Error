# Railway_hep와 survey 앱 수정

비교 중에 찾은 문제를 고쳤습니다. 두 저장소 모두 작성자(JungWonSeok97)의 저장소이므로 **푸시하지 않았고**, 로컬 브랜치와 패치 파일로 남겼습니다.

| 대상 | 로컬 브랜치 | 패치 |
|---|---|---|
| Railway_hep (`f864cce` 기준) | `vendor/Railway_hep-fixed`의 `fix/benchmark-findings` | `railway_hep-0001-*.patch`(코드), `railway_hep-0002-*.patch`(설정과 모델 교체, 8MB), `railway_hep-0003-*.patch`(README) |
| survey (`aa47df3` 기준) | 패치 파일만(작업용 클론은 임시 폴더) | `survey-0001-lightning-fog-options.patch` |

두 패치 모두 원래 커밋의 새 클론에 `git am`으로 적용되는 것을 확인했고, 적용한 Railway_hep에서 Node 시험 6개가 모두 통과합니다.

적용:
```bash
cd Railway_hep && git am ../railway-human-error-model-benchmark/fixes/railway_hep-000*.patch
cd survey && git am ../railway-human-error-model-benchmark/fixes/survey-0001-lightning-fog-options.patch
```
모델 교체(0002)는 바이너리 패치입니다. 코드 수정(0001)만 먼저 받을 수도 있습니다.

## 고친 문제

| # | 문제 | 원인 | 고친 곳 | 확인 |
|---|---|---|---|---|
| 1 | 수면시간, 절차 복잡성, 황사, 마찰계수의 값이 반대로 들어감 | `input.html:900`의 보기 순서 보정표가 비어 있어 화면 순서대로 코드를 보냄 | 보정표 4문항 채움 | 설문에서 "6시간 이내 수면"(평균 0.77)이 "8시간 이상"(0.47)보다 나쁘게 판정됨. `tests/test_bench.py::test_fixed_form_sends_expert_direction`이 모든 보기를 확인 |
| 2 | 환경(F.6) 모델이 늘 "나쁨" | 설문 앱이 낙뢰와 안개를 "발생할 경우"로만 보여 줌(`survey/src/lib/surveyData.ts:72–73`). 응답 3,433건 전부가 낙뢰·안개 상황이라 71%가 "나쁨"이고 다른 문항의 영향이 거의 없음 | survey 앱에 "발생하지 않을 경우" 보기 추가, 조합을 모두 만드는 대신 문항마다 독립적으로 뽑기 | 지금 데이터로 다시 학습해도 시험 질문의 97%를 "나쁨"으로 판정. **새 설문을 받아 다시 학습해야 고쳐집니다** |
| 3 | 최종 HEP가 1을 넘음(최대 약 2) | 보정한 진단·수행 HEP를 더함 | `result.html`, `detail_result.html`: `1 - (1 - 진단)(1 - 수행)` | 새 Node 시험(모든 PSF 최악)이 원래 코드에서 1.99999로 실패하고 고친 코드에서 통과. 기존 시험 포함 6/6 통과 |
| 4 | 응답자가 뚜렷하게 반응하는 문항을 모델이 무시 | (가) 보기가 둘인 문항을 학습 때는 0/1, 웹앱은 0/2로 보내 모델이 처음 보는 값을 받음(장비 가용성, 절차 유무). (나) F.2/F.4/F.6/F.8은 문항 2~3개를 GMM으로 묶으면서 한 문항의 정보가 사라짐 | 모든 PSF를 "문항 하나 = 특성 하나"로 바꾸고(`config/catboost_config.py`), 설문 26,416건으로 웹앱이 보내는 값 그대로 다시 학습(`models/*.cbm`), 쓰지 않게 된 `models/gmm/*.joblib` 삭제 | 응답자가 반응하는 문항 중 모델이 무시하는 문항 7개 → 0개, 거꾸로 반응하는 문항 2개 → 0개(`bench/item_effects.py`) |
| 5 | `jargon_need_map`, `info_uncertainty_map`의 0과 2가 반대 | 텍스트 매핑 오기 | `config/mappings.py` | 설문에서 "전문용어 항상 필요", "불확실성 높음"이 더 나쁘게 판정됨. 기존 CNN 경로(`predictors/cnn.py:486, 593`)만 영향 |

## 모델 교체(0002)의 근거
- 시험 응답자 79명(학습에 쓰지 않은 사람)에 대한 성능은 원래 모델과 통계적으로 차이가 없습니다. 응답자끼리도 같은 조건에서 38~62%만 같은 답을 고르므로, 개별 응답 정답률은 CatBoost든 로지스틱 회귀든 56~59%에 머뭅니다.
- 원래 모델은 이 설문으로 학습된 것으로 보입니다(설문 저장 2025-11~2026-04, 저장소 생성 2026-10). 그래서 원래 모델의 시험 점수는 이미 본 응답이 섞인 값입니다.
- 대신 새 모델은 모든 문항에 응답자와 같은 방향으로 반응하고, 웹앱이 보내는 값과 학습한 값이 같습니다. 학습 데이터는 `data/survey/responses.jsonl` 전체이고, `bench/train_catboost.py --rows all --app vendor/Railway_hep-fixed`로 다시 만들 수 있습니다. 하이퍼파라미터는 원래 모델에서 읽은 값(500회, 깊이 6, 학습률 0.05)을 그대로 썼습니다.
- `bench/parity_check.py`가 고친 Railway_hep의 실제 예측 함수(`predictors/catboost.py`)를 화면이 보내는 코드로 호출해, 벤치마크 예측과 모두 같은지 확인합니다.

## 남은 일
- survey 앱 패치는 빌드하지 않았습니다(`npm install`이 필요). 배포 전에 `npm run build`와 화면 확인이 필요합니다.
- 환경(F.6)은 낙뢰·안개가 없는 조건의 설문을 새로 받아야 제대로 학습할 수 있습니다. 그 전까지 F.6 판정은 거의 항상 "나쁨"입니다.
- 두 저장소에 반영하려면 작성자에게 패치를 전달하거나, 권한이 있으면 브랜치를 푸시해 PR을 엽니다.
