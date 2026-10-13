# Railway Human Error

철도 인적 오류 관련 프로젝트를 연결하는 상위 저장소입니다. 두 프로젝트는 Git 서브모듈로 등록되어 있으며 각 저장소에서 독립적으로 관리합니다.

| 경로 | 저장소 | 역할 |
| --- | --- | --- |
| `Railway_hep/` | [Railway_hep](https://github.com/JungWonSeok97/Railway_hep) | PSF 및 HEP 예측 웹앱 |
| `survey/` | [survey](https://github.com/JungWonSeok97/survey) | 설문 프로젝트 |

## 전체 프로젝트 내려받기

```bash
git clone --recurse-submodules https://github.com/JungWonSeok97/Railway_Human_Error.git
cd Railway_Human_Error
```

이미 상위 저장소만 내려받았다면 다음 명령으로 서브모듈을 초기화합니다.

```bash
git submodule update --init --recursive
```

## 참조 갱신

이 저장소는 각 하위 저장소의 특정 커밋을 기록합니다. `.gitmodules`에서 추적 브랜치를 `main`으로 지정했지만 하위 저장소의 변경은 자동 반영되지 않습니다.

최신 `main` 커밋을 확인하고 반영하려면 상위 저장소에서 실행합니다.

```bash
git submodule update --remote Railway_hep survey
git diff --submodule
git add Railway_hep survey
git commit -m "Update submodule references"
git push origin main
```

상위 저장소에 기록된 버전을 다시 내려받으려면 다음 명령을 사용합니다.

```bash
git pull --ff-only
git submodule update --init --recursive
```

## 개발 및 배포

각 프로젝트의 설치 및 실행 방법은 해당 하위 저장소의 README를 참고합니다. 서브모듈 내부에서 개발할 때는 작업 브랜치로 전환하고 해당 저장소에 먼저 커밋·푸시한 뒤, 상위 저장소에서 참조를 갱신합니다.

이 상위 저장소의 서브모듈 등록은 기존 하위 저장소의 코드나 Vercel 연결 설정을 변경하지 않습니다.
