[English](README.md) | 한국어

# House Prices – Advanced Regression Techniques

미국 아이오와주 에임스(Ames)의 집값을 79개 피처로 예측한 프로젝트입니다. **최종 퍼블릭 리더보드 점수는 0.11516**입니다(log 가격 RMSE, 목표 0.114).

최종 결과는 모델 3개의 블렌드에 원가법 하이브리드 모델과 데이터를 보고 만든 규칙 2개를 더한 것입니다. 이 저장소에는 어떤 변경이 리더보드까지 넘어왔고 어떤 변경이 넘어오지 못했는지, 그리고 채택·기각을 어떤 기준으로 판단했는지를 기록했습니다.

## 문제 정의와 평가 지표

- **과제**: 집 1,460채로 학습해서 test 집 1,459채의 `SalePrice`를 예측하는 회귀 문제입니다([Kaggle 대회](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)).
- **평가 지표**: log(예측 가격)과 log(실제 가격) 사이의 RMSE입니다. 싼 집과 비싼 집의 오차가 같은 비율로 계산됩니다.
- **검증 방식**: 이상치 2채(Id 524, 1299)를 뺀 train에서 `KFold(5, shuffle=True, random_state=42)`로 교차 검증(Cross-Validation, CV)을 했습니다. 피처를 바꿀 때는 `RepeatedKFold(5×3)`로 15번 짝지어 비교했고, 평균과 함께 15번 중 몇 번 좋아졌는지를 봤습니다.

## EDA 요약

자세한 내용은 [eda/README.ko.md](eda/README.ko.md)에 있습니다.

1. **타깃이 오른쪽으로 치우쳐 있습니다**(왜도 1.88). 그래서 모델은 `log1p(SalePrice)`를 예측합니다.
2. **범주형 결측(NA)은 대부분 "해당 시설 없음"이라는 뜻입니다**(예: 수영장 없음, 차고 없음). 최빈값이 아니라 `'None'`으로 채웠습니다.
3. **Id 524와 1299는 이상치입니다.** 지상 면적이 매우 크고 OverallQual이 10인데 가격은 낮습니다. 둘 다 완공 전에 거래된 `Partial` 거래입니다. 학습에서 제외했습니다.
4. **test의 Id 2550이 같은 조건입니다**(GrLivArea 5,095로 4,000 초과, `Partial` 거래). 별도 규칙을 만든 근거입니다(아래 참고).

## 접근 과정

### Step 3 – 기본 모델: 단순하게 시작하고, 다르게 틀리는 모델을 더하기
- Lasso 단독은 LB 0.12558, Lasso와 XGBoost를 50:50으로 섞으면 LB 0.12427이었습니다. 선형 모델과 트리 모델은 틀리는 집이 다릅니다.
- **Id 2550 규칙**: 모든 모델이 Id 2550을 크고 비싼 집으로 예측했습니다. 그래서 이 집의 예측을 524·1299의 평균 log 가격으로 고정했습니다. 집 한 채에 대한 규칙으로 LB가 0.12427에서 **0.11876**으로 좋아졌습니다.

### Step 4 – 업그레이드: CV는 좋아졌지만 리더보드는 나빠짐
- Lasso + XGBoost + SVR **스택(stack)**으로 LB **0.11660**을 얻었습니다.
  - 스택: 각 모델의 OOF 예측을 입력으로 받아, 모델별 가중치를 메타 모델이 학습하는 방식입니다.
  - OOF(Out-of-Fold) 예측: train의 각 집을 그 집을 보지 않은 모델로 예측한 값입니다.
- 이어서 상호작용 피처, 비정상 거래 샘플 가중치, kNN, 구조 모델을 추가했습니다. CV는 0.1058까지 좋아졌지만 LB는 0.11715로 나빠졌습니다.
- 제출 6번에서 CV와 LB의 상관은 **−0.79**였습니다. CV가 좋아질수록 리더보드가 나빠진 것입니다.
- **적대적 검증(adversarial validation)** 결과 train과 test의 분포 차이는 없었습니다. 적대적 검증은 train 행과 test 행을 구분하는 분류기를 학습해 보고, 구분하지 못하면 두 데이터가 비슷하다고 판단하는 방법입니다. 따라서 점수 차이의 원인은 분포 차이가 아니었습니다. 같은 train에 계속 맞추다 보니 CV가 낙관적으로 변했다고 판단했습니다.

### Step 5 – 새 노트북: 다시 정리하고, 제출 전에 판단 기준부터 정하기
- **v2 피처 정리**: Box-Cox, 순서형 인코딩 확대, 품질×면적, 희귀 더미 제거. 모두 거의 모든 집에 적용되는 변경입니다. Lasso CV가 0.1117에서 0.1093으로 좋아졌고(15번 중 15번 개선), 스택은 LB **0.11535**를 얻었습니다.
- **v3**(구조 모델에 품질 제곱항 추가)는 LB 0.11713으로 기각했습니다. 제곱항은 train에 몇 채 없는 최고·최저 품질 구간에서 추정되어, test에서 예측이 흔들렸습니다.
- **v4 하이브리드**: 원가법 구조 모델(땅값 + 건물 면적 × 단가 × 품질 × 감가)이 먼저 가격을 예측하고, Lasso는 그 잔차만 학습합니다. 하이브리드 단독 CV는 0.1054였습니다(Lasso 0.1092, 구조 모델 0.1103). XGBoost, SVR과 블렌드해서 LB **0.11526**을 얻었습니다.
- **코너 보정**: IDOTRR 동네이면서 상업지역(`C (all)`)이고 코너 땅인 test 6채만 +10% 했습니다. LB **0.11516**으로 최고 점수입니다.
- **v4.1과 v5는 기각했습니다.** 구조 모델 계열의 비중이 33%를 넘으면 리더보드가 매번 나빠졌습니다. v5는 채택·기각 점수 기준을 제출 전에 미리 정해 두었습니다.

## 결과

주요 버전만 정리했습니다. CV와 LB 숫자 전체, 버전별 판단 이유는 [docs/experiments.md](docs/experiments.md)에 있습니다.

| 단계 | 버전 | CV (log RMSE) | 퍼블릭 LB | 판단 |
|---|---|---|---|---|
| 3 | Lasso | 0.1116 | 0.12558 | 기준점 |
| 3 | Lasso + XGBoost 블렌드 | 0.1081 | 0.12427 | 채택 |
| 3 | + Id 2550 규칙 | – | 0.11876 | 채택 |
| 4 | Lasso + XGBoost + SVR 스택 | 0.1077 | 0.11660 | 채택 |
| 4 | + 상호작용, 가중치, kNN, 구조 모델 | 0.1058 | 0.11715 | 기각 |
| 5 | v2 피처 정리, 스택 | 0.1049 | 0.11535 | 채택 |
| 5 | v3 구조 모델 + 제곱항 | 0.1035 | 0.11713 | 기각 |
| 5 | v4 하이브리드, 블렌드 | 0.1036 | 0.11526 | 채택 |
| 5 | **v4 블렌드 + 코너 보정** | – | **0.11516** | **최종** |

`–`는 test 집에만 적용되는 변경이라 CV로 측정할 수 없다는 뜻입니다. 블렌드(blend)는 가중치를 학습하지 않고 모델 예측을 단순 평균하는 방식입니다.

## 핵심 발견 / 배운 점

1. **리더보드까지 넘어온 개선은 세 종류였습니다.** (1) 데이터를 보고 만든 규칙(Id 2550, 코너 보정), (2) 거의 모든 집에 적용되는 피처 정리(v2), (3) 다르게 틀리는 모델을 적당한 비중으로 섞기.
2. **CV와 리더보드는 반대로 움직일 수 있습니다.** Step 4에서는 CV가 좋아지는 동안 LB가 나빠졌습니다(제출 6번에서 상관 −0.79). 적대적 검증으로 분포 차이를 원인에서 제외한 뒤, 작은 CV 개선만으로 판단하지 않도록 절차를 바꿨습니다.
   - 15번 짝지어 비교해서 평균뿐 아니라 개선 횟수를 확인합니다.
   - 모델 단독 점수와 함께 스택 안에서 받는 가중치를 확인합니다.
   - 제출 전에 채택·기각 LB 기준을 적어 둡니다(v5).
3. **train의 일부 구간에 맞춘 변경은 넘어오지 않았습니다.** 극단적인 품질 구간에서 추정한 제곱항이나 한 동네(IDOTRR)만을 위한 제약은 CV나 test 예측의 안정성은 개선했지만, LB는 개선하지 못했습니다.
4. **남은 오차는 가격이 정해진 이유가 79개 피처 밖에 있는 집들에 몰려 있습니다.** 상업지역, 2008~2009년 금융위기 시기의 `Partial` 거래, 비정상 거래가 여기에 해당합니다.

## 재현 방법

**환경**: 노트북은 Kaggle Notebook에서 실행했습니다. 로컬에서는 다음을 설치합니다.

```bash
pip install -r requirements.txt
```

**데이터**: 대회 규칙상 재배포할 수 없어서 저장소에 포함하지 않았습니다. Kaggle에서 대회 규칙에 동의한 뒤 다음을 실행합니다.

```bash
kaggle competitions download -c house-prices-advanced-regression-techniques -p data
unzip data/house-prices-advanced-regression-techniques.zip -d data
```

각 노트북은 `/kaggle/input/**/train.csv`를 먼저 찾고, 없으면 `../data/`를 사용합니다.

**실행 순서**:
1. `eda/eda.ipynb` – EDA
2. `notebooks/01_baseline_and_stack.ipynb` – Step 3~4
3. `notebooks/02_v2_features.ipynb` – v2
4. `notebooks/03_v4_hybrid.ipynb` – 최종 모델. 셀 #8이 `submission_v4_blend.csv`를, 셀 #10이 최종 파일 `submission_v4_blend_corner10.csv`를 만듭니다.

노트북은 실험 기록으로 쓰기 위해 실행했던 상태 그대로(출력 포함) 두었습니다. `src/`에는 노트북 02~03의 코드를 다시 쓸 수 있게 모듈로 나눠 담았습니다.

```python
from src.preprocessing import load_data, build
from src.structure_model import prep_struct, fit_hybrid
from src.rules import id2550_rule, corner_adjust
```

## 폴더 구조

```
house-prices-advanced-regression/
├── README.md / README.ko.md
├── requirements.txt
├── data/                          # 커밋하지 않음 – Kaggle에서 다운로드
├── eda/
│   ├── README.md / README.ko.md   # EDA 요약
│   ├── eda.ipynb
│   └── figures/
├── notebooks/
│   ├── 01_baseline_and_stack.ipynb   # Step 3~4
│   ├── 02_v2_features.ipynb          # Step 5 v2
│   └── 03_v4_hybrid.ipynb            # Step 5 v4 (최종)
├── src/
│   ├── preprocessing.py   # 결측 처리, 이상치 제거, 변환
│   ├── features.py        # 피처 생성
│   ├── models.py          # 베이스 모델과 하이퍼파라미터
│   ├── structure_model.py # 원가법 구조 모델, 하이브리드
│   ├── cv.py              # KFold 평가, 오차 상관, 스택/블렌드 비교
│   └── rules.py           # Id 2550 규칙, 코너 보정
├── docs/
│   └── experiments.md     # 전체 실험 기록
└── submissions/
    └── final_v4_blend_corner.csv
```
