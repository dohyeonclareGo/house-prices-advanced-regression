[English](README.md) | 한국어

# EDA 요약

노트북: [eda.ipynb](eda.ipynb). 각 발견은 **무엇을 봤는지 → 무엇을 알게 됐는지 → 모델링에서 어떻게 반영했는지** 순서로 정리했습니다.

데이터는 train 1,460채, test 1,459채이고 피처는 79개입니다. `MSSubClass`, `MoSold`, `YrSold`를 범주형으로 돌리면 수치형 33개, 범주형(명목형) 46개가 됩니다.

## 1. 타깃이 오른쪽으로 치우쳐 있다

- **본 것**: `SalePrice`의 분포. [그래프: SalePrice 히스토그램, log1p 변환 전후]
- **알게 된 것**: 왜도(skewness) 1.88, 첨도(kurtosis) 6.54입니다. 소수의 비싼 집이 오른쪽으로 긴 꼬리를 만듭니다. 평가 지표가 log 가격 RMSE라서 오차를 비율로 잽니다.
- **반영**: 모델은 `log1p(SalePrice)`로 학습하고, 예측은 `expm1`으로 되돌립니다. log 공간에서 학습하면 평가 지표와 기준이 맞습니다.

## 2. 범주형 결측(NA)은 대부분 "해당 시설 없음"이다

- **본 것**: train과 test를 합친 2,917채의 결측 개수. [그래프: 컬럼별 결측 개수]
  - `PoolQC` 2,908, `MiscFeature` 2,812, `Alley` 2,719, `Fence` 2,346, `FireplaceQu` 1,420
  - 차고 관련 컬럼 157~159, 지하 관련 컬럼 79~82
- **알게 된 것**: 데이터 설명서를 보면 이 컬럼들의 NA는 "수영장·차고·지하가 없다"는 뜻입니다. 범주별 가격 그래프에서도 NA("None") 범주가 대체로 가장 쌉니다. 결측 자체가 정보입니다.
- **반영**: 범주형 NA는 `'None'`으로, 짝이 되는 수치형(차고 면적, 지하 면적 등)은 0으로 채웠습니다. 나머지 컬럼은 다음과 같이 채웠습니다.
  - `LotFrontage`: 같은 동네(`Neighborhood`)의 중앙값
  - `GarageYrBlt`: `YearBuilt`. test에 있는 2207년 오타도 `YearBuilt`로 바꿉니다.
  - 결측이 몇 개뿐인 컬럼(`MSZoning`, `KitchenQual` 등): train의 최빈값

## 3. 이상치 2채: Id 524, 1299

- **본 것**: `GrLivArea`, `TotalBsmtSF`, `1stFlrSF`, `GarageArea`, `LotArea`, `OverallQual`과 `SalePrice`의 산점도. [그래프: GrLivArea vs SalePrice, 524·1299 표시]
- **알게 된 것**:
  - 524와 1299는 `GrLivArea` 그래프의 오른쪽 아래에 있습니다. 면적은 매우 큰데 가격이 낮습니다.
  - 둘 다 OverallQual 10이고, 완공 전에 거래된 `Partial` 거래입니다.
  - 1299는 `TotalBsmtSF`, `1stFlrSF`에서도 다른 집들과 동떨어져 있습니다.
  - 반면 큰 집인 692와 1183은 비싸고 추세를 따릅니다. `LotArea`가 극단적인 집들의 가격도 정상입니다.
- **반영**: 524와 1299만 학습에서 뺐습니다(1,458채 사용). 692와 1183은 남겼습니다. `LotArea`는 빼지 않고 변환으로 처리했습니다.

## 4. 같은 조건의 test 집: Id 2550

- **본 것**: 이상치 2채와 같은 조건(`GrLivArea > 4000`이고 `SaleCondition == 'Partial'`)인 test 집.
- **알게 된 것**: 딱 1채, Id 2550입니다(GrLivArea 5,095, Edwards, Partial). 이상치를 학습에서 뺐기 때문에 모델은 이 집을 일반 추세대로 크고 비싼 집으로 예측합니다.
- **반영**: 이 집의 예측을 524·1299의 평균 log 가격(약 171,930)으로 고정했습니다. 이 규칙 하나로 LB가 0.12427에서 0.11876으로 좋아졌습니다(`src/rules.py`의 `id2550_rule`).

## 5. 치우친 수치형 피처와 대부분 0인 컬럼

- **본 것**: 수치형 피처별 히스토그램과 왜도. [그래프: 수치형 피처 히스토그램 격자, 왜도 표시]
- **알게 된 것**:
  - `LotArea`, `GrLivArea`, `TotalBsmtSF` 같은 면적 컬럼은 오른쪽으로 치우쳐 있습니다.
  - `PoolArea`, `MiscVal`, `3SsnPorch`, `LowQualFinSF`, `BsmtFinSF2`, `ScreenPorch`, `EnclosedPorch`는 대부분의 집에서 0입니다.
  - `KitchenAbvGr`는 거의 항상 1입니다.
  - `YearRemodAdd = 1950`은 1950년 이전 기록이 잘린 값입니다.
- **반영**:
  - 치우친 컬럼: v1에서는 로그 변환을 했고, v2에서는 왜도의 절댓값이 0.5를 넘는 연속형 전부에 Box-Cox 변환을 했습니다. λ는 train으로만 추정합니다.
  - 대부분 0인 컬럼: 원래 값을 빼고, 값이 0보다 큰지만 나타내는 `Has*` 피처로 바꿨습니다.

## 6. 가격과 관련된 축: 품질, 크기, 연식, 차고

- **본 것**: 전처리 후 log `SalePrice`와의 상관계수 상위 30개. [그래프: 상관계수 상위 30개 막대그래프]
- **알게 된 것**:
  - 1위권은 `OverallQual` 0.82와 `TotalSF`(지하 + 1층 + 2층) 0.82입니다. `GrLivArea` 0.74보다 높으므로, 지상 면적보다 총면적이 가격을 더 잘 설명합니다.
  - 가격을 결정하는 큰 축은 네 가지입니다.
    - 품질: `OverallQual`, `ExterQual`, `KitchenQual` 등
    - 크기: `TotalSF`, `GrLivArea`, `TotalBath` 등
    - 연식: `HouseAge` −0.59, `RemodAge` −0.57
    - 차고: `GarageCars`, `GarageFinish` 등
  - 순서형 인코딩을 하자 원래 문자열이던 `ExterQual`, `KitchenQual` 등이 상위권에 들어왔습니다.
  - 면적보다 유무가 중요한 경우도 있습니다. `OpenPorchSF`(면적)는 상위 30위 밖이지만, `HasOpenPorchSF`는 0.45입니다.
- **반영**:
  - 파생 변수: `TotalSF`, `TotalBath`, `HouseAge`, `RemodAge`, `TotalPorchSF`
  - 품질 컬럼은 순서형 숫자로 인코딩했고, v2에서 대상 컬럼을 늘렸습니다.
  - v2에서 품질×면적 상호작용(`Qual_SF`, `Qual_Liv`)을 추가했습니다.
  - 같은 네 축이 원가법 구조 모델의 구조(땅, 건물 면적, 품질, 감가)가 되었습니다.

## 7. 다중공선성(multicollinearity)

- **본 것**: 상관계수의 절댓값이 0.8 이상인 피처 쌍.
- **알게 된 것**:
  - `HouseAge`와 `YearBuilt`의 상관은 −1.00입니다. `YrSold`가 2006~2010년으로 좁아서 사실상 같은 정보입니다.
  - 그 밖의 쌍: `GarageCars`–`GarageArea` 0.89, `TotalSF`–`GrLivArea` 0.86, `YearBuilt`–`GarageYrBlt` 0.84, `GrLivArea`–`TotRmsAbvGrd` 0.82
  - `FireplaceQu`, `HasFireplaces`, `Fireplaces`는 서로 0.86~0.95입니다.
- **반영**: 일반 선형 회귀는 계수가 불안정해지므로 규제가 있는 선형 모델(Lasso, Ridge, ElasticNet)을 썼습니다. 나중에 중복 피처를 빼는 실험도 했지만 효과는 없었습니다(±0.0001).

## 8. 숫자로 적힌 범주형과 희귀 범주

- **본 것**: 각 코드의 의미와 빈도.
- **알게 된 것**:
  - `MSSubClass`(건물 유형), `MoSold`, `YrSold`는 숫자로 저장되어 있지만 범주입니다.
  - `Condition2`, `RoofMatl`, `Heating`에는 몇 채뿐인 범주가 있습니다.
- **반영**:
  - 숫자로 적힌 세 컬럼은 문자열로 바꾼 뒤 원-핫 인코딩했습니다.
  - train에서 10채 미만인 범주는 `'Other'`로 묶었습니다.
  - v2에서는 train에서 최빈값과 다른 집이 5채 미만인 원-핫 컬럼을 삭제했습니다(302 → 252개).

## 이 노트북에 없는 분석

- 잔차 분석(모델이 어떤 집을 크게 틀리는지)은 `notebooks/01_baseline_and_stack.ipynb`의 #5-3, #5-4에 있습니다. 남은 오차는 상업지역, 2008~2009년 금융위기 시기의 `Partial` 거래, 비정상 거래에 몰려 있습니다.
- train/test 적대적 검증(adversarial validation, 분포 차이 없음): [TODO: 노트북과 셀 위치]
