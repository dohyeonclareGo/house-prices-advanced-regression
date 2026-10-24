English | [한국어](README.ko.md)

# EDA summary

Notebook: [eda.ipynb](eda.ipynb). Each finding follows the same order: **what I looked at → what I learned → how it was used in modeling**.

Data: 1,460 train houses and 1,459 test houses, with 79 features. After treating `MSSubClass`, `MoSold`, and `YrSold` as categorical, there are 33 numeric and 46 categorical features.

## 1. The target is right-skewed

- **Looked at**: the distribution of `SalePrice`. [Graph: SalePrice histogram, before and after log1p]
- **Learned**: skewness 1.88 and kurtosis 6.54. A few expensive houses form a long right tail. The metric is RMSE on log price, so errors are measured as ratios.
- **Used**: models are trained on `log1p(SalePrice)` and predictions are converted back with `expm1`. Training in log space matches the metric.

## 2. Categorical NAs mostly mean "no such facility"

- **Looked at**: missing counts in train + test combined (2,917 houses). `PoolQC` is missing for 2,908 houses, `MiscFeature` for 2,812, `Alley` for 2,719, `Fence` for 2,346, `FireplaceQu` for 1,420, the garage columns for 157–159, and the basement columns for 79–82. [Graph: missing count by column]
- **Learned**: according to the data description, NA in these columns means the house has no pool, garage, basement, and so on. In the price plots by category, the NA ("None") group is usually the cheapest. The missing value itself carries information.
- **Used**: categorical NAs were filled with `'None'`, and the matching numeric columns (garage area, basement area, and so on) with 0. Other columns:
  - `LotFrontage`: median of the same `Neighborhood`
  - `GarageYrBlt`: `YearBuilt`, and a test typo (year 2207) is also replaced with `YearBuilt`
  - the few remaining columns (`MSZoning`, `KitchenQual`, and others): the train mode

## 3. Two outliers: Id 524 and 1299

- **Looked at**: scatter plots of `GrLivArea`, `TotalBsmtSF`, `1stFlrSF`, `GarageArea`, `LotArea`, and `OverallQual` against `SalePrice`. [Graph: GrLivArea vs SalePrice, with 524 and 1299 marked]
- **Learned**: 524 and 1299 are at the bottom right of the `GrLivArea` plot: very large living area, low price. Both have OverallQual 10, and both are `Partial` sales (sold before construction was finished). 1299 is also far from the others in `TotalBsmtSF` and `1stFlrSF`. Two other large houses, 692 and 1183, are expensive and follow the trend. Extreme `LotArea` values have normal prices.
- **Used**: only 524 and 1299 are removed from training (1,458 houses remain). 692 and 1183 are kept. `LotArea` is kept and transformed instead.

## 4. A test house with the same condition: Id 2550

- **Looked at**: test houses with the same condition as the two outliers (`GrLivArea > 4000` and `SaleCondition == 'Partial'`).
- **Learned**: exactly one house matches: Id 2550 (GrLivArea 5,095, Edwards, Partial). The outliers were removed from training, so the models predict it from the general trend as a large, expensive house.
- **Used**: its prediction is fixed to the mean log price of 524 and 1299 (about 171,930). This rule alone improved LB from 0.12427 to 0.11876 (`src/rules.py`, `id2550_rule`).

## 5. Skewed numeric features and mostly-zero columns

- **Looked at**: histograms and skewness of each numeric feature. [Graph: grid of numeric feature histograms with skewness]
- **Learned**:
  - Area columns such as `LotArea`, `GrLivArea`, and `TotalBsmtSF` are right-skewed.
  - `PoolArea`, `MiscVal`, `3SsnPorch`, `LowQualFinSF`, `BsmtFinSF2`, `ScreenPorch`, and `EnclosedPorch` are 0 for most houses.
  - `KitchenAbvGr` is almost always 1.
  - `YearRemodAdd = 1950` is a capped value for records before 1950.
- **Used**:
  - Skewed columns: log-transformed in v1, then Box-Cox in v2 for every continuous column with |skewness| > 0.5. λ is estimated on train only.
  - Mostly-zero columns: replaced with `Has*` indicators (whether the value is > 0).

## 6. What correlates with price: quality, size, age, garage

- **Looked at**: correlation with log `SalePrice` after preprocessing (top 30). [Graph: top-30 correlation bar chart]
- **Learned**:
  - The top features are `OverallQual` 0.82 and `TotalSF` 0.82 (basement + 1st + 2nd floor), above `GrLivArea` 0.74. The total area explains price better than the above-ground area.
  - The main factors group into four: quality (`OverallQual`, `ExterQual`, `KitchenQual`, …), size (`TotalSF`, `GrLivArea`, `TotalBath`, …), age (`HouseAge` −0.59, `RemodAge` −0.57), and garage (`GarageCars`, `GarageFinish`, …).
  - Ordinal encoding moved string columns such as `ExterQual` and `KitchenQual` into the top ranks.
  - Presence can matter more than size: `OpenPorchSF` (area) is outside the top 30, but `HasOpenPorchSF` has 0.45.
- **Used**:
  - Derived features: `TotalSF`, `TotalBath`, `HouseAge`, `RemodAge`, and `TotalPorchSF`.
  - Quality columns are encoded as ordinal numbers (more of them in v2).
  - v2 adds quality × area interactions (`Qual_SF`, `Qual_Liv`).
  - The same four factors became the structure of the cost-approach structure model (land, building area, quality, depreciation).

## 7. Multicollinearity

- **Looked at**: feature pairs with |r| ≥ 0.8.
- **Learned**:
  - `HouseAge` and `YearBuilt` have r = −1.00. `YrSold` only spans 2006–2010, so they carry the same information.
  - Other pairs: `GarageCars`–`GarageArea` 0.89, `TotalSF`–`GrLivArea` 0.86, `YearBuilt`–`GarageYrBlt` 0.84, `GrLivArea`–`TotRmsAbvGrd` 0.82.
  - `FireplaceQu`, `HasFireplaces`, and `Fireplaces` are 0.86–0.95 with each other.
- **Used**: plain linear regression would give unstable coefficients, so regularized linear models (Lasso, Ridge, ElasticNet) were used. Removing the duplicate features was also tested later, with no effect (±0.0001).

## 8. Number-coded categories and rare categories

- **Looked at**: the meaning and frequency of each code.
- **Learned**:
  - `MSSubClass` (building class), `MoSold`, and `YrSold` are stored as numbers but are categories.
  - Some categories of `Condition2`, `RoofMatl`, and `Heating` appear in only a few houses.
- **Used**:
  - The three number-coded columns are converted to strings, then one-hot encoded.
  - Categories with fewer than 10 train houses are merged into `'Other'`.
  - In v2, one-hot columns where fewer than 5 train houses differ from the most common value are dropped (302 → 252 columns).

## Not in this notebook

- The residual analysis (which houses the models get most wrong) is in `notebooks/01_baseline_and_stack.ipynb` (#5-3, #5-4). The remaining error is concentrated in commercial zones, `Partial` sales during the 2008–2009 financial crisis, and abnormal sales.
- Adversarial validation between train and test (no distribution difference found): [TODO: notebook and cell location]
