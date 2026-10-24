English | [한국어](README.ko.md)

# House Prices – Advanced Regression Techniques

Predicting house sale prices in Ames, Iowa from 79 features. 
**Final public leaderboard: 0.11516** (log-price RMSE).

The result came from a blend of three models, a hybrid cost-approach model, and two hand-written rules based on the data. This repository records which changes reached the leaderboard and which did not, and the criteria used to decide.

## Problem and metric

- **Task**: regression of `SalePrice` for 1,459 test houses, trained on 1,460 houses ([Kaggle competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)).
- **Metric**: RMSE between log(predicted price) and log(actual price). Errors on cheap and expensive houses count by the same ratio.
- **Validation**: `KFold(5, shuffle=True, random_state=42)` on the training set after removing 2 outliers (Id 524, 1299). Feature changes were compared with `RepeatedKFold(5×3)`: 15 paired scores, and I counted how many of the 15 improved.

## EDA summary

Details: [eda/README.md](eda/README.md)

1. **The target is right-skewed** (skewness 1.88), so the models predict `log1p(SalePrice)`.
2. **Most categorical NAs mean "this facility does not exist"** (for example, no pool or no garage). They are filled with `'None'`, not with the mode.
3. **Id 524 and 1299 are outliers**: they have very large living areas and OverallQual 10, but low prices, and both are `Partial` sales (sold before construction was finished). They are removed from training.
4. **Test house Id 2550 matches the same condition** (GrLivArea 5,095 > 4,000 and a `Partial` sale). This is the basis for a separate rule (see below).

## Approach

### Step 3 – Baseline: start simple, then add a model with different strengths
- Lasso alone scored LB 0.12558; a 50:50 blend of Lasso and XGBoost scored LB 0.12427. Linear and tree models make their largest errors on different houses, so averaging the two predictions reduces the large errors of each.
- **Id 2550 rule**: In train, Id 524 and 1299 have over 4,000 sq ft of above-ground living area but sold far below what their size suggests, because they were Partial sales (sold before construction was complete). These two houses were removed from training as outliers. As a result, the models learned only that larger houses cost more, and every model predicted a very high price for Id 2550, the one test house with the same profile (over 4,000 sq ft and a Partial sale). The prediction for this house was therefore fixed to the mean log price of 524 and 1299. This single-house rule improved the LB from 0.12427 to **0.11876**.

### Step 4 – Upgrade: CV improved, the leaderboard did not
- A Lasso + XGBoost + SVR **stack** reached LB **0.11660**. A stack is a meta-model that learns how much weight to give each model's OOF predictions; OOF (out-of-fold) predictions are predictions for each training house made by a model that did not see that house.
- I then added interaction features, sample weights for abnormal sales, kNN, and a structure model. CV improved to 0.1058, but LB got worse (0.11715).
- Across 6 submissions, CV and LB had a correlation of **−0.79**: the better the CV, the worse the leaderboard.
- **Adversarial validation** showed no distribution difference between train and test. In adversarial validation, a classifier is trained to tell train rows from test rows; if it cannot, the two sets look alike. So the gap was not caused by a distribution shift. I concluded that the CV had become optimistic from repeatedly tuning on the same training data.

### Step 5 – New notebook: rebuild, and fix the decision rule before submitting
- **v2 features** (Box-Cox, more ordinal encodings, quality × area, removal of rare dummies). Every change applies to almost every house. Lasso CV went from 0.1117 to 0.1093 (better in 15 of 15 splits), and the stack reached LB **0.11535**.
- **v3** (squared quality term in the structure model) was rejected at LB 0.11713. The squared term was fitted on few training houses at the extremes, and it was unstable on test.
- **v4 hybrid**: a cost-approach structure model predicts the price first (land value + building area × unit cost × quality × depreciation), and Lasso learns only its residuals. Hybrid alone: CV 0.1054 (Lasso 0.1092, structure model 0.1103). Blended with XGBoost and SVR, it reached LB **0.11526**.
- **Corner adjustment**: +10% for the 6 test houses that are in IDOTRR, zoned commercial (`C (all)`), and on a corner lot. LB **0.11516** (best).
- **v4.1 and v5** were rejected. Whenever the structure-model family had more than 33% of the weight, LB got worse in every case. For v5, the accept/reject thresholds were written down before submitting.

## Results

Main versions only. The full log, with all CV and LB numbers and the reason for each decision, is in [docs/experiments.md](docs/experiments.md).

| Step | Version | CV (log RMSE) | Public LB | Decision |
|---|---|---|---|---|
| 3 | Lasso | 0.1116 | 0.12558 | baseline |
| 3 | Lasso + XGBoost blend | 0.1081 | 0.12427 | kept |
| 3 | + Id 2550 rule | – | 0.11876 | kept |
| 4 | Lasso + XGBoost + SVR stack | 0.1077 | 0.11660 | kept |
| 4 | + interactions, weights, kNN, structure model | 0.1058 | 0.11715 | rejected |
| 5 | v2 features, stack | 0.1049 | 0.11535 | kept |
| 5 | v3 structure model with squared term | 0.1035 | 0.11713 | rejected |
| 5 | v4 hybrid, blend | 0.1036 | 0.11526 | kept |
| 5 | **v4 blend + corner adjustment** | – | **0.11516** | **final** |

`–` means the change applies only to test houses, so CV cannot measure it. A blend is a simple average of model predictions with no learned weights.

## Key findings

1. **Three kinds of changes reached the leaderboard**: (1) rules made from looking at the data (Id 2550, corner adjustment), (2) feature cleanup that applies to almost every house (v2), and (3) blending models that make different errors, at moderate weights.
2. **CV and the leaderboard can move in opposite directions.** In Step 4, CV improved while LB got worse (correlation −0.79 over 6 submissions). After adversarial validation ruled out a distribution shift, I stopped trusting small CV gains on their own and changed the decision process:
   - compare changes on 15 paired splits, and count wins instead of only reading the mean;
   - check a model's weight in the stack as well as its solo score;
   - write the accept/reject LB thresholds before submitting (v5).
3. **Changes that fit a small part of the training data did not transfer.** A squared quality term fitted on extreme-quality houses, and a constraint for one neighborhood (IDOTRR), improved CV or test-prediction stability, but not LB.
4. **The remaining error is concentrated in houses whose price is explained by something outside the 79 features**: commercial zoning, `Partial` sales during the 2008–2009 financial crisis, and abnormal sales.

## Reproduce

**Environment**: the notebooks were run on Kaggle Notebooks. Locally:

```bash
pip install -r requirements.txt
```

**Data** (not included; competition rules do not allow redistribution). Accept the competition rules on Kaggle, then:

```bash
kaggle competitions download -c house-prices-advanced-regression-techniques -p data
unzip data/house-prices-advanced-regression-techniques.zip -d data
```

Each notebook looks for `/kaggle/input/**/train.csv` first and falls back to `../data/`.

**Order**:
1. `eda/eda.ipynb` – EDA
2. `notebooks/01_baseline_and_stack.ipynb` – Steps 3–4
3. `notebooks/02_v2_features.ipynb` – v2
4. `notebooks/03_v4_hybrid.ipynb` – final model. Cell #8 writes `submission_v4_blend.csv`, and cell #10 writes the final `submission_v4_blend_corner10.csv`.

The notebooks are kept as they were run, outputs included, as the record of each experiment. `src/` contains the same code from notebooks 02–03, organized into modules for reuse:

```python
from src.preprocessing import load_data, build
from src.structure_model import prep_struct, fit_hybrid
from src.rules import id2550_rule, corner_adjust
```

## Repository structure

```
house-prices-advanced-regression/
├── README.md / README.ko.md
├── requirements.txt
├── data/                          # not committed – download from Kaggle
├── eda/
│   ├── README.md / README.ko.md   # EDA summary
│   ├── eda.ipynb
│   └── figures/
├── notebooks/
│   ├── 01_baseline_and_stack.ipynb   # Steps 3–4
│   ├── 02_v2_features.ipynb          # Step 5 v2
│   └── 03_v4_hybrid.ipynb            # Step 5 v4 (final)
├── src/
│   ├── preprocessing.py   # missing values, outlier removal, transforms
│   ├── features.py        # feature creation
│   ├── models.py          # base models and hyperparameters
│   ├── structure_model.py # cost-approach structure model, hybrid
│   ├── cv.py              # KFold scoring, error correlation, stack vs blend
│   └── rules.py           # Id 2550 rule, corner adjustment
├── docs/
│   └── experiments.md     # full experiment log (Korean)
└── submissions/
    └── final_v4_blend_corner.csv
```
