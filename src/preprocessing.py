"""결측 처리, 이상치 제거, 변환. notebooks/02·03의 #1 전처리 함수(build)를 옮긴 것."""
import glob
import os

import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax, skew

from .features import QUAL, add_basic_features, add_interactions

OUTLIER_IDS = [524, 1299]   # 면적은 매우 크지만 Partial 거래로 가격이 낮은 2채

NONE_COLS = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
             'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']
ZERO_COLS = ['GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
             'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
MODE_COLS = ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']
SPARSE_COLS = ['MiscVal', '3SsnPorch', 'LowQualFinSF', 'BsmtFinSF2',
               'ScreenPorch', 'EnclosedPorch', 'PoolArea']   # 대부분 0 → Has* 유무만 남김


def find_data_dir(local='data'):
    """Kaggle이면 /kaggle/input 아래, 로컬이면 local 폴더."""
    cands = glob.glob('/kaggle/input/**/train.csv', recursive=True)
    return os.path.dirname(cands[0]) if cands else local


def load_data(data_dir=None):
    data_dir = data_dir or find_data_dir()
    return pd.read_csv(f'{data_dir}/train.csv'), pd.read_csv(f'{data_dir}/test.csv')


def fill_missing(a, n):
    """a: train+test를 합친 데이터, n: 앞쪽 train 행 수 (최빈값은 train으로만 계산)."""
    for c in NONE_COLS:   # NA = 해당 시설 없음
        a[c] = a[c].fillna('None')
    for c in ZERO_COLS:
        a[c] = a[c].fillna(0)
    a['GarageYrBlt'] = a['GarageYrBlt'].fillna(a['YearBuilt'])
    a.loc[a['GarageYrBlt'] > 2010, 'GarageYrBlt'] = a['YearBuilt']   # test의 2207년 오타
    a['LotFrontage'] = a.groupby('Neighborhood')['LotFrontage'].transform(lambda s: s.fillna(s.median()))
    a['Functional'] = a['Functional'].fillna('Typ')
    for c in MODE_COLS:
        a[c] = a[c].fillna(a[:n][c].mode()[0])
    a = a.drop(columns=['Utilities'])
    for c in ['MSSubClass', 'MoSold', 'YrSold']:
        a[c] = a[c].astype(str)
    return a


def encode_ordinal(a, more_ord=True):
    for c in ['ExterQual', 'KitchenQual', 'BsmtQual', 'FireplaceQu', 'HeatingQC']:
        a[c] = a[c].map(QUAL)
    a['GarageFinish'] = a['GarageFinish'].map({'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3})
    a['BsmtExposure'] = a['BsmtExposure'].map({'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4})
    if more_ord:   # [v2] 추가
        for c in ['GarageQual', 'GarageCond', 'BsmtCond', 'PoolQC']:
            a[c] = a[c].map(QUAL)
        a['Functional'] = a['Functional'].map({'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3,
                                               'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7})
        a['CentralAir'] = (a['CentralAir'] == 'Y').astype(int)
        a['PavedDrive'] = a['PavedDrive'].map({'N': 0, 'P': 1, 'Y': 2})
    return a


def group_rare(a, n):
    """대부분 0인 컬럼 삭제 + train에서 10채 미만인 범주는 'Other'로."""
    a = a.drop(columns=SPARSE_COLS)
    for c in ['Condition2', 'RoofMatl', 'Heating']:
        cnt = a[:n][c].value_counts()
        a[c] = a[c].where(a[c].isin(cnt[cnt >= 10].index), 'Other')
    return a


def transform_skewed(a, n, boxcox=True):
    if boxcox:   # [v2] 왜도 0.5 이상인 연속형 전부 Box-Cox, λ는 train으로만 추정
        num = [c for c in a.select_dtypes(include='number').columns
               if a[c].nunique() > 15 and not c.startswith('Has')
               and c not in ('YearBuilt', 'YearRemodAdd', 'GarageYrBlt')]
        sk = a[:n][num].apply(lambda s: skew(s))
        for c in sk[sk.abs() > 0.5].index:
            if (a[c] < 0).any():
                continue
            try:
                lam = boxcox_normmax(a[:n][c] + 1, method='mle')
            except Exception:
                lam = 0.0   # 추정 실패하면 로그 변환과 같음
            a[c] = boxcox1p(a[c], lam)
    else:        # v1 방식
        for c in ['LotArea', 'LotFrontage', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF',
                  '1stFlrSF', 'GrLivArea', 'TotalSF', 'MasVnrArea', 'TotalPorchSF']:
            a[c] = np.log1p(a[c])
    return a


def one_hot(a, n, drop_rare=True):
    a = pd.get_dummies(a).astype(float)
    if drop_rare:   # [v2] train에서 5채 미만만 다른 값 → 과적합 재료
        diff = (a[:n] != a[:n].mode().iloc[0]).sum()
        a = a.drop(columns=diff[diff < 5].index)
    return a


def build(train, test, boxcox=True, more_ord=True, interact=True, drop_rare=True):
    """옵션을 모두 False로 하면 v1, 기본값이 v2. 반환: X, X_test, y(log1p), 이상치 제거된 train."""
    train = train[~train['Id'].isin(OUTLIER_IDS)].reset_index(drop=True)
    y = np.log1p(train['SalePrice'])
    n = len(train)
    a = pd.concat([train.drop(columns=['Id', 'SalePrice']),
                   test.drop(columns=['Id'])]).reset_index(drop=True)

    a = fill_missing(a, n)
    a = add_basic_features(a)
    a = encode_ordinal(a, more_ord)
    a = group_rare(a, n)
    a = transform_skewed(a, n, boxcox)
    if interact:   # [v2] 기본 탑재
        a = add_interactions(a)
    a = one_hot(a, n, drop_rare)

    X, X_test = a[:n].reset_index(drop=True), a[n:].reset_index(drop=True)
    return X, X_test, y, train
