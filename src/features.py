"""피처 생성. preprocessing.build()가 순서대로 호출한다."""

QUAL = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

HAS_COLS = ['PoolArea', '2ndFlrSF', 'GarageArea', 'TotalBsmtSF', 'MasVnrArea', 'Fireplaces',
            'MiscVal', '3SsnPorch', 'LowQualFinSF', 'BsmtFinSF2', 'ScreenPorch', 'EnclosedPorch',
            'WoodDeckSF', 'OpenPorchSF']


def add_basic_features(a):
    """총면적·욕실 수·연식 + 시설 유무(Has*). 결측 처리 뒤, 인코딩 전에 호출."""
    ys = a['YrSold'].astype(int)
    a['TotalSF'] = a['TotalBsmtSF'] + a['1stFlrSF'] + a['2ndFlrSF']
    a['TotalBath'] = a['FullBath'] + 0.5 * a['HalfBath'] + a['BsmtFullBath'] + 0.5 * a['BsmtHalfBath']
    a['HouseAge'] = (ys - a['YearBuilt']).clip(lower=0)
    a['RemodAge'] = (ys - a['YearRemodAdd']).clip(lower=0)
    a['TotalPorchSF'] = (a['OpenPorchSF'] + a['EnclosedPorch'] + a['3SsnPorch']
                         + a['ScreenPorch'] + a['WoodDeckSF'])
    for c in HAS_COLS:
        a['Has' + c] = (a[c] > 0).astype(int)
    return a


def add_interactions(a):
    """[v2] 품질×면적. Box-Cox 변환 뒤에 호출 (노트북과 같은 순서)."""
    a['Qual_SF'] = a['OverallQual'] * a['TotalSF']
    a['Qual_Liv'] = a['OverallQual'] * a['GrLivArea']
    return a
