"""원가법 구조 모델과 하이브리드(구조 모델 + Lasso 잔차 보정). notebooks/03의 #4, #6-1."""
import numpy as np
from scipy.optimize import least_squares
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

Q = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
SALE = ['Abnorml', 'Family', 'Alloca', 'Partial', 'AdjLand']
ZONE = ['C (all)', 'RM', 'RH', 'FV']
BLDG = ['2fmCon', 'Duplex', 'Twnhs', 'TwnhsE']
YRS = ['2007', '2008', '2009', '2010']
FOUND = ['CBlock', 'BrkTil', 'Slab', 'Stone', 'Wood']   # 기준: PConc

V4 = ('지하마감',)   # 실험(7-0)으로 고른 최종 항목
HYB_ALPHA = 0.0005


def prep_struct(d):
    """원본 컬럼(전처리 전)에서 구조 모델용 입력을 만든다."""
    d = d.copy()
    for c in ['TotalBsmtSF', 'GarageArea', 'BsmtFullBath', 'BsmtHalfBath',
              'BsmtFinSF1', 'BsmtFinSF2', 'GarageCars']:
        d[c] = d[c].fillna(0)
    ys = d['YrSold'].astype(int)
    d['age'] = (ys - d['YearBuilt']).clip(lower=0)
    d['remod_age'] = (ys - d['YearRemodAdd']).clip(lower=0)
    d['MSZoning'] = d['MSZoning'].fillna('RL')
    d['KQ'] = d['KitchenQual'].map(Q).fillna(3)
    d['EQ'] = d['ExterQual'].map(Q).fillna(3)
    d['BQ'] = d['BsmtQual'].fillna('None').map(Q)
    d['HQ'] = d['HeatingQC'].map(Q).fillna(3)
    d['FUNC'] = d['Functional'].fillna('Typ').map({'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3,
                                                   'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7})
    d['bath'] = d['FullBath'] + 0.5 * d['HalfBath'] + d['BsmtFullBath'] + 0.5 * d['BsmtHalfBath']
    d['porch'] = d[['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch', '3SsnPorch']].sum(axis=1)
    d['air'] = (d['CentralAir'] == 'Y').astype(float)
    d['bs_fin'] = d['BsmtFinSF1'] + d['BsmtFinSF2']
    d['YrSold_s'] = ys.astype(str)
    return d


class StructModel:
    """가격 = [땅값 + 건물값] x 전체 배수 + 부대시설, 그 위에 거래 조건·용도지역 % 보정.
    opts로 항목을 켜고 끔. opts=()이면 v3와 같은 모델."""
    EXTRA = {'2층단가': ['w_2nd'], '지하마감': ['w_fin'], '외장품질': ['g_e'], '지하품질': ['g_b'],
             '난방품질': ['g_h'], '기능상태': ['g_f'], '차고칸수': ['e_car'], '품질가속': ['g_q2'],
             '건물유형': ['t_' + c for c in BLDG], '판매연도': ['y_' + c for c in YRS],
             '기초공법': ['f_' + c for c in FOUND]}

    def __init__(self, nbs, opts=(), lam=0.3):
        self.idx = {n: i for i, n in enumerate(nbs)}; self.K = len(nbs)
        self.opts, self.lam = set(opts), lam
        names = [f'land{i}' for i in range(self.K)] + [f'bld{i}' for i in range(self.K)]
        names += ['b_lot', 'w_bs', 'w_gar', 'g_q', 'g_c', 'd_age', 'd_rem']
        names += ['s_' + c for c in SALE] + ['z_' + c for c in ZONE]
        names += ['e_bath', 'e_fp', 'e_porch', 'g_k', 'g_air']
        for o in sorted(self.opts):
            names += self.EXTRA[o]
        self.pos = {n: i for i, n in enumerate(names)}

    def _init(self):
        K = self.K; th = np.zeros(len(self.pos))
        th[:K], th[K:2*K] = np.log(30), np.log(80)
        init = dict(b_lot=0.5, w_bs=0.4, w_gar=0.3, g_q=0.15, g_c=0.03, d_age=0.005, d_rem=0.002,
                    e_bath=3, e_fp=3, e_porch=1, g_k=0.05, g_air=0.05, w_2nd=1.0, w_fin=0.2, e_car=3)
        for k, v in init.items():
            if k in self.pos: th[self.pos[k]] = v
        return th

    def _predict(self, th, d):
        p = lambda n: th[self.pos[n]]
        on = lambda o: o in self.opts
        K = self.K; i = d['Neighborhood'].map(self.idx).values
        land_val = np.exp(th[:K][i]) * (d['LotArea'].values / 10000) ** p('b_lot')
        # 면적: 지상 + 지하(+마감 추가 단가) + 차고
        if on('2층단가'):
            living = d['1stFlrSF'].values + p('w_2nd') * d['2ndFlrSF'].values + d['LowQualFinSF'].values
        else:
            living = d['GrLivArea'].values
        bsmt = p('w_bs') * d['TotalBsmtSF'].values
        if on('지하마감'): bsmt = bsmt + p('w_fin') * d['bs_fin'].values
        area = living + bsmt + p('w_gar') * d['GarageArea'].values
        # 건물 품질 (로그 배수)
        q = d['OverallQual'].values - 5
        qlog = p('g_q') * q + p('g_c') * (d['OverallCond'].values - 5)
        if on('품질가속'): qlog = qlog + p('g_q2') * q ** 2
        if on('외장품질'): qlog = qlog + p('g_e') * (d['EQ'].values - 3)
        if on('지하품질'): qlog = qlog + p('g_b') * (d['BQ'].values - 3) * (d['BQ'].values > 0)
        if on('난방품질'): qlog = qlog + p('g_h') * (d['HQ'].values - 5)
        dep = np.exp(-p('d_age') * d['age'].values - p('d_rem') * d['remod_age'].values)
        total = land_val + np.exp(th[K:2*K][i]) * area * np.exp(qlog) * dep / 1000
        # 전체 배수 + 부대시설(천 달러)
        mult = p('g_k') * (d['KQ'].values - 3) + p('g_air') * (d['air'].values - 1)
        if on('기능상태'): mult = mult + p('g_f') * (d['FUNC'].values - 7)
        total = (total * np.exp(mult) + p('e_bath') * d['bath'].values
                 + p('e_fp') * d['Fireplaces'].values + p('e_porch') * d['porch'].values / 100)
        if on('차고칸수'): total = total + p('e_car') * d['GarageCars'].values
        # % 보정
        adj = (sum(p('s_' + c) * (d['SaleCondition'].values == c) for c in SALE)
               + sum(p('z_' + c) * (d['MSZoning'].values == c) for c in ZONE))
        if on('건물유형'): adj = adj + sum(p('t_' + c) * (d['BldgType'].values == c) for c in BLDG)
        if on('판매연도'): adj = adj + sum(p('y_' + c) * (d['YrSold_s'].values == c) for c in YRS)
        if on('기초공법'): adj = adj + sum(p('f_' + c) * (d['Foundation'].values == c) for c in FOUND)
        return np.log1p(np.clip(total, 1, None) * np.exp(adj) * 1000)

    def fit(self, d, yv):
        K = self.K
        def residual(th):
            pen = 0.1 * np.sqrt(self.lam) * np.r_[th[:K] - th[:K].mean(), th[K:2*K] - th[K:2*K].mean()]
            return np.r_[self._predict(th, d) - yv, pen]
        self.th = least_squares(residual, self._init(), max_nfev=5000).x
        return self

    def predict(self, d):
        return self._predict(self.th, d)

    def params(self):
        return {n: self.th[i] for n, i in self.pos.items() if not n.startswith(('land', 'bld'))}


def struct_oof(S_train, y, nbs, opts, cv):
    o = np.zeros(len(y))
    for tr_i, va_i in cv.split(S_train):
        o[va_i] = StructModel(nbs, opts).fit(S_train.iloc[tr_i], y[tr_i]).predict(S_train.iloc[va_i])
    return o


def fit_hybrid(S_tr, X_tr, y_tr, w_tr, nbs, opts=V4):
    """구조 모델이 먼저 예측 → Lasso는 구조 모델의 잔차만 학습."""
    sm = StructModel(nbs, opts).fit(S_tr, y_tr)
    resid = y_tr - sm.predict(S_tr)
    lm = make_pipeline(RobustScaler(), Lasso(alpha=HYB_ALPHA, max_iter=50000))
    lm.fit(X_tr, resid, lasso__sample_weight=w_tr)
    return sm, lm


def predict_hybrid(sm, lm, S, X):
    return sm.predict(S) + lm.predict(X)


def hybrid_oof(S_train, X, y, w, nbs, cv, opts=V4):
    o = np.zeros(len(y))
    for tr_i, va_i in cv.split(X):
        sm, lm = fit_hybrid(S_train.iloc[tr_i], X.iloc[tr_i], y[tr_i], w[tr_i], nbs, opts)
        o[va_i] = predict_hybrid(sm, lm, S_train.iloc[va_i], X.iloc[va_i])
    return o
