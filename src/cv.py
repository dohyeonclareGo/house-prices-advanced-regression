"""KFold 평가, 오차 상관, 스택/blend 비교. notebooks/03의 #2, #6, #7."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_predict, cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)            # OOF용
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)   # 피처 비교용 (15번)


def rmse(y, p):
    return np.sqrt(((np.asarray(y) - np.asarray(p)) ** 2).mean())


def cv_scores(model, X, y, cv=rkf):
    """반복 교차검증 15번의 fold별 RMSE. 두 설정을 같은 분할로 짝지어 비교할 때 쓴다."""
    return -cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')


def paired_compare(model, X_base, X_new, y, cv=rkf):
    """평균 RMSE와 '15번 중 몇 번 좋아졌나'를 같이 본다."""
    base, new = cv_scores(model, X_base, y, cv), cv_scores(model, X_new, y, cv)
    return {'base': base.mean(), 'new': new.mean(), 'wins': f'{(new < base).sum()}/{len(base)}'}


def oof_predictions(models, X, y, cv=kf):
    """models: {이름: (모델, fit params 또는 None)}. 모든 모델을 같은 분할로 OOF 예측."""
    oof = pd.DataFrame(index=range(len(y)))
    for name, (m, params) in models.items():
        oof[name] = cross_val_predict(m, X, y, cv=cv, params=params)
    return oof


def error_corr(oof, y):
    """모델 간 오차 상관. 낮을수록 다르게 틀린다 = 섞었을 때 효과가 크다."""
    return oof.sub(np.asarray(y), axis=0).corr()


def stack_vs_blend(oof, y, cols, n_seeds=10):
    """스택(양수 가중치 선형회귀, 메타 학습도 바깥 CV로 채점)과 단순 평균 비교."""
    meta = LinearRegression(positive=True)
    stack_s = [rmse(y, cross_val_predict(meta, oof[cols], y, cv=KFold(5, shuffle=True, random_state=s)))
               for s in range(n_seeds)]
    meta.fit(oof[cols], y)
    return {'stack': np.mean(stack_s), 'stack_std': np.std(stack_s),
            'blend': rmse(y, oof[cols].mean(axis=1)),
            'weights': dict(zip(cols, meta.coef_.round(2))), 'meta': meta}
