"""베이스 모델과 하이퍼파라미터. notebooks/03의 #3, #5."""
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor


def make_lasso():
    return make_pipeline(RobustScaler(), Lasso(alpha=0.0005, max_iter=50000))


def make_svr():
    return make_pipeline(RobustScaler(), SVR(C=15, epsilon=0.008, gamma=0.0005))


def make_xgb(seed=42):
    return XGBRegressor(learning_rate=0.02, n_estimators=2500, max_depth=3, subsample=0.7,
                        colsample_bytree=0.2, min_child_weight=3, random_state=seed, n_jobs=-1)


def abnormal_weights(train, w=0.5):
    """비정상 거래(SaleCondition != Normal) 학습 가중치 0.5. Lasso에만 쓴다."""
    return np.where((train['SaleCondition'] != 'Normal').values, w, 1.0)
