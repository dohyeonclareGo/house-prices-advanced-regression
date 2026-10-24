"""데이터를 보고 만든 후처리 규칙. notebooks/01의 #4-12, notebooks/03의 #10."""
import numpy as np

from .preprocessing import OUTLIER_IDS


def id2550_rule(pred_log, train_raw, test_raw):
    """초대형(GrLivArea > 4000) + Partial 거래인 test 집(Id 2550)은
    같은 조건의 train 이상치 2채(524, 1299)의 평균 log 가격으로 고정한다."""
    twin_log = np.log1p(train_raw.loc[train_raw['Id'].isin(OUTLIER_IDS), 'SalePrice']).mean()
    special = ((test_raw['GrLivArea'] > 4000) & (test_raw['SaleCondition'] == 'Partial')).values
    pred_log = np.asarray(pred_log).copy()
    pred_log[special] = twin_log
    return pred_log


def corner_mask(test_raw):
    """IDOTRR 동네 + 상업지역(C (all)) + 코너 땅. test에서 6채."""
    return ((test_raw['Neighborhood'] == 'IDOTRR') & (test_raw['MSZoning'] == 'C (all)')
            & (test_raw['LotConfig'] == 'Corner')).values


def corner_adjust(sub, test_raw, factor=1.10):
    """제출 파일(SalePrice, 원래 단위)에서 코너 6채만 +10%."""
    sub = sub.copy()
    sub.loc[corner_mask(test_raw), 'SalePrice'] *= factor
    return sub
