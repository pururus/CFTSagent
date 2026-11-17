"""Факторизация запрос–API и KNN-представление нового запроса."""

from .cf_base import CFBaseRanker


class QueryKNNCFRanker(CFBaseRanker):
    """MSE/ALS либо BPR, с усреднением факторов train-соседей."""

    variant = 'query'
