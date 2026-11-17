"""CF с линейной проекцией замороженного эмбеддинга запроса."""

from .cf_base import CFBaseRanker


class ProjectedQueryCFRanker(CFBaseRanker):
    """Обучает W и факторы API через MSE или BPR с Adam.

    В коде эмбеддинги записаны строками, поэтому p = e @ W.
    Это транспонированная запись p = W e для векторов-столбцов.
    """

    variant = 'projected'
