"""Ранжирование по числу позитивных запросов."""
import json
from pathlib import Path
from .base_ranker import BaseRanker


class PopularityRanker(BaseRanker):
    """Подсчёт позитивных связей каждого документа."""
    def __init__(self, **kwargs):
        if kwargs:
            raise TypeError('Неизвестные параметры')
        self.scores = None

    def fit(self, corpus, train_qrels, *, train_queries=None):
        self.validate_input(corpus, train_qrels)
        scores = {key: 0 for key in corpus}
        for docs in train_qrels.values():
            for doc, relevance in docs.items():
                if doc not in scores:
                    raise ValueError('Неизвестный документ')
                scores[doc] += int(relevance > 0)
        self.scores = scores
        self.candidates = list(scores)

    def prepare_candidates(self, corpus):
        if self.scores is None:
            raise RuntimeError('Модель не подготовлена')
        self.validate_input(corpus)
        self.candidates = list(corpus)

    def predict(self, query, top_k):
        if self.scores is None:
            raise RuntimeError('Модель не подготовлена')
        if type(top_k) is not int or top_k < 0:
            raise ValueError('Недопустимый top_k')
        return sorted(self.candidates, key=lambda doc: (-self.scores.get(doc, 0), doc))[:top_k]

    def save(self, path):
        if self.scores is None:
            raise RuntimeError('Модель не подготовлена')
        path = Path(path)
        path.mkdir()
        (path / 'model.json').write_text(json.dumps({'version': 2, 'type': 'popularity', 'scores': list(self.scores.items())}), encoding='utf-8')

    def load(self, path):
        try:
            data = json.loads((Path(path) / 'model.json').read_text(encoding='utf-8'))
            pairs = data['scores']
            if data['version'] != 2 or data['type'] != 'popularity' or not isinstance(pairs, list) or any(not isinstance(p, list) or len(p) != 2 or type(p[0]) is not int or not 0 <= p[0] < 2**63 or type(p[1]) is not int or p[1] < 0 for p in pairs):
                raise ValueError('Неверное состояние')
            scores = dict(pairs)
            if len(scores) != len(pairs): raise ValueError('Повтор ID')
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError('Неверное состояние') from exc
        self.scores = scores
        self.candidates = list(scores)
