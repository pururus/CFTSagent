"""Ранжирование по числу позитивных запросов."""
import json
from collections import Counter
from pathlib import Path
from .base_ranker import BaseRanker, Corpus, ModelPath, Qrels, Queries


class PopularityRanker(BaseRanker):
    """Подсчёт позитивных связей каждого документа."""
    def __init__(self, **kwargs: object) -> None:
        if kwargs:
            raise TypeError('Неизвестные параметры')
        self.scores = None

    def fit(self, corpus: Corpus, train_qrels: Qrels, *, train_queries: Queries | None = None) -> None:
        self.validate_input(corpus, train_qrels)
        scores = Counter({key: 0 for key in corpus})
        for docs in train_qrels.values():
            for doc, relevance in docs.items():
                if doc not in scores:
                    raise ValueError('Неизвестный документ')
                scores[doc] += int(relevance > 0)
        candidates = sorted(scores, key=lambda doc: (-scores[doc], doc))
        self.scores = scores
        self.candidates = candidates

    def prepare_candidates(self, corpus: Corpus) -> None:
        if self.scores is None:
            raise RuntimeError('Модель не подготовлена')
        self.validate_input(corpus)
        self.candidates = sorted(corpus, key=lambda doc: (-self.scores[doc], doc))

    def predict(self, query: str, top_k: int) -> list[int]:
        if self.scores is None:
            raise RuntimeError('Модель не подготовлена')
        if type(top_k) is not int or top_k < 0:
            raise ValueError('Недопустимый top_k')
        return self.candidates[:top_k]

    def save(self, path: ModelPath) -> None:
        if self.scores is None:
            raise RuntimeError('Модель не подготовлена')
        path = Path(path)
        path.mkdir()
        (path / 'model.json').write_text(json.dumps({'version': 2, 'type': 'popularity', 'scores': list(self.scores.items())}), encoding='utf-8')

    def load(self, path: ModelPath) -> None:
        try:
            data = json.loads((Path(path) / 'model.json').read_text(encoding='utf-8'))
            pairs = data['scores']
            if data['version'] != 2 or data['type'] != 'popularity' or not isinstance(pairs, list) or any(not isinstance(p, list) or len(p) != 2 or type(p[0]) is not int or not 0 <= p[0] < 2**63 or type(p[1]) is not int or p[1] < 0 for p in pairs):
                raise ValueError('Неверное состояние')
            scores = Counter(dict(pairs))
            if len(scores) != len(pairs): raise ValueError('Повтор ID')
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError('Неверное состояние') from exc
        candidates = sorted(scores, key=lambda doc: (-scores[doc], doc))
        self.scores = scores
        self.candidates = candidates
