"""Лексический ранжировщик BM25."""
import json
import math
import re
from collections import Counter
from pathlib import Path
from .base_ranker import BaseRanker


class BM25Ranker(BaseRanker):
    """BM25 с положительным IDF и уникальными токенами запроса."""
    def __init__(self, k1=1.5, b=0.75):
        if not math.isfinite(k1) or not math.isfinite(b) or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError('Недопустимые параметры BM25')
        self.k1, self.b, self.documents = k1, b, None

    @staticmethod
    def _tokens(text):
        return re.findall(r'\w+', text.casefold())

    def fit(self, corpus, train_qrels, *, train_queries=None):
        self.validate_input(corpus, train_qrels)
        if any(doc not in corpus for docs in train_qrels.values() for doc in docs):
            raise ValueError('Неизвестный документ')
        self.documents = {key: Counter(self._tokens(value['description'])) for key, value in corpus.items()}
        self.candidates = self.documents

    def prepare_candidates(self, corpus):
        if self.documents is None:
            raise RuntimeError('Модель не подготовлена')
        self.validate_input(corpus)
        self.candidates = {key: Counter(self._tokens(value['description'])) for key, value in corpus.items()}

    def predict(self, query, top_k):
        if self.documents is None:
            raise RuntimeError('Модель не подготовлена')
        if type(top_k) is not int or top_k < 0:
            raise ValueError('Недопустимый top_k')
        n = len(self.documents)
        lengths = {doc: sum(counts.values()) for doc, counts in self.candidates.items()}
        avg = sum(sum(counts.values()) for counts in self.documents.values()) / n if n else 0
        scores = dict.fromkeys(self.candidates, 0.0)
        if avg:
            for term in set(self._tokens(query)):
                df = sum(term in counts for counts in self.documents.values())
                idf = math.log1p((n - df + .5) / (df + .5))
                for doc, counts in self.candidates.items():
                    f = counts.get(term, 0)
                    if f:
                        scores[doc] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * lengths[doc] / avg))
        return sorted(scores, key=lambda doc: (-scores[doc], doc))[:top_k]

    def save(self, path):
        if self.documents is None:
            raise RuntimeError('Модель не подготовлена')
        path = Path(path)
        path.mkdir()
        (path / 'model.json').write_text(json.dumps({'version': 2, 'type': 'bm25', 'k1': self.k1, 'b': self.b, 'documents': list(self.documents.items())}), encoding='utf-8')

    def load(self, path):
        try:
            data = json.loads((Path(path) / 'model.json').read_text(encoding='utf-8'))
            if data['version'] != 2 or data['type'] != 'bm25' or not isinstance(data['documents'], list):
                raise ValueError('Неверное состояние')
            candidate = BM25Ranker(data['k1'], data['b'])
            for doc, counts in data['documents']:
                if type(doc) is not int or not 0 <= doc < 2**63 or not isinstance(counts, dict) or any(not isinstance(k, str) or type(v) is not int or v <= 0 for k, v in counts.items()):
                    raise ValueError('Неверный индекс')
            candidate.documents = {doc: Counter(counts) for doc, counts in data['documents']}
            if len(candidate.documents) != len(data['documents']): raise ValueError('Повтор ID')
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError('Неверное состояние') from exc
        self.k1, self.b, self.documents = candidate.k1, candidate.b, candidate.documents
        self.candidates = self.documents
