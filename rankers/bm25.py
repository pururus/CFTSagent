"""Лексический ранжировщик BM25."""
import json
import math
import re
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Mapping
from numpy.typing import NDArray
from .base_ranker import BaseRanker, Corpus, ModelPath, Qrels, Queries


class BM25Ranker(BaseRanker):
    """BM25 с положительным IDF и уникальными токенами запроса."""
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if not math.isfinite(k1) or not math.isfinite(b) or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError('Недопустимые параметры BM25')
        self.k1, self.b, self.documents = k1, b, None

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r'\w+', text.casefold())

    def _prepare_index(self, documents: Mapping[int, Counter[str]]) -> tuple[NDArray[np.int64], dict[str, tuple[NDArray[np.intp], NDArray[np.float64]]]]:
        ids = np.array(list(documents), dtype=np.int64)
        postings = {}
        if self.avgdl_:
            for index, counts in enumerate(documents.values()):
                length = sum(counts.values())
                adjustment = self.k1 * (1 - self.b + self.b * length / self.avgdl_)
                for term, frequency in counts.items():
                    df = self.df_.get(term, 0)
                    idf = math.log1p((self.n_documents_ - df + .5) / (df + .5))
                    score = idf * frequency * (self.k1 + 1) / (frequency + adjustment)
                    indices, values = postings.setdefault(term, ([], []))
                    indices.append(index)
                    values.append(score)
        postings = {term: (np.asarray(indices, dtype=np.intp), np.asarray(values, dtype=np.float64))
                    for term, (indices, values) in postings.items()}
        return ids, postings

    def _set_documents(self, documents: Mapping[int, Counter[str]]) -> None:
        self.documents = documents
        self.n_documents_ = len(documents)
        self.df_ = Counter()
        total_length = 0
        for counts in documents.values():
            self.df_.update(counts.keys())
            total_length += sum(counts.values())
        self.avgdl_ = total_length / self.n_documents_ if self.n_documents_ else 0
        self.candidate_ids_, self.postings_ = self._prepare_index(documents)
        self.candidates = documents

    def fit(self, corpus: Corpus, train_qrels: Qrels, *, train_queries: Queries | None = None) -> None:
        self.validate_input(corpus, train_qrels)
        documents = {key: Counter(self._tokens(value['description'])) for key, value in corpus.items()}
        candidate = BM25Ranker(self.k1, self.b)
        candidate._set_documents(documents)
        self.__dict__.update(candidate.__dict__)

    def prepare_candidates(self, corpus: Corpus) -> None:
        if self.documents is None:
            raise RuntimeError('Модель не подготовлена')
        self.validate_input(corpus)
        documents = {key: Counter(self._tokens(value['description'])) for key, value in corpus.items()}
        ids, postings = self._prepare_index(documents)
        self.candidates = documents
        self.candidate_ids_, self.postings_ = ids, postings

    def predict(self, query: str, top_k: int) -> list[int]:
        if self.documents is None:
            raise RuntimeError('Модель не подготовлена')
        if type(top_k) is not int or top_k < 0:
            raise ValueError('Недопустимый top_k')
        if not top_k or not len(self.candidate_ids_):
            return []
        scores = np.zeros(len(self.candidate_ids_), dtype=np.float64)
        for term in sorted(set(self._tokens(query))):
            posting = self.postings_.get(term)
            if posting is not None:
                indices, values = posting
                scores[indices] += values
        order = np.lexsort((self.candidate_ids_, -scores))[:top_k]
        return self.candidate_ids_[order].tolist()

    def save(self, path: ModelPath) -> None:
        if self.documents is None:
            raise RuntimeError('Модель не подготовлена')
        path = Path(path)
        path.mkdir()
        (path / 'model.json').write_text(json.dumps({'version': 2, 'type': 'bm25', 'k1': self.k1, 'b': self.b, 'documents': list(self.documents.items())}), encoding='utf-8')

    def load(self, path: ModelPath) -> None:
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
            candidate._set_documents(candidate.documents)
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError('Неверное состояние') from exc
        self.__dict__.update(candidate.__dict__)
