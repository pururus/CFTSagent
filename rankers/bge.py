"""Семантический ранжировщик на замороженном локальном энкодере."""
import json
import numpy as np
from pathlib import Path
from numpy.typing import ArrayLike, NDArray
from .base_ranker import BaseRanker, Corpus, Encoder, ModelPath, Qrels, Queries


class ZeroShotBGERanker(BaseRanker):
    """Cosine-поиск с локальным BGE или внедрённым энкодером."""
    def __init__(self, model_path: ModelPath | None = None, revision: str | None = None, query_instruction: str = '', max_length: int = 512, batch_size: int = 32, encoder: Encoder | None = None, device: str = 'cuda') -> None:
        if type(batch_size) is not int or batch_size < 1 or type(max_length) is not int or max_length < 1:
            raise ValueError('Недопустимый размер')
        self.model_path, self.revision = str(model_path) if model_path else None, revision
        self.query_instruction, self.max_length, self.batch_size = query_instruction, max_length, batch_size
        self.device = device
        self.encoder, self.vectors = encoder, None

    def _encoder(self) -> Encoder:
        if self.encoder is None:
            if not self.model_path or not Path(self.model_path).is_dir():
                raise FileNotFoundError('Локальные веса не найдены')
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.model_path, local_files_only=True, revision=self.revision, device=self.device)
            self.encoder.max_seq_length = self.max_length
        return self.encoder

    @staticmethod
    def _normalize_rows(values: ArrayLike) -> NDArray[np.float64]:
        matrix = np.asarray(values)
        if matrix.dtype.kind not in 'fiu' or matrix.ndim != 2 or not matrix.shape[1]:
            raise ValueError('Ожидается вещественная матрица эмбеддингов')
        matrix = matrix.astype(np.float64, copy=False)
        if not np.isfinite(matrix).all():
            raise ValueError('Неверный вектор')
        scale = np.max(np.abs(matrix), axis=1, keepdims=True)
        scaled = np.divide(matrix, scale, out=np.zeros_like(matrix), where=scale != 0)
        norm = np.linalg.norm(scaled, axis=1, keepdims=True)
        return np.divide(scaled, norm, out=np.zeros_like(scaled), where=norm != 0)

    @staticmethod
    def _normalize(vector: ArrayLike) -> NDArray[np.float64]:
        np_vector = np.asarray(vector)
        if np_vector.dtype.kind not in 'fiu' or np_vector.ndim != 1:
            raise ValueError('Ожидается вещественный вектор')
        np_vector = np_vector.astype(np.float64, copy=False)
        if not np.isfinite(np_vector).all():
            raise ValueError('Неверный вектор')
        scale = np.max(np.abs(np_vector), axis=0, keepdims=True)
        scaled = np.divide(np_vector, scale, out=np.zeros_like(np_vector), where=scale != 0)
        norm = np.linalg.norm(scaled, axis=0, keepdims=True)
        return np.divide(scaled, norm, out=np.zeros_like(scaled), where=norm != 0)

    def _encode_corpus(self, corpus: Corpus, dimension: int = 0) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        ids = np.array(list(corpus), dtype=np.int64)
        batches = []
        for offset in range(0, len(ids), self.batch_size):
            texts = [corpus[int(key)]['description'] for key in ids[offset:offset + self.batch_size]]
            encoded = self._encoder().encode(texts)
            if not hasattr(encoded, '__len__'):
                encoded = list(encoded)
            matrix = self._normalize_rows(encoded)
            if len(matrix) != len(texts):
                raise ValueError('Неверное число векторов')
            if dimension and matrix.shape[1] != dimension:
                raise ValueError('Несогласованные размерности')
            dimension = matrix.shape[1]
            batches.append(matrix)
        matrix = np.concatenate(batches) if batches else np.empty((0, dimension), dtype=np.float64)
        return ids, matrix

    def fit(self, corpus: Corpus, train_qrels: Qrels, *, train_queries: Queries | None = None) -> None:
        self.validate_input(corpus, train_qrels)
        ids, vectors = self._encode_corpus(corpus)
        self.api_ids_, self.vectors = ids, vectors
        self.candidate_ids_, self.candidates = ids, vectors

    def prepare_candidates(self, corpus: Corpus) -> None:
        if self.vectors is None:
            raise RuntimeError('Модель не подготовлена')
        self.validate_input(corpus)
        ids, vectors = self._encode_corpus(corpus, self.vectors.shape[1])
        self.candidate_ids_, self.candidates = ids, vectors

    def predict(self, query: str, top_k: int) -> list[int]:
        if self.vectors is None:
            raise RuntimeError('Модель не подготовлена')
        if type(top_k) is not int or top_k < 0:
            raise ValueError('Недопустимый top_k')
        if not top_k or not len(self.candidate_ids_):
            return []
        scores = np.zeros(len(self.candidate_ids_), dtype=np.float64)
        if query:
            encoded = self._encoder().encode([self.query_instruction + query])
            if not hasattr(encoded, '__len__'):
                encoded = list(encoded)
            matrix = self._normalize_rows(encoded)
            if matrix.shape != (1, self.candidates.shape[1]):
                raise ValueError('Несогласованные размерности')
            scores = self.candidates @ matrix[0]
        order = np.lexsort((self.candidate_ids_, -scores))[:top_k]
        return self.candidate_ids_[order].tolist()

    def save(self, path: ModelPath) -> None:
        if self.vectors is None:
            raise RuntimeError('Модель не подготовлена')
        path = Path(path)
        path.mkdir()
        data = {'version': 2, 'type': 'bge', 'model_path': self.model_path, 'revision': self.revision, 'query_instruction': self.query_instruction, 'max_length': self.max_length, 'batch_size': self.batch_size, 'device': self.device, 'vectors': [[int(key), row.tolist()] for key, row in zip(self.api_ids_, self.vectors)]}
        (path / 'model.json').write_text(json.dumps(data, allow_nan=False), encoding='utf-8')

    def load(self, path: ModelPath) -> None:
        try:
            data = json.loads((Path(path) / 'model.json').read_text(encoding='utf-8'))
            if data['version'] != 2 or data['type'] != 'bge' or not isinstance(data['vectors'], list):
                raise ValueError('Неверное состояние')
            candidate = ZeroShotBGERanker(data['model_path'], data['revision'], data['query_instruction'], data['max_length'], data['batch_size'], self.encoder, data.get('device', 'cuda'))
            if candidate.encoder is None and (not candidate.model_path or not Path(candidate.model_path).is_dir()):
                raise FileNotFoundError('Локальные веса не найдены')
            pairs = data['vectors']
            if any(not isinstance(pair, list) or len(pair) != 2
                   or type(pair[0]) is not int or not 0 <= pair[0] < 2**63 for pair in pairs):
                raise ValueError('Недопустимые ID')
            ids = np.array([pair[0] for pair in pairs], dtype=np.int64)
            if len(np.unique(ids)) != len(ids):
                raise ValueError('Повтор ID')
            vectors = self._normalize_rows([pair[1] for pair in pairs]) if pairs else np.empty((0, 0))
            candidate.api_ids_, candidate.vectors = ids, vectors
            candidate.candidate_ids_, candidate.candidates = ids, vectors
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError('Неверное состояние') from exc
        self.__dict__.update(candidate.__dict__)
