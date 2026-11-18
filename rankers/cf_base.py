"""Общие данные, инференс и сохранение трёх моделей CF."""

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import sparse

from numpy.typing import ArrayLike, NDArray

from .base_ranker import BaseRanker, Corpus, Encoder, ModelPath, Qrels, Queries
from .bge import ZeroShotBGERanker
from .cf_training import fit_als, fit_gradient


class CFBaseRanker(BaseRanker):
    """Факторы CF с текстовым поиском train-соседей нового запроса."""

    variant = 'query'

    def __init__(self, *, loss: str = 'mse', negative_sampling: str = 'random', factors: int = 32,
                 epochs: int = 20, regularization: float = 0.1, positive_weight: float = 1.0,
                 learning_rate: float = 0.01, batch_size: int = 128, n_neighbors: int = 10,
                 neighbor_weighting: str = 'cosine', negatives_per_positive: int = 1,
                 hard_pool_size: int = 20, seed: int = 42, device: str = 'cpu', encoder: Encoder | None = None,
                 model_path: ModelPath | None = None, revision: str | None = None, encoder_device: str | None = None,
                 query_instruction: str = 'Represent this sentence for searching relevant passages: ',
                 max_length: int = 512, encoder_batch_size: int = 32) -> None:
        if loss not in {'mse', 'bpr'} or negative_sampling not in {'random', 'semantic'}:
            raise ValueError('Неизвестная цель или стратегия семплинга')
        if neighbor_weighting not in {'cosine', 'uniform'}:
            raise ValueError('Неизвестный способ взвешивания соседей')
        integers = (factors, epochs, batch_size, n_neighbors,
                    negatives_per_positive, hard_pool_size, max_length, encoder_batch_size)
        if any(type(value) is not int or value < 1 for value in integers):
            raise ValueError('Размеры и количества должны быть положительными целыми')
        if type(seed) is not int or seed < 0:
            raise ValueError('seed должен быть неотрицательным целым')
        for value in (regularization, positive_weight, learning_rate):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('Регуляризация, вес и шаг должны быть положительными')
        if not isinstance(query_instruction, str):
            raise ValueError('Инструкция должна быть строкой')
        self.config = dict(
            loss=loss, negative_sampling=negative_sampling, factors=factors,
            epochs=epochs, regularization=regularization, positive_weight=positive_weight,
            learning_rate=learning_rate, batch_size=batch_size, n_neighbors=n_neighbors,
            neighbor_weighting=neighbor_weighting, negatives_per_positive=negatives_per_positive,
            hard_pool_size=hard_pool_size, seed=seed, device=str(device),
            model_path=str(model_path) if model_path else None, revision=revision,
            encoder_device=str(encoder_device or device), query_instruction=query_instruction,
            max_length=max_length, encoder_batch_size=encoder_batch_size,
        )
        self._embedder = ZeroShotBGERanker(
            model_path=model_path, revision=revision, device=encoder_device or device,
            query_instruction=query_instruction, max_length=max_length,
            batch_size=encoder_batch_size, encoder=encoder,
        )
        self._fitted = False

    @staticmethod
    def _normalized(matrix: ArrayLike) -> NDArray[np.float32]:
        matrix = np.asarray(matrix)
        if matrix.dtype.kind not in 'fiu':
            raise ValueError('Ожидаются вещественные числовые эмбеддинги')
        matrix = matrix.astype(np.float64)
        if matrix.ndim != 2 or not matrix.shape[1] or not np.isfinite(matrix).all():
            raise ValueError('Ожидается конечная двумерная матрица эмбеддингов')
        # Сначала уменьшаем масштаб: квадрат компонент не переполняется.
        scale = np.max(np.abs(matrix), axis=1, keepdims=True)
        scaled = np.divide(matrix, scale, out=np.zeros_like(matrix), where=scale != 0)
        norm = np.linalg.norm(scaled, axis=1, keepdims=True)
        return np.divide(scaled, norm, out=np.zeros_like(scaled), where=norm != 0).astype(np.float32)

    def encode_queries(self, texts: Iterable[str]) -> NDArray[np.float32]:
        """Кодировать запросы одним энкодером для обучения и инференса."""
        return self._encode(texts, query=True)

    def encode_apis(self, descriptions: Iterable[str]) -> NDArray[np.float32]:
        """Кодировать описания API для семантических негативов."""
        return self._encode(descriptions, query=False)

    def _encode(self, texts: Iterable[str], *, query: bool) -> NDArray[np.float32]:
        texts = list(texts)
        if not texts or any(not isinstance(text, str) for text in texts):
            raise ValueError('Нужен непустой список строк для кодирования')
        prefix = self.config['query_instruction'] if query else ''
        batches = []
        size = self.config['encoder_batch_size']
        for offset in range(0, len(texts), size):
            batch = [prefix + text for text in texts[offset:offset + size]]
            encoded = self._normalized(self._embedder._encoder().encode(batch))
            if len(encoded) != len(batch):
                raise ValueError('Энкодер изменил количество строк')
            batches.append(encoded)
        return np.concatenate(batches)

    def _cached_embeddings(self, ids: NDArray[np.int64], texts: Iterable[str], cache: Mapping[int, ArrayLike] | None, *, query: bool) -> NDArray[np.float32]:
        if cache is None:
            return self._encode(texts, query=query)
        if (not isinstance(cache, Mapping)
                or any(type(key) is not int or not 0 <= key < 2**63 for key in cache)
                or set(cache) != set(ids)):
            raise ValueError('Кэш должен содержать ровно переданные train ID')
        return self._normalized([cache[identifier] for identifier in ids])

    def _make_rows(self, interactions: sparse.csr_matrix, embeddings: NDArray[np.floating]) -> tuple[sparse.csr_matrix, NDArray[np.floating], NDArray[np.int64]]:
        return interactions, embeddings, np.arange(interactions.shape[0], dtype=np.int64)

    def fit(self, corpus: Corpus, train_qrels: Qrels, *, train_queries: Queries | None = None,
            train_query_embeddings: Mapping[int, ArrayLike] | None = None, train_api_embeddings: Mapping[int, ArrayLike] | None = None) -> None:
        """Обучить только train-данные, при необходимости используя кэши по ID."""
        self.validate_input(corpus, train_qrels)
        if not corpus or not isinstance(train_queries, Mapping) or not train_queries:
            raise ValueError('CF требует непустые train_queries и train-корпус')
        if any(type(key) is not int or key < 0 or key > np.iinfo(np.int64).max or not isinstance(text, str)
               for key, text in train_queries.items()):
            raise ValueError('train_queries должны сопоставлять int64 ID строкам')
        if any(key > np.iinfo(np.int64).max for key in corpus):
            raise ValueError('API_id выходит за диапазон int64')
        if not set(train_qrels) <= set(train_queries):
            raise ValueError('В qrels есть запрос вне train_queries')

        query_ids = np.array(sorted(train_queries), dtype=np.int64)
        api_ids = np.array(sorted(corpus), dtype=np.int64)
        api_index = {int(api_id): i for i, api_id in enumerate(api_ids)}
        rows, columns = [], []
        for row, request_id in enumerate(query_ids):
            for api_id, relevance in train_qrels.get(int(request_id), {}).items():
                if relevance > 0:
                    rows.append(row)
                    columns.append(api_index[api_id])
        interactions = sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, columns)),
            shape=(len(query_ids), len(api_ids)),
        )
        embeddings = self._cached_embeddings(
            query_ids, [train_queries[int(q)] for q in query_ids],
            train_query_embeddings, query=True,
        )
        targets, row_embeddings, query_rows = self._make_rows(interactions, embeddings)
        api_embeddings = None
        if self.config['loss'] == 'bpr' and self.config['negative_sampling'] == 'semantic':
            api_embeddings = self._cached_embeddings(
                api_ids, [corpus[int(i)]['description'] for i in api_ids],
                train_api_embeddings, query=False,
            )
            if api_embeddings.shape[1] != embeddings.shape[1]:
                raise ValueError('Эмбеддинги API и запросов имеют разные размерности')

        cfg = self.config
        if cfg['loss'] == 'mse' and self.variant != 'projected':
            left, items, history = fit_als(
                targets, cfg['factors'], cfg['epochs'], cfg['regularization'],
                cfg['positive_weight'], cfg['seed'],
            )
        else:
            left, items, history = fit_gradient(
                targets, row_embeddings, api_embeddings,
                projected=self.variant == 'projected', loss=cfg['loss'],
                sampling=cfg['negative_sampling'], factors=cfg['factors'],
                epochs=cfg['epochs'], regularization=cfg['regularization'],
                positive_weight=cfg['positive_weight'], learning_rate=cfg['learning_rate'],
                batch_size=cfg['batch_size'], negatives_per_positive=cfg['negatives_per_positive'],
                hard_pool_size=cfg['hard_pool_size'], seed=cfg['seed'], device=cfg['device'],
            )
        if not np.isfinite(left).all() or not np.isfinite(items).all():
            raise ValueError('Неконечные факторы после обучения')
        # Публикуем новое состояние после успешной подготовки и оптимизации.
        self.query_ids_, self.api_ids_ = query_ids, api_ids
        self.query_embeddings_, self.row_embeddings_ = embeddings, row_embeddings
        self.query_rows_, self.targets_ = query_rows, targets
        self.left_factors_, self.item_factors_ = left, items
        self.loss_history_ = history
        self._fitted = True
        self.prepare_candidates(corpus)

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError('Сначала выполните fit или load')

    def prepare_candidates(self, corpus: Corpus) -> None:
        """Сопоставить полный каталог с обученными факторами без переобучения."""
        self._require_fit()
        self.validate_input(corpus)
        if any(key > np.iinfo(np.int64).max for key in corpus):
            raise ValueError('API_id выходит за диапазон int64')
        self.candidate_ids_ = np.array(sorted(corpus), dtype=np.int64)
        lookup = {int(identifier): i for i, identifier in enumerate(self.api_ids_)}
        self.candidate_indices_ = np.array([lookup.get(int(i), -1) for i in self.candidate_ids_])

    def _query_factor(self, embedding: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.variant == 'projected':
            return embedding @ self.left_factors_
        similarity = self.query_embeddings_ @ embedding
        count = min(self.config['n_neighbors'], len(similarity))
        neighbors = np.lexsort((self.query_ids_, -similarity))[:count]
        weights = np.maximum(similarity[neighbors], 0)
        if self.config['neighbor_weighting'] == 'uniform' or not weights.sum():
            weights = np.ones(count, dtype=np.float32)
        weights = weights / weights.sum()
        # Повторные семейства автоматически получают суммарный вес соседей.
        return weights @ self.left_factors_[self.query_rows_[neighbors]]

    def predict_from_embedding(self, embedding: ArrayLike, top_k: int) -> list[int]:
        """Ранжировать с уже рассчитанным эмбеддингом того же энкодера."""
        self._require_fit()
        if type(top_k) is not int or top_k < 0:
            raise ValueError('Недопустимый top_k')
        vector = np.asarray(embedding)
        if vector.ndim != 1 or len(vector) != self.query_embeddings_.shape[1]:
            raise ValueError('Неверная размерность эмбеддинга запроса')
        vector = self._normalized(vector[None, :])[0]
        if top_k == 0 or len(self.candidate_ids_) == 0:
            return []
        factor = self._query_factor(vector)
        scores = np.zeros(len(self.candidate_ids_), dtype=np.float32)
        known = self.candidate_indices_ >= 0
        scores[known] = self.item_factors_[self.candidate_indices_[known]] @ factor
        order = np.lexsort((self.candidate_ids_, -scores))[:top_k]
        return [int(i) for i in self.candidate_ids_[order]]

    def predict(self, query: str, top_k: int) -> list[int]:
        self._require_fit()
        if not isinstance(query, str):
            raise ValueError('Запрос должен быть строкой')
        if type(top_k) is not int or top_k < 0:
            raise ValueError('Недопустимый top_k')
        if top_k == 0 or not len(self.candidate_ids_):
            return []
        return self.predict_from_embedding(self.encode_queries([query])[0], top_k)

    def save(self, path: ModelPath) -> None:
        """Сохранить train-состояние в JSON и NPZ без pickle."""
        self._require_fit()
        path = Path(path)
        metadata = dict(version=1, variant=self.variant, config=self.config,
                        target_shape=list(self.targets_.shape), loss_history=self.loss_history_)
        text = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)
        path.mkdir()
        np.savez_compressed(
            path / 'factors.npz', query_ids=self.query_ids_, api_ids=self.api_ids_,
            query_embeddings=self.query_embeddings_, row_embeddings=self.row_embeddings_,
            query_rows=self.query_rows_, left_factors=self.left_factors_,
            item_factors=self.item_factors_, target_data=self.targets_.data,
            target_indices=self.targets_.indices, target_indptr=self.targets_.indptr,
        )
        (path / 'model.json').write_text(text, encoding='utf-8')

    def load(self, path: ModelPath) -> None:
        """Проверить сохранённые массивы до замены состояния экземпляра."""
        path = Path(path)
        try:
            metadata = json.loads((path / 'model.json').read_text(encoding='utf-8'))
            if metadata['version'] != 1 or metadata['variant'] != self.variant:
                raise ValueError('Несовместимая версия или вариант CF')
            candidate = type(self)(encoder=self._embedder.encoder, **metadata['config'])
            with np.load(path / 'factors.npz', allow_pickle=False) as data:
                arrays = {name: data[name].copy() for name in data.files}
            for name in ('query_embeddings', 'row_embeddings', 'left_factors',
                         'item_factors', 'target_data'):
                if arrays[name].dtype.kind not in 'fiu':
                    raise ValueError('Сохранённые значения должны быть вещественными')
            for name in ('target_indices', 'target_indptr'):
                if arrays[name].ndim != 1 or arrays[name].dtype.kind not in 'iu':
                    raise ValueError('Индексы CSR должны быть целочисленными')
            query_ids, api_ids = arrays['query_ids'], arrays['api_ids']
            for ids in (query_ids, api_ids):
                if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or not len(ids)
                        or (ids < 0).any() or (ids > np.iinfo(np.int64).max).any()
                        or len(np.unique(ids)) != len(ids)):
                    raise ValueError('Недопустимые сохранённые ID')
            embeddings, row_embeddings = arrays['query_embeddings'], arrays['row_embeddings']
            left, items, query_rows = arrays['left_factors'], arrays['item_factors'], arrays['query_rows']
            if embeddings.ndim != 2 or embeddings.shape[0] != len(query_ids) or not embeddings.shape[1]:
                raise ValueError('Неверный размер train-эмбеддингов')
            if row_embeddings.ndim != 2 or row_embeddings.shape[1] != embeddings.shape[1] or not len(row_embeddings):
                raise ValueError('Неверный размер эмбеддингов строк')
            expected_left = embeddings.shape[1] if self.variant == 'projected' else len(row_embeddings)
            factors = candidate.config['factors']
            if left.shape != (expected_left, factors) or items.shape != (len(api_ids), factors):
                raise ValueError('Неверный размер факторов')
            if query_rows.shape != (len(query_ids),) or query_rows.dtype.kind not in 'iu' or (query_rows < 0).any() or (query_rows >= len(row_embeddings)).any():
                raise ValueError('Неверное соответствие запросов и строк')
            if any(not np.isfinite(a).all() for a in (embeddings, row_embeddings, left, items, arrays['target_data'])):
                raise ValueError('Неконечные сохранённые значения')
            shape = tuple(metadata['target_shape'])
            if shape != (len(row_embeddings), len(api_ids)):
                raise ValueError('Неверная форма матрицы взаимодействий')
            targets = sparse.csr_matrix((arrays['target_data'], arrays['target_indices'], arrays['target_indptr']), shape=shape)
            targets.check_format(full_check=True)
            if (targets.data <= 0).any():
                raise ValueError('Неверные позитивные взаимодействия')
            history = metadata['loss_history']
            if not isinstance(history, list) or len(history) != candidate.config['epochs'] or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in history):
                raise ValueError('Неверная история обучения')
            candidate.query_ids_, candidate.api_ids_ = query_ids, api_ids
            candidate.query_embeddings_, candidate.row_embeddings_ = embeddings, row_embeddings
            candidate.query_rows_, candidate.targets_ = query_rows, targets
            candidate.left_factors_, candidate.item_factors_ = left, items
            candidate.loss_history_ = history
            candidate._fitted = True
            candidate.prepare_candidates({int(i): {'description': ''} for i in api_ids})
        except (KeyError, TypeError, IndexError, json.JSONDecodeError) as exc:
            raise ValueError('Повреждённое состояние CF') from exc
        self.__dict__.update(candidate.__dict__)
