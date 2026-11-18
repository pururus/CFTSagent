"""CF на семействах train-запросов, выделенных по эмбеддингам."""

import math
from typing import Any

import numpy as np
from scipy import sparse
from numpy.typing import NDArray

from .cf_base import CFBaseRanker


class FamilyQueryKNNCFRanker(CFBaseRanker):
    """Кластеризация, агрегация взаимодействий и факторизация семейств."""

    variant = 'family'

    def __init__(self, *, clustering: str = 'kmeans', n_families: int = 32, eps: float = 0.2,
                 min_samples: int = 5, aggregation: str = 'binary', count_transform: str = 'log1p', **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if clustering not in {'kmeans', 'minibatch_kmeans', 'dbscan'}:
            raise ValueError('Неизвестный метод кластеризации')
        if aggregation not in {'binary', 'frequency'} or count_transform not in {'count', 'log1p', 'sqrt'}:
            raise ValueError('Неизвестная агрегация или функция количества')
        if type(n_families) is not int or n_families < 1 or type(min_samples) is not int or min_samples < 1:
            raise ValueError('Размеры кластеризации должны быть положительными целыми')
        if isinstance(eps, bool) or not isinstance(eps, (float, int)) or not math.isfinite(eps) or eps <= 0:
            raise ValueError('eps должен быть положительным')
        self.config.update(clustering=clustering, n_families=n_families, eps=eps,
                           min_samples=min_samples, aggregation=aggregation,
                           count_transform=count_transform)

    def _make_rows(self, interactions: sparse.csr_matrix, embeddings: NDArray[np.floating]) -> tuple[sparse.csr_matrix, NDArray[np.float32], NDArray[np.int64]]:
        from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans

        cfg = self.config
        if cfg['clustering'] == 'dbscan':
            labels = DBSCAN(eps=cfg['eps'], min_samples=cfg['min_samples'], metric='cosine').fit_predict(embeddings)
            next_label = int(labels.max()) + 1
            for index in np.flatnonzero(labels == -1):
                labels[index] = next_label
                next_label += 1
        else:
            if cfg['n_families'] > len(embeddings):
                raise ValueError('n_families превышает число train-запросов')
            algorithm = KMeans if cfg['clustering'] == 'kmeans' else MiniBatchKMeans
            labels = algorithm(n_clusters=cfg['n_families'], n_init=10, random_state=cfg['seed']).fit_predict(embeddings)
        _, labels = np.unique(labels, return_inverse=True)
        n_families = int(labels.max()) + 1
        membership = sparse.csr_matrix(
            (np.ones(len(labels), dtype=np.float32), (labels, np.arange(len(labels)))),
            shape=(n_families, len(labels)),
        )
        targets = (membership @ interactions).tocsr()
        if cfg['aggregation'] == 'binary':
            targets.data[:] = 1.0
        elif cfg['count_transform'] == 'log1p':
            targets.data = np.log1p(targets.data)
        elif cfg['count_transform'] == 'sqrt':
            targets.data = np.sqrt(targets.data)
        sizes = np.asarray(membership.sum(axis=1)).ravel()
        centroids = self._normalized((membership @ embeddings) / sizes[:, None])
        return targets, centroids, labels.astype(np.int64)
