"""Оптимизация факторов CF и выбор негативов только из train API."""

import numpy as np
from typing import Callable
from numpy.typing import NDArray
from scipy import sparse


def mse_objective(targets, rows, items, regularization, positive_weight):
    """Посчитать MSE с учётом всех нулей через матрицы Грама."""
    rows = np.asarray(rows, dtype=np.float64)
    items = np.asarray(items, dtype=np.float64)
    entries = targets.tocoo()
    predicted = np.einsum('ij,ij->i', rows[entries.row], items[entries.col])
    squared = np.sum((rows.T @ rows) * (items.T @ items))
    squared += np.sum(entries.data ** 2 - 2 * entries.data * predicted)
    squared += (positive_weight - 1) * np.sum((entries.data - predicted) ** 2)
    penalty = regularization * (np.sum(rows ** 2) + np.sum(items ** 2))
    return float((max(0.0, squared) + penalty) / np.prod(targets.shape))


def fit_als(targets, factors, epochs, regularization, positive_weight, seed,
            progress_callback: Callable[[int, float], None] | None = None):
    """ALS для взвешенной квадратичной ошибки по всей train-матрице."""
    rng = np.random.default_rng(seed)
    items = rng.normal(0, 0.1, (targets.shape[1], factors))
    rows = np.zeros((targets.shape[0], factors), dtype=np.float64)
    history = []

    def update(matrix, fixed):
        gram = fixed.T @ fixed + regularization * np.eye(factors)
        result = np.zeros((matrix.shape[0], factors), dtype=np.float64)
        for row in range(matrix.shape[0]):
            start, end = matrix.indptr[row:row + 2]
            indices, values = matrix.indices[start:end], matrix.data[start:end]
            selected = fixed[indices]
            system = gram + (positive_weight - 1) * (selected.T @ selected)
            right = positive_weight * (selected.T @ values)
            result[row] = np.linalg.solve(system, right)
        return result

    transposed = targets.T.tocsr()
    for _ in range(epochs):
        rows = update(targets, items)
        items = update(transposed, rows)
        history.append(mse_objective(targets, rows, items, regularization, positive_weight))
        if progress_callback is not None:
            progress_callback(len(history), history[-1])
    return rows.astype(np.float32), items.astype(np.float32), history


class NegativeSampler:
    """Случайные негативы или выбор среди H наиболее удалённых train API.

    Входные эмбеддинги нормированы, поэтому меньшее скалярное
    произведение соответствует большему cosine-расстоянию.
    """

    def __init__(self, targets: sparse.csr_matrix, mode: str, rng: np.random.Generator,
                 row_embeddings: NDArray[np.floating] | None = None,
                 api_embeddings: NDArray[np.floating] | None = None, hard_pool_size: int = 20, batch_size: int = 128) -> None:
        self.targets = targets.tocsr()
        self.mode = mode
        self.rng = rng
        self.n_items = targets.shape[1]
        self.positives = [set(targets.indices[targets.indptr[r]:targets.indptr[r + 1]])
                          for r in range(targets.shape[0])]
        self.pools = {}
        if mode == 'semantic':
            for offset in range(0, targets.shape[0], batch_size):
                scores = row_embeddings[offset:offset + batch_size] @ api_embeddings.T
                for local, values in enumerate(scores):
                    row = offset + local
                    valid = np.ones(self.n_items, dtype=bool)
                    valid[list(self.positives[row])] = False
                    ids = np.flatnonzero(valid)
                    order = np.argsort(values[ids], kind='stable')
                    self.pools[row] = ids[order[:hard_pool_size]]

    def sample(self, row: int) -> int | None:
        """Вернуть индекс отрицательного train API или None, если их нет."""
        positives = self.positives[row]
        if len(positives) == self.n_items:
            return None
        if self.mode == 'semantic':
            return int(self.rng.choice(self.pools[row]))
        for _ in range(32):
            candidate = int(self.rng.integers(self.n_items))
            if candidate not in positives:
                return candidate
        available = np.array([i for i in range(self.n_items) if i not in positives])
        return int(self.rng.choice(available))


def fit_gradient(targets, row_embeddings, api_embeddings, *, projected,
                 loss, sampling, factors, epochs, regularization,
                 positive_weight, learning_rate, batch_size, negatives_per_positive,
                 hard_pool_size, seed, device,
                 progress_callback: Callable[[int, float], None] | None = None):
    """Adam для BPR и для проекционного MSE, вычисляемого батчами."""
    import torch
    from torch.nn import functional as functional

    rng = np.random.default_rng(seed)
    n_rows, n_items = targets.shape
    shape = (row_embeddings.shape[1], factors) if projected else (n_rows, factors)
    left = torch.nn.Parameter(torch.tensor(rng.normal(0, 0.1, shape), dtype=torch.float32, device=device))
    items = torch.nn.Parameter(torch.tensor(rng.normal(0, 0.1, (n_items, factors)), dtype=torch.float32, device=device))
    embeddings = torch.tensor(row_embeddings, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([left, items], lr=learning_rate)
    entries = targets.tocoo()
    sampler = None
    eligible = np.empty(0, dtype=np.int64)
    positive_mean = 1.0
    if loss == 'bpr':
        sampler = NegativeSampler(targets, sampling, rng, row_embeddings,
                                  api_embeddings, hard_pool_size, batch_size)
        eligible = np.array([i for i, row in enumerate(entries.row)
                             if len(sampler.positives[row]) < n_items], dtype=np.int64)
        if not len(eligible):
            raise ValueError('BPR требует хотя бы одну строку с позитивом и негативом')
        positive_mean = float(entries.data[eligible].mean())
    history = []
    for _ in range(epochs):
        order = rng.permutation(n_rows) if loss == 'mse' else rng.permutation(
            np.repeat(eligible, negatives_per_positive))
        total_loss, total_examples = 0.0, 0
        for offset in range(0, len(order), batch_size):
            batch = order[offset:offset + batch_size]
            if loss == 'mse':
                row_ids = torch.tensor(batch, dtype=torch.long, device=device)
                row_vectors = embeddings[row_ids] @ left
                truth = torch.tensor(targets[batch].toarray(), dtype=torch.float32, device=device)
                weights = torch.where(truth > 0, positive_weight, 1.0)
                objective = (weights * (row_vectors @ items.T - truth).square()).mean()
                objective += regularization * (left.square().sum() + items.square().sum()) / (n_rows * n_items)
            else:
                row_indices = entries.row[batch]
                row_ids = torch.tensor(row_indices, dtype=torch.long, device=device)
                positive = torch.tensor(entries.col[batch], dtype=torch.long, device=device)
                negative = torch.tensor([sampler.sample(int(row)) for row in row_indices], dtype=torch.long, device=device)
                row_vectors = embeddings[row_ids] @ left if projected else left[row_ids]
                differences = (row_vectors * (items[positive] - items[negative])).sum(dim=1)
                weights = torch.tensor(entries.data[batch], dtype=torch.float32, device=device)
                # Масштабирование глобальным средним сохраняет веса агрегации между батчами.
                objective = (weights * functional.softplus(-differences)).mean() / positive_mean
                objective += regularization * (left.square().mean() + items.square().mean())
            if not torch.isfinite(objective):
                raise ValueError('Обучение CF дало нечисловую функцию потерь')
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()
            total_loss += float(objective.detach().cpu()) * len(batch)
            total_examples += len(batch)
        history.append(total_loss / total_examples)
        if progress_callback is not None:
            progress_callback(len(history), history[-1])
    return left.detach().cpu().numpy(), items.detach().cpu().numpy(), history
