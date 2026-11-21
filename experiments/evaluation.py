"""Оценка обученных моделей на полном каталоге одного датасета."""
from pathlib import Path
from time import perf_counter
import resource
import sys

import numpy as np
import pandas as pd

METRICS = ('Recall', 'NDCG', 'FullCoverage')
CUTOFFS = (3, 5, 10, 20)


def metric_matrix(rankings, positives, kmax=20):
    """Значения [запрос, метрика, K-1]. Пустая разметка обозначается NaN."""
    hits = np.zeros((len(positives), kmax), dtype=float)
    counts = np.asarray([len(items) for items in positives])
    for i, (ranking, relevant) in enumerate(zip(rankings, positives, strict=True)):
        top = ranking[:kmax]
        hits[i, :len(top)] = [api in relevant for api in top]
    discounts = 1 / np.log2(np.arange(2, kmax + 2))
    cumulative = hits.cumsum(axis=1)
    ideal = np.cumsum(discounts)[np.minimum(counts[:, None], np.arange(1, kmax + 1)).clip(1) - 1]
    values = np.stack((cumulative / counts.clip(1)[:, None],
                       (hits * discounts).cumsum(axis=1) / ideal,
                       cumulative == counts[:, None]), axis=1)
    values[counts == 0] = np.nan
    return values


def curve_frame(name, values):
    index = pd.MultiIndex.from_product([[name], METRICS, range(1, values.shape[-1] + 1)],
                                       names=['model', 'metric', 'k'])
    return pd.Series(values.reshape(-1), index=index, name='value').reset_index()


def paired_intervals(scores, seed=68, repetitions=1000, reference=None):
    """Парные интервалы относительно выбранной модели (по умолчанию BGE, если доступна)."""
    names = list(scores)
    cube = np.stack(list(scores.values()), axis=1)
    count = len(cube)
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, repetitions, 16):
        weights = rng.multinomial(count, np.full(count, 1 / count),
                                  size=min(16, repetitions - start))
        draws.append((weights / count) @ cube.reshape(count, -1))
    means = np.concatenate(draws).reshape(repetitions, *cube.shape[1:])
    if reference is None:
        reference = 'BGE' if 'BGE' in scores else names[0]
    ref = names.index(reference)
    differences = means - means[:, ref:ref + 1]
    index = pd.MultiIndex.from_product([names, METRICS, CUTOFFS], names=['model','metric','k'])
    point = cube.mean(axis=0)
    return pd.DataFrame(dict(reference=reference, mean=point.ravel(),
                             low=np.quantile(means, .025, axis=0).ravel(),
                             high=np.quantile(means, .975, axis=0).ravel(),
                             delta=(point - point[ref]).ravel(),
                             delta_low=np.quantile(differences, .025, axis=0).ravel(),
                             delta_high=np.quantile(differences, .975, axis=0).ravel()),
                        index=index).reset_index()


def evaluate_suite(models, data, split, output, train_api_ids, bootstrap_seed=68):
    """Подготовить только кандидатов, оценить на val/test и сохранить результаты."""
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    queries = data.queries[split]
    ids = sorted(queries)
    labels = [set(data.qrels[split].get(i, {})) for i in ids]
    active = np.array([bool(items) for items in labels], dtype=bool)
    if not active.any():
        raise ValueError(f'В {split} нет запросов с положительными связями')
    catalogue = set(data.corpus)
    coverage = pd.DataFrame([
        (i, len(items), len(items & catalogue), len(items & train_api_ids))
        for i, items in zip(ids, labels)
    ], columns=['Request_id', 'positives', 'in_catalogue', 'in_train_history'])
    coverage.to_csv(directory / 'coverage_by_query.csv', index=False)
    coverage_summary = dict(queries=len(ids), evaluated_queries=int(active.sum()),
                            excluded_no_positive=int((~active).sum()),
                            catalogue_size=len(catalogue), known_candidates=len(catalogue & train_api_ids))
    for column in ('in_catalogue', 'in_train_history'):
        coverage_summary[column + '_macro'] = (coverage.loc[active,column] / coverage.loc[active,'positives']).mean()
        coverage_summary[column + '_micro'] = coverage[column].sum() / coverage.positives.sum()
    pd.DataFrame([coverage_summary]).to_csv(directory / 'coverage.csv', index=False)
    curves, scores, timings = [], {}, []
    for number, (name, model) in enumerate(models.items()):
        start = perf_counter()
        model.prepare_candidates(data.corpus)
        ready = perf_counter()
        rankings = [model.predict(queries[i], max(CUTOFFS)) for i in ids]
        predicted = perf_counter()
        values = metric_matrix(rankings, labels)[active]
        curves.append(curve_frame(name, values.mean(axis=0)))
        scores[name] = values[:, :, np.array(CUTOFFS) - 1]
        filename = f'predictions_{number:02}.npz'
        np.savez_compressed(directory / filename, request_ids=ids, ranked_api_ids=rankings)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        timings.append(dict(model=name, predictions=filename, prepare_s=ready-start,
                            predict_s=predicted-ready,
                            process_peak_rss_bytes=peak if sys.platform == 'darwin' else peak*1024))
    pd.DataFrame(timings).to_csv(directory / 'runtime.csv', index=False)
    paired_intervals(scores, seed=bootstrap_seed).to_csv(directory / 'paired_intervals.csv', index=False)
    return pd.concat(curves, ignore_index=True)
