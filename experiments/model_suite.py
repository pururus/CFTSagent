"""Конфигурации и обучение общего набора ранжировщиков."""

from dataclasses import dataclass
from copy import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from rankers.bge import ZeroShotBGERanker
from rankers.bm25 import BM25Ranker
from rankers.cf_base import CFBaseRanker
from rankers.family_query_knn_cf import FamilyQueryKNNCFRanker
from rankers.popularity import PopularityRanker
from rankers.projected_query_cf import ProjectedQueryCFRanker
from rankers.query_knn_cf import QueryKNNCFRanker


QUERY_INSTRUCTION = 'Represent this sentence for searching relevant passages: '
FAMILY_NEIGHBORS = (1, 2, 3, 5, 10)
OBJECTIVES = {
    'MSE': dict(loss='mse'),
    'BPR/random': dict(loss='bpr', negative_sampling='random'),
    'BPR/semantic': dict(loss='bpr', negative_sampling='semantic'),
}


@dataclass
class ModelSpec:
    """Фабрика модели и параметры, фиксируемые до тестирования."""

    factory: type
    config: dict


def resolve_device(choice='auto'):
    """Выбрать доступное устройство для энкодера и оптимизатора Adam."""
    import torch

    if choice == 'cpu':
        return 'cpu'
    if choice == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        return 'mps' if torch.backends.mps.is_available() else 'cpu'
    if choice == 'cuda' and torch.cuda.is_available():
        return choice
    if choice == 'mps' and torch.backends.mps.is_available():
        return choice
    if choice in ('cuda', 'mps'):
        raise RuntimeError(f'Устройство {choice} недоступно в текущем ядре Python. Выберите auto или cpu.')
    raise ValueError('Укажите одно из устройств: auto, cpu, cuda, mps')


def cf_config(seed=67, *, device='cpu'):
    """Общие параметры девяти фиксированных вариантов CF."""
    return dict(factors=16, epochs=5, regularization=.1, positive_weight=1.,
                n_neighbors=10, neighbor_weighting='cosine', seed=seed,
                device=device, encoder_device=device, learning_rate=.01,
                batch_size=128, query_instruction=QUERY_INSTRUCTION,
                max_length=512, encoder_batch_size=32,
                negatives_per_positive=1, hard_pool_size=20)


def fixed_specs(seed=67, *, device='cpu'):
    """Три бейзлайна и девять фиксированных конфигураций CF."""
    specs = {
        'Popularity': ModelSpec(PopularityRanker, {}),
        'BM25': ModelSpec(BM25Ranker, dict(k1=1.5, b=.75)),
        'BGE': ModelSpec(ZeroShotBGERanker, dict(
            device=device, max_length=512, batch_size=32,
            query_instruction=QUERY_INSTRUCTION)),
    }
    families = {
        'Query-KNN CF': (QueryKNNCFRanker, {}),
        'Family CF': (FamilyQueryKNNCFRanker, dict(
            clustering='minibatch_kmeans', n_families=64, aggregation='binary')),
        'Projected CF': (ProjectedQueryCFRanker, {}),
    }
    for family, (factory, options) in families.items():
        for objective, loss_config in OBJECTIVES.items():
            specs[f'{family} | {objective}'] = ModelSpec(
                factory, {**cf_config(seed, device=device), **options, **loss_config})
    return specs


def family_specs(seed=67, *, device='cpu'):
    """48 обучений для выбора Family CF на validation."""
    clusterings = {
        **{f'MiniBatchKMeans-{size}': dict(clustering='minibatch_kmeans', n_families=size)
           for size in (32, 64, 128)},
        'DBSCAN': dict(clustering='dbscan', eps=.2, min_samples=5),
    }
    aggregations = {
        'binary': dict(aggregation='binary'),
        **{name: dict(aggregation='frequency', count_transform=name)
           for name in ('count', 'sqrt', 'log1p')},
    }
    specs = {}
    for clustering, cluster_config in clusterings.items():
        for aggregation, aggregation_config in aggregations.items():
            for objective, loss_config in OBJECTIVES.items():
                name = f'Family CF | {clustering} | {aggregation} | {objective}'
                specs[name] = ModelSpec(FamilyQueryKNNCFRanker, {
                    **cf_config(seed, device=device), **cluster_config, **aggregation_config, **loss_config})
    return specs


def train_suite(data, encoder, specs=None, seed=67):
    """Обучить модели исключительно на train и вернуть время и историю потерь.

    Полный каталог подготавливает общий оценщик перед каждым val/test.
    Объект data предоставляет corpus, queries и qrels с ключами split.
    """
    if specs is None:
        specs = fixed_specs(seed)
    train_qrels = data.qrels['train']
    train_ids = {api for links in train_qrels.values() for api, rel in links.items() if rel > 0}
    train_corpus = {api: data.corpus[api] for api in sorted(train_ids)}
    models, records = {}, []
    for name, spec in specs.items():
        config = dict(spec.config)
        if issubclass(spec.factory, (CFBaseRanker, ZeroShotBGERanker)):
            config['encoder'] = encoder
        model = spec.factory(**config)
        started = perf_counter()
        model.fit(train_corpus, train_qrels, train_queries=data.queries['train'])
        records.append(dict(model=name, fit_seconds=perf_counter() - started,
                            loss_history=getattr(model, 'loss_history_', [])))
        models[name] = model
    return models, pd.DataFrame(records)


def family_neighbors(model, data, split='val', max_neighbors=10):
    """Один раз найти train-соседей для всей сетки Family CF."""
    query_ids = list(data.queries[split])
    vectors = model.encode_queries([data.queries[split][key] for key in query_ids])
    # predict_from_embedding повторно нормализует эмбеддинг перед поиском.
    vectors = model._normalized(vectors)
    neighbor_rows, similarities = [], []
    for vector in vectors:
        scores = model.query_embeddings_ @ vector
        rows = np.lexsort((model.query_ids_, -scores))[:max_neighbors]
        neighbor_rows.append(rows)
        similarities.append(scores[rows])
    return query_ids, np.asarray(neighbor_rows), np.asarray(similarities)


def family_validation_summary(model, data, name, neighbors, cache, split='val',
                              artifact_dir=None, reference=None):
    """Ранжировать кандидатов с готовыми соседями и усреднить метрики."""
    from experiments.evaluation import metric_matrix

    started = perf_counter()
    query_ids, rows, similarities = cache
    rankings = []
    known = model.candidate_indices_ >= 0
    known_factors = model.item_factors_[model.candidate_indices_[known]]
    if not known.any():
        ranking = model.candidate_ids_[:20].tolist()
        rankings = [ranking] * len(query_ids)
    else:
        rankings = _family_rankings(model, rows, similarities, neighbors, known, known_factors)
    values = metric_matrix(rankings, [data.qrels[split][key] for key in query_ids], kmax=20)
    if artifact_dir is not None:
        _save_family_artifacts(model, name, neighbors, split, query_ids, rankings,
                               values, artifact_dir, reference, perf_counter() - started)
    counts = np.isfinite(values).sum(axis=0)
    means = np.divide(np.nansum(values, axis=0), counts,
                      out=np.full(counts.shape, np.nan), where=counts > 0)
    return pd.DataFrame([
        dict(model=name, metric=metric, k=k + 1, value=float(means[index, k]))
        for index, metric in enumerate(('Recall', 'NDCG', 'FullCoverage'))
        for k in range(20)
    ])


def _save_family_artifacts(model, name, neighbors, split, query_ids, rankings,
                           values, artifact_dir, reference, seconds):
    """Сохранить выдачи, метрики и парные нормальные интервалы по запросам.

    reference — общий изменяемый словарь с model, Request_id и metrics.
    Пустой словарь фиксирует первый допустимый вариант как базовый.
    Метрики имеют форму (число запросов, 3, 4), K=(3,5,10,20).
    """
    directory = Path(artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    ks = np.array([3, 5, 10, 20])
    compact = values[:, :, ks - 1]
    ids = np.asarray(query_ids, dtype=np.int64)
    if reference is None:
        reference = {}
    if not reference:
        reference.update(model=name, Request_id=ids.copy(), metrics=compact.copy())
    if not np.array_equal(reference['Request_id'], ids):
        raise ValueError('Парное сравнение требует одинакового порядка Request_id')
    if np.shape(reference['metrics']) != compact.shape:
        raise ValueError('Форма reference metrics должна быть (n_queries, 3, 4)')
    token = hashlib.sha256(f'{split}/{name}'.encode()).hexdigest()[:20]
    predictions_path = directory / f'{token}.npz'
    intervals_path = directory / f'{token}_paired_intervals.csv'
    with predictions_path.open('xb') as file:
        np.savez_compressed(file, Request_id=ids,
                            ranked_api_ids=np.asarray(rankings, dtype=np.int64),
                            metrics=compact, metric_names=np.array(['Recall', 'NDCG', 'FullCoverage']),
                            k=ks)
    intervals = []
    for metric_index, metric in enumerate(('Recall', 'NDCG', 'FullCoverage')):
        for k_index, k in enumerate(ks):
            differences = compact[:, metric_index, k_index] - reference['metrics'][:, metric_index, k_index]
            differences = differences[np.isfinite(differences)]
            count = len(differences)
            mean = float(differences.mean()) if count else np.nan
            se = float(differences.std(ddof=1) / np.sqrt(count)) if count > 1 else np.nan
            intervals.append(dict(model=name, reference=reference['model'], metric=metric,
                                  k=int(k), queries=count, difference=mean, standard_error=se,
                                  low=mean - 1.959963984540054 * se,
                                  high=mean + 1.959963984540054 * se,
                                  method='paired_query_mean_normal_95'))
    pd.DataFrame(intervals).to_csv(intervals_path, index=False, mode='x')
    record = dict(model=name, split=split, config={**model.config, 'n_neighbors': neighbors},
                  seconds=seconds, queries=len(ids), excluded_no_positive=int(np.isnan(compact[:, 0, 0]).sum()),
                  predictions=predictions_path.name, intervals=intervals_path.name,
                  reference=reference['model'], uncertainty='paired_query_mean_normal_95',
                  uncertainty_seed=68, resampling=False)
    with (directory / 'artifacts.jsonl').open('a', encoding='utf-8') as file:
        file.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')


def _family_rankings(model, rows, similarities, neighbors, known, known_factors):
    """Ранжировать без повторного копирования факторов известных API."""
    rankings = []
    for neighbor_rows, similarity in zip(rows[:, :neighbors], similarities[:, :neighbors]):
        weights = np.maximum(similarity, 0)
        if model.config['neighbor_weighting'] == 'uniform' or not weights.sum():
            weights = np.ones(len(neighbor_rows), dtype=np.float32)
        weights = weights / weights.sum()
        factor = weights @ model.left_factors_[model.query_rows_[neighbor_rows]]
        scores = np.zeros(len(model.candidate_ids_), dtype=np.float32)
        scores[known] = known_factors @ factor
        order = np.lexsort((model.candidate_ids_, -scores))[:20]
        rankings.append(model.candidate_ids_[order].tolist())
    return rankings


def evaluate_family_grid(grid, data, split='test', artifact_dir=None, reference=None):
    """Оценить всю замороженную сетку Family CF с общими соседями без fit."""
    if not grid or not data.queries.get(split):
        raise ValueError('Нужны обученная сетка и непустой split')
    first = next(iter(grid.values()))
    cache = family_neighbors(first, data, split=split)
    if reference is None:
        reference = {}
    curves = []
    for fit_id, model in grid.items():
        model.prepare_candidates(data.corpus)
        for neighbors in FAMILY_NEIGHBORS:
            curves.append(family_validation_summary(
                model, data, f'{fit_id} | n={neighbors}', neighbors, cache, split=split,
                artifact_dir=artifact_dir, reference=reference))
    return pd.concat(curves, ignore_index=True)


def select_family(data, encoder, evaluate_callback=None, top_n=3, seed=67,
                  artifact_dir=None, reference=None, *, device='cpu'):
    """Перебрать семейства на val и переобучить выбранные модели на train.

    evaluate_callback(models, data, split='val') возвращает длинную таблицу
    model, metric, k, value. Допускаются значения отдельных запросов.
    Все 240 кривых сохраняются вызывающим
    кодом. Test при выборе конфигураций не используется.
    """
    if not data.queries.get('val'):
        raise ValueError('Для выбора Family CF требуется непустой val')
    if evaluate_callback is not None and artifact_dir is not None:
        raise ValueError('Сохранение per-query артефактов требует встроенного оценщика')
    if reference is None:
        reference = {}
    curves, fit_records, candidate_specs = [], [], {}
    neighbor_cache = None
    frozen_grid = {}
    shared_embeddings = shared_query_ids = None
    infeasible = {'BPR требует хотя бы одну строку с позитивом и негативом',
                  'n_families превышает число train-запросов'}
    for fit_id, spec in family_specs(seed, device=device).items():
        try:
            models, resources = train_suite(data, encoder, {fit_id: spec})
        except ValueError as error:
            if str(error) not in infeasible:
                raise
            fit_records.append(dict(model=fit_id, status='infeasible', reason=str(error)))
            continue
        fit_records.append(dict(resources.iloc[0], status='ok'))
        model = models[fit_id]
        if shared_embeddings is None:
            shared_embeddings, shared_query_ids = model.query_embeddings_, model.query_ids_
        frozen = copy(model)
        frozen.config = dict(model.config)
        frozen.query_embeddings_, frozen.query_ids_ = shared_embeddings, shared_query_ids
        del frozen.targets_, frozen.row_embeddings_
        frozen_grid[fit_id] = frozen
        model.prepare_candidates(data.corpus)
        if evaluate_callback is None and neighbor_cache is None:
            neighbor_cache = family_neighbors(model, data)
        for neighbors in FAMILY_NEIGHBORS:
            name = f'{fit_id} | n={neighbors}'
            model.config['n_neighbors'] = neighbors
            if evaluate_callback is None:
                frame = family_validation_summary(model, data, name, neighbors, neighbor_cache,
                                                   artifact_dir=artifact_dir, reference=reference)
            else:
                frame = evaluate_callback({name: model}, data, split='val')
            curves.append(frame.groupby(['model', 'metric', 'k'], as_index=False).value.mean())
            candidate_specs[name] = ModelSpec(spec.factory, {
                **spec.config, 'n_neighbors': neighbors})
        del model, models
    if not curves:
        raise ValueError('Нет допустимых конфигураций Family CF')
    summary = pd.concat(curves, ignore_index=True)
    scores = summary[summary.k == 10].pivot(index='model', columns='metric', values='value')
    scores = scores.reset_index().sort_values(
        ['FullCoverage', 'Recall', 'NDCG', 'model'], ascending=[False, False, False, True])
    selected = {name: candidate_specs[name] for name in scores.head(top_n).model}
    models, final_resources = train_suite(data, encoder, selected)
    return dict(models=models, specs=selected, scores=scores, curves=summary,
                fit_resources=pd.DataFrame(fit_records), final_resources=final_resources,
                frozen_grid=frozen_grid)
