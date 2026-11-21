"""Чтение подготовленных таблиц, сопоставление каталогов и кэш энкодера."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from io import BytesIO
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    corpus: dict[int, dict]
    queries: dict[str, dict[int, str]]
    qrels: dict[str, dict[int, dict[int, float]]]


def load_dataset(directory, splits=("train", "val", "test"), *, git_ref=None, repo_root=None):
    """Загрузить готовые таблицы и сохранить назначенные запросам сплиты.

    Для переноса передайте splits=("test",). Позитивные связи остальных
    сплитов при этом в возвращаемый объект не попадают.
    """
    directory = Path(directory)

    def read_table(name, **kwargs):
        path = directory / (name + ".parquet")
        if git_ref is None:
            return pd.read_parquet(path, **kwargs)
        root = Path(repo_root or Path.cwd()).resolve()
        absolute = path if path.is_absolute() else root / path
        relative = absolute.resolve().relative_to(root).as_posix()
        content = subprocess.check_output(["git", "show", f"{git_ref}:{relative}"], cwd=root)
        return pd.read_parquet(BytesIO(content), **kwargs)

    apis = read_table("apis")
    requests = read_table("requests")
    requests = requests[requests["split"].isin(splits)]
    request_ids = [int(identifier) for identifier in requests.Request_id]
    interactions = read_table(
        "interactions",
        filters=[("Request_id", "in", request_ids)],
    )
    corpus = {
        int(row.API_id): {"description": row.description}
        for row in apis.itertuples(index=False)
    }
    queries = {
        split: dict(zip(frame.Request_id.map(int), frame.text))
        for split, frame in requests.groupby("split", sort=False)
    }
    positives = {identifier: {} for identifier in request_ids}
    for row in interactions.itertuples(index=False):
        if row.relevance > 0:
            positives[int(row.Request_id)][int(row.API_id)] = float(row.relevance)
    qrels = {
        split: {identifier: positives[identifier] for identifier in partition}
        for split, partition in queries.items()
    }
    return Dataset(corpus, queries, qrels)


def _api_identity(document):
    """Извлечь квалифицированное имя API из JSON-документации."""
    try:
        metadata = json.loads(document["description"])
    except (ValueError, TypeError):
        return None
    if not isinstance(metadata, dict):
        return None
    tool = metadata.get("tool_name")
    api = metadata.get("api_name")
    if isinstance(tool, str) and isinstance(api, str) and tool.strip() and api.strip():
        return tool.strip(), api.strip()
    return None


def _unique_index(corpus, key_function):
    groups = defaultdict(list)
    for identifier, document in corpus.items():
        key = key_function(document)
        if key:
            groups[key].append(identifier)
    return {key: identifiers[0] for key, identifiers in groups.items() if len(identifiers) == 1}


def align_target(source, target):
    """Перенести целевые API в пространство ID исходного каталога.

    Совпадение принимается при однозначном квалифицированном имени или
    точном описании. Конфликтующие свидетельства и повторные соответствия
    получают новые ID. Исходные target_id сохраняются для экспорта.
    """
    identities = _unique_index(source.corpus, _api_identity)
    descriptions = _unique_index(source.corpus, lambda doc: doc["description"])
    proposals = {}
    for identifier, document in target.corpus.items():
        by_identity = identities.get(_api_identity(document))
        by_description = descriptions.get(document["description"])
        candidates = {value for value in (by_identity, by_description) if value is not None}
        if len(candidates) == 1:
            proposals[identifier] = (candidates.pop(), "identity" if by_identity is not None else "description")
    counts = Counter(value[0] for value in proposals.values())
    next_id = max(source.corpus, default=-1) + 1
    mapping = {}
    rows = []
    for identifier in sorted(target.corpus):
        proposal = proposals.get(identifier)
        if proposal is not None and counts[proposal[0]] == 1:
            aligned, method = proposal
        else:
            aligned, method = next_id, "unmatched"
            next_id += 1
        mapping[identifier] = aligned
        rows.append(dict(target_id=identifier, aligned_id=aligned, method=method))
    corpus = {mapping[key]: document for key, document in target.corpus.items()}
    qrels = {
        split: {
            identifier: {mapping[api]: relevance for api, relevance in links.items()}
            for identifier, links in partition.items()
        }
        for split, partition in target.qrels.items()
    }
    return Dataset(corpus, target.queries, qrels), pd.DataFrame(
        rows, columns=["target_id", "aligned_id", "method"]
    )


class CachedEncoder:
    """Кэш по полному тексту для одного фиксированного энкодера.

    Префикс запроса добавляет вызывающий ранжировщик. Различные инструкции
    формируют различные ключи. Возвращаемые массивы можно изменять безопасно.
    """

    def __init__(self, encoder, batch_size=32):
        self.encoder = encoder
        self.batch_size = batch_size
        self.cache = {}
        self.dimension = 0

    @classmethod
    def from_local(cls, path, device="cpu", batch_size=32, max_length=512):
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(str(path), device=device, local_files_only=True)
        encoder.max_seq_length = max_length
        return cls(encoder, batch_size=batch_size)

    def encode(self, texts):
        texts = list(texts)
        missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            vectors = np.asarray(self.encoder.encode(batch), dtype=np.float32)
            self.dimension = vectors.shape[1]
            self.cache.update(zip(batch, vectors.copy()))
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.stack([self.cache[text] for text in texts])
