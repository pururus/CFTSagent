"""Общий контракт ранжировщиков API без привязки к формату датасета."""

from abc import ABC, abstractmethod
import math
from os import PathLike
from typing import Any, Mapping, Protocol, Union

from numpy.typing import ArrayLike


Corpus = Mapping[int, Mapping[str, Any]]
Qrels = Mapping[int, Mapping[int, float]]
Queries = Mapping[int, str]
ModelPath = Union[str, PathLike[str]]


class Encoder(Protocol):
    """Минимальный интерфейс локального или внедрённого энкодера."""

    def encode(self, texts: list[str], /) -> ArrayLike:
        ...


class BaseRanker(ABC):
    """Принимает нормализованные данные в памяти; файлы читают адаптеры.

    Corpus сопоставляет целочисленный ID API с полем description (строка).
    Qrels сопоставляет целочисленный ID
    запроса с отображением ID документов в числовую релевантность.
    Позитивная связь имеет релевантность > 0; отсутствующая неизвестна.
    """

    @staticmethod
    def validate_input(corpus: Corpus, qrels: Qrels | None = None) -> None:
        """Проверить целые ID, текст и ссылки train-взаимодействий."""
        for api_id, doc in corpus.items():
            if type(api_id) is not int or not 0 <= api_id < 2**63 or not isinstance(doc, Mapping) or not isinstance(doc.get('description'), str):
                raise ValueError('Ожидается неотрицательный целый API_id и строка description')
        if qrels is not None:
            for request_id, docs in qrels.items():
                if type(request_id) is not int or not 0 <= request_id < 2**63 or not isinstance(docs, Mapping):
                    raise ValueError('Недопустимый Request_id или qrels')
                for api_id, relevance in docs.items():
                    if type(api_id) is not int or api_id not in corpus or isinstance(relevance, bool) or not isinstance(relevance, (int, float)) or not math.isfinite(relevance):
                        raise ValueError('Недопустимая связь qrels')

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """Задать параметры без чтения датасетов и загрузки весов из сети."""
        raise NotImplementedError

    @abstractmethod
    def fit(self, corpus: Corpus, train_qrels: Qrels, *, train_queries: Queries | None = None) -> None:
        """Подготовить модель только на train API и train-связях.

        Popularity считает связи; BM25 строит индекс; zero-shot BGE
        кодирует документы без обучения весов. BM25 и BGE могут
        игнорировать train_qrels. Пустой корпус допустим. Неизвестный
        ID документа в qrels вызывает ValueError. Входы не изменяются.
        CF использует train_queries с текстами только тренировочных запросов.
        Бейзлайны могут игнорировать train_queries.
        """
        raise NotImplementedError

    @abstractmethod
    def prepare_candidates(self, corpus: Corpus) -> None:
        """Принять каталог кандидатов при инференсе без изменения train-статистик.

        Вызывать после fit или load. Метки и запросы здесь не принимаются.
        При повторном вызове каталог кандидатов полностью заменяется.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, query: str, top_k: int) -> list[int]:
        """Вернуть уникальные ID по убыванию оценки на всём корпусе.

        При равных оценках ID сортируются по возрастанию.
        top_k должен быть целым числом >= 0 (bool недопустим), иначе
        ValueError. Возвращается min(top_k, размер корпуса) ID.
        До fit/load вызывается RuntimeError. Пустой запрос допустим.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: ModelPath) -> None:
        """Сохранить подготовленное состояние в новый каталог path.

        Сохранить параметры, версию формата, порядок ID и артефакты,
        необходимые для восстановления. До fit/load — RuntimeError;
        существующий path — FileExistsError, без перезаписи или удаления.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: ModelPath) -> None:
        """Восстановить состояние экземпляра из ранее сохранённого каталога.

        Проверить формат и согласованность ID/артефактов до замены
        состояния. Ошибочный формат — ValueError; отсутствие пути —
        FileNotFoundError. Автоматическое скачивание весов недопустимо.
        После успешной загрузки predict работает без повторного fit.
        """
        raise NotImplementedError
