"""Общий контракт ранжировщиков API без привязки к формату датасета."""

from abc import ABC, abstractmethod
from os import PathLike
from typing import Any, Mapping, Union


Corpus = Mapping[int, Mapping[str, Any]]
Qrels = Mapping[int, Mapping[int, float]]
ModelPath = Union[str, PathLike[str]]


class BaseRanker(ABC):
    """Принимает нормализованные данные в памяти; файлы читают адаптеры.

    Corpus сопоставляет целочисленный ID API с полем description (строка).
    Qrels сопоставляет целочисленный ID
    запроса с отображением ID документов в числовую релевантность.
    Позитивная связь имеет релевантность > 0; отсутствующая неизвестна.
    """

    @staticmethod
    def validate_input(corpus, qrels=None):
        """Проверить целые ID, текст и ссылки train-взаимодействий."""
        for api_id, doc in corpus.items():
            if type(api_id) is not int or api_id < 0 or not isinstance(doc, Mapping) or not isinstance(doc.get('description'), str):
                raise ValueError('Ожидается неотрицательный целый API_id и строка description')
        if qrels is not None:
            for request_id, docs in qrels.items():
                if type(request_id) is not int or request_id < 0 or not isinstance(docs, Mapping):
                    raise ValueError('Недопустимый Request_id или qrels')
                for api_id, relevance in docs.items():
                    if type(api_id) is not int or api_id not in corpus or isinstance(relevance, bool) or not isinstance(relevance, (int, float)) or not __import__('math').isfinite(relevance):
                        raise ValueError('Недопустимая связь qrels')

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """Задать параметры без чтения датасетов и загрузки весов из сети."""
        raise NotImplementedError

    @abstractmethod
    def fit(self, corpus: Corpus, train_qrels: Qrels) -> None:
        """Подготовить модель только на train API и train-связях.

        Popularity считает связи; BM25 строит индекс; zero-shot BGE
        кодирует документы без обучения весов. BM25 и BGE могут
        игнорировать train_qrels. Пустой корпус допустим. Неизвестный
        ID документа в qrels вызывает ValueError. Входы не изменяются.
        Тексты train-запросов будущему KNN передаются отдельным контрактом.
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
