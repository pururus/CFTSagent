"""Семантический ранжировщик на замороженном локальном энкодере."""
import json
import math
from pathlib import Path
from .base_ranker import BaseRanker


class ZeroShotBGERanker(BaseRanker):
    """Cosine-поиск с локальным BGE или внедрённым энкодером."""
    def __init__(self, model_path=None, revision=None, query_instruction='', max_length=512, batch_size=32, encoder=None, device='cuda'):
        if type(batch_size) is not int or batch_size < 1 or type(max_length) is not int or max_length < 1:
            raise ValueError('Недопустимый размер')
        self.model_path, self.revision = str(model_path) if model_path else None, revision
        self.query_instruction, self.max_length, self.batch_size = query_instruction, max_length, batch_size
        self.device = device
        self.encoder, self.vectors = encoder, None

    def _encoder(self):
        if self.encoder is None:
            if not self.model_path or not Path(self.model_path).is_dir():
                raise FileNotFoundError('Локальные веса не найдены')
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.model_path, local_files_only=True, revision=self.revision, device=self.device)
            self.encoder.max_seq_length = self.max_length
        return self.encoder

    @staticmethod
    def _normalize(vector):
        vector = [float(x) for x in vector]
        if not vector or not all(math.isfinite(x) for x in vector):
            raise ValueError('Неверный вектор')
        norm = math.sqrt(sum(x*x for x in vector))
        return [x / norm for x in vector] if norm else vector

    def fit(self, corpus, train_qrels):
        self.validate_input(corpus, train_qrels)
        if any(doc not in corpus for docs in train_qrels.values() for doc in docs):
            raise ValueError('Неизвестный документ')
        ids = list(corpus)
        vectors = {}
        for offset in range(0, len(ids), self.batch_size):
            batch = ids[offset:offset+self.batch_size]
            texts = [corpus[k]['description'] for k in batch]
            encoded = list(self._encoder().encode(texts))
            if len(encoded) != len(batch):
                raise ValueError('Неверное число векторов')
            vectors.update({key: self._normalize(vec) for key, vec in zip(batch, encoded)})
        if len({len(v) for v in vectors.values()}) > 1:
            raise ValueError('Несогласованные размерности')
        self.vectors = vectors
        self.candidates = vectors

    def prepare_candidates(self, corpus):
        if self.vectors is None:
            raise RuntimeError('Модель не подготовлена')
        self.validate_input(corpus)
        ids = list(corpus)
        vectors = {}
        for offset in range(0, len(ids), self.batch_size):
            batch = ids[offset:offset + self.batch_size]
            texts = [corpus[k]['description'] for k in batch]
            encoded = list(self._encoder().encode(texts))
            if len(encoded) != len(batch):
                raise ValueError('Неверное число векторов')
            vectors.update({key: self._normalize(vec) for key, vec in zip(batch, encoded)})
        dimensions = {len(v) for v in vectors.values()}
        if len(dimensions) > 1 or (vectors and self.vectors and dimensions != {len(next(iter(self.vectors.values())))}):
            raise ValueError('Несогласованные размерности')
        self.candidates = vectors

    def predict(self, query, top_k):
        if self.vectors is None:
            raise RuntimeError('Модель не подготовлена')
        if type(top_k) is not int or top_k < 0:
            raise ValueError('Недопустимый top_k')
        scores = dict.fromkeys(self.candidates, 0.0)
        if query and scores:
            encoded = list(self._encoder().encode([self.query_instruction + query]))
            if len(encoded) != 1:
                raise ValueError('Неверное число векторов')
            q = self._normalize(encoded[0])
            if len(q) != len(next(iter(self.candidates.values()))):
                raise ValueError('Несогласованные размерности')
            scores = {key: sum(a*b for a,b in zip(q, vec)) for key, vec in self.candidates.items()}
        return sorted(scores, key=lambda doc: (-scores[doc], doc))[:top_k]

    def save(self, path):
        if self.vectors is None:
            raise RuntimeError('Модель не подготовлена')
        path = Path(path)
        path.mkdir()
        data = {'version': 2, 'type': 'bge', 'model_path': self.model_path, 'revision': self.revision, 'query_instruction': self.query_instruction, 'max_length': self.max_length, 'batch_size': self.batch_size, 'device': self.device, 'vectors': list(self.vectors.items())}
        (path / 'model.json').write_text(json.dumps(data, allow_nan=False), encoding='utf-8')

    def load(self, path):
        try:
            data = json.loads((Path(path) / 'model.json').read_text(encoding='utf-8'))
            if data['version'] != 2 or data['type'] != 'bge' or not isinstance(data['vectors'], list):
                raise ValueError('Неверное состояние')
            candidate = ZeroShotBGERanker(data['model_path'], data['revision'], data['query_instruction'], data['max_length'], data['batch_size'], self.encoder, data.get('device', 'cuda'))
            if candidate.encoder is None and (not candidate.model_path or not Path(candidate.model_path).is_dir()):
                raise FileNotFoundError('Локальные веса не найдены')
            vectors = {key: self._normalize(vec) for key, vec in data['vectors'] if type(key) is int and key >= 0}
            if len(vectors) != len(data['vectors']) or len({len(v) for v in vectors.values()}) > 1:
                raise ValueError('Несогласованные векторы')
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError('Неверное состояние') from exc
        self.__dict__.update(candidate.__dict__)
        self.vectors = vectors
        self.candidates = vectors
