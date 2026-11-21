# ToolBench G1

G1 содержит запросы, каталог API и разметку релевантности. Если нужны готовые данные для эксперимента, таблицы уже лежат в [processed/G1/](processed/G1/). Для воспроизведения обработки скачайте пять исходных файлов из [Adorg/ToolBench, каталог retrieval/G1](https://huggingface.co/datasets/Adorg/ToolBench/tree/0c25df5e6937aae854c2cc1bb203297600afce8b/retrieval/G1).

## Исходные файлы

Все пять файлов используют табуляцию как разделитель. Заголовок есть только у `corpus.tsv`.

| Файл | Что хранит |
|---|---|
| `corpus.tsv` | Каталог API. Поле `docid` содержит ID API, а `document_content` — строку JSON с документацией, включая имя API, описание и параметры. |
| `train.query.txt` | ID и текст запроса из исходной обучающей выборки. Один запрос может встречаться в нескольких строках. |
| `test.query.txt` | ID и текст запроса из исходной тестовой выборки в том же формате. |
| `qrels.train.tsv` | Связи обучающих запросов с релевантными API. Четыре столбца по порядку — ID запроса, служебное поле, ID API и метка релевантности. |
| `qrels.test.tsv` | Связи тестовых запросов с API в том же формате. В обоих файлах разметки все метки релевантности равны 1. |

Например, строка `87388  0  4310  1` из `qrels.test.tsv` связывает запрос `87388` из `test.query.txt` с API `4310` из `corpus.tsv`. У одного запроса может быть несколько релевантных API.

## Как скачать

Выполните из корня проекта. Команды загружают файлы из той же версии источника, которая использовалась для подготовки Parquet.

```bash
(
  set -e
  cd data/ToolBench
  mkdir -p raw/G1
  source_url="https://huggingface.co/datasets/Adorg/ToolBench/resolve/0c25df5e6937aae854c2cc1bb203297600afce8b/retrieval/G1"

  for name in corpus.tsv train.query.txt test.query.txt qrels.train.tsv qrels.test.tsv; do
    curl --fail --location --retry 3 "$source_url/$name" -o "raw/G1/$name.part"
    mv "raw/G1/$name.part" "raw/G1/$name"
  done
)
```

## Как получить Parquet

Откройте [process_data.ipynb](process_data.ipynb) с рабочей директорией `data/ToolBench/`. В окружении нужны `pandas`, `scikit-learn` и `pyarrow`. Для обработки только G1 выполните ячейки с импортами и определениями функций, затем ячейку `process_data("G1")`.

| Файл в `processed/G1/` | Столбцы | Строк |
|---|---|---:|
| `requests.parquet` | `Request_id`, `text`, `split` | 87 419 |
| `apis.parquet` | `API_id`, `description` | 10 439 |
| `interactions.parquet` | `Request_id`, `API_id`, `relevance` | 424 169 |

В `id_maps/requests.parquet` и `id_maps/apis.parquet` хранится соответствие между исходными ID и числовыми ID в таблицах проекта.

**Разбиение.** Ноутбук объединяет исходные train и test, убирает точные повторы запросов, API и пар запрос–API, затем делит запросы заново с `SEED = 67`. Получается 69 935 запросов в train, 8 742 в val и 8 742 в test. Все связи одного запроса относятся к его выборке. Для оценки на исходном тесте ToolBench используйте `test.query.txt` и `qrels.test.tsv`.

**Проверки.** После удаления повторов ID запросов и API уникальны. Тексты запросов также уникальны после удаления пробелов по краям и приведения к нижнему регистру. Каждый ID из разметки найден в соответствующей таблице. Поле `metadata` отсутствует у запросов и API, поле `title` отсутствует у API. У 789 API пустое поле `api_description`. Эти API сохранены, а итоговый столбец `description` содержит полный JSON из `document_content`.
