"""Организация воспроизводимого запуска и экспорт результатов каждого этапа."""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
from uuid import uuid4

import pandas as pd

from experiments.dataset_io import align_target, load_dataset
from experiments.evaluation import evaluate_suite
from experiments.model_suite import QUERY_INSTRUCTION, evaluate_family_grid
from experiments.reporting import export_results


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def model_configs(specs):
    return {name: {'config': spec.config, 'factory': spec.factory.__name__}
            for name, spec in specs.items()}


def file_digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def start_run(root, weights, fixed, family, seed, bootstrap_seed):
    run = root / 'output' / 'toollens_transfer' / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8])
    run.mkdir(parents=True)
    sources = [*sorted((root / 'rankers').glob('*.py')),
               *sorted((root / 'experiments').glob('*.py')),
               root / 'experiments' / 'toollens_toolbench.ipynb']
    write_json(run / 'protocol.json', dict(
        seed=seed, bootstrap_seed=bootstrap_seed, bootstrap_repetitions=1000, cpu_threads=4,
        python=platform.python_version(), platform=platform.platform(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        packages={name: version(name) for name in ('numpy','pandas','pyarrow','scipy','scikit-learn',
                  'torch','sentence-transformers','transformers','matplotlib','jinja2')},
        source_hashes={str(p.relative_to(root)): file_digest(p) for p in sources},
        encoder_files={str(p.relative_to(weights)): file_digest(p)
                       for p in sorted(weights.rglob('*')) if p.is_file()},
        fixed=model_configs(fixed), family=model_configs(family),
        query_instruction=QUERY_INSTRUCTION, neighbors=[1,2,3,5,10],
        family_selection=['FullCoverage@10', 'Recall@10', 'NDCG@10', 'name']))
    return run


def dataset_record(data, directory):
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([(split, key) for split, part in data.queries.items() for key in sorted(part)],
                 columns=['split', 'Request_id']).to_csv(directory / 'split_ids.csv', index=False)
    pd.Series(sorted(data.corpus), name='API_id').to_csv(directory / 'candidate_ids.csv', index=False)


def store_selection(result, directory):
    directory.mkdir(parents=True, exist_ok=True)
    result['scores'].to_csv(directory / 'validation_ranking.csv', index=False)
    result['fit_resources'].to_csv(directory / 'grid_fits.csv', index=False)
    result['final_resources'].to_csv(directory / 'selected_fits.csv', index=False)
    write_json(directory / 'selected_configs.json', model_configs(result['specs']))
    return export_results(result['curves'], directory, family_grid=True, context='ToolLens validation')


def test_catalogue(models, family_grid, data, directory, history, encoder, bootstrap_seed, *, context):
    """Оценка замороженных моделей и всей сетки Family в одном test-каталоге."""
    dataset_record(data, directory)
    encoder.encode([QUERY_INSTRUCTION + text for text in data.queries['test'].values()])
    curves = evaluate_suite(models, data, 'test', directory / 'comparison', history, bootstrap_seed)
    table = export_results(curves, directory / 'comparison', context=context)
    family_curves = evaluate_family_grid(family_grid, data, split='test',
                                         artifact_dir=directory / 'family' / 'per_query')
    export_results(family_curves, directory / 'family', family_grid=True, context=context)
    return table


def transfer_group(root, target_root, group, source, models, family_grid, history,
                   encoder, run, bootstrap_seed, fallback_ref='main'):
    """Загрузить только test-связи ToolBench, сопоставить каталог и оценить перенос."""
    path = target_root / group
    directory = run / 'ToolBench' / group
    directory.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        target = load_dataset(path, splits=('test',))
        provenance = dict(directory=str(path), hashes={name: file_digest(path / name)
                          for name in ('apis.parquet','requests.parquet','interactions.parquet')})
    else:
        ref = subprocess.check_output(['git', 'rev-parse', '--verify', fallback_ref + '^{commit}'],
                                      text=True, cwd=root).strip()
        relative = Path('data/ToolBench/processed') / group
        target = load_dataset(relative, splits=('test',), git_ref=ref, repo_root=root)
        provenance = dict(git_commit=ref, directory=str(relative))
    write_json(directory / 'data_source.json', provenance)
    target, mapping = align_target(source, target)
    mapping.to_csv(directory / 'api_mapping.csv', index=False)
    return test_catalogue(models, family_grid, target, directory, history, encoder, bootstrap_seed,
                          context=f'ToolBench {group} test')
