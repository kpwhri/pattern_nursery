import csv
import json
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pattern_nursery.cli.main import app

runner = CliRunner()


def create_csv_corpus(path: Path):
    docs = [
        {'id': '1', 'text': 'Väinämöinen, old and steadfast.'},
        {'id': '2', 'text': 'Ilmarinen, smith of the sky.'},
        {'id': '3', 'text': 'Lemminkäinen, the reckless hero.'},
        {'id': '4', 'text': 'Louhi, mistress of the North.'}
    ]
    with open(path, 'w', encoding='utf8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['id', 'text'])
        writer.writeheader()
        writer.writerows(docs)
    return docs


def create_sqlite_corpus(path: Path, table_name='documents'):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute(f'CREATE TABLE {table_name} (id TEXT PRIMARY KEY, text TEXT, shard_id INTEGER)')
    docs = [
        ('1', 'Väinämöinen, old and steadfast.', 0),
        ('2', 'Ilmarinen, smith of the sky.', 1),
        ('3', 'Lemminkäinen, the reckless hero.', 0),
        ('4', 'Louhi, mistress of the North.', 1)
    ]
    cursor.executemany(f'INSERT INTO {table_name} VALUES (?, ?, ?)', docs)
    conn.commit()
    conn.close()


@pytest.fixture
def csv_corpus(tmp_path):
    corpus_path = tmp_path / 'corpus.csv'
    with open(corpus_path, 'w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'text', 'extra'])
        writer.writeheader()
        writer.writerow({'id': '1', 'text': 'Väinämöinen was an old and wise wizard.', 'extra': 'first'})
        writer.writerow({'id': '2', 'text': 'Ilmarinen was a great blacksmith.', 'extra': 'second'})
        writer.writerow({'id': '3', 'text': 'Lemminkäinen was a handsome young man.', 'extra': 'third'})
        writer.writerow({'id': '4', 'text': 'Louhi was the mistress of Pohjola.', 'extra': 'fourth'})
    return corpus_path


def test_csv_corpus_indexing_and_sharding(csv_corpus, tmp_path):
    work_dir = tmp_path / 'work'
    work_dir.mkdir()

    # 1. Index corpus
    result = runner.invoke(app, [
        'index-corpus',
        '--input', str(csv_corpus),
        '--work-dir', str(work_dir),
        '--total-shards', '2'
    ])
    assert result.exit_code == 0

    # Check if index DB exists
    index_db_path = work_dir / 'corpus_index' / 'corpus.id.sqlite3'
    assert index_db_path.exists()

    # 2. Init index
    result = runner.invoke(app, [
        'init-index-shard',
        '--input', str(csv_corpus),
        '--work-dir', str(work_dir),
        '--run-name', 'test_run',
        '--shard-id', '0',
        '--total-shards', '2',
        '--collection', 'notes',
        '--backend', 'mock',
        '--model-path', 'mock-model'
    ])
    assert result.exit_code == 0

    shard0_manifest = work_dir / 'runs' / 'test_run' / 'vectorstore' / 'notes' / 'shards' / 'shard_00000' / 'manifest.json'
    assert shard0_manifest.exists()
    with open(shard0_manifest, 'r') as f:
        manifest = json.load(f)
        # Should have 2 docs in shard 0 if split evenly
        assert manifest['counts']['docs_seen'] == 2
        assert manifest['corpus']['kind'] == 'csv'

    # 3. Test retrieving from CSV
    # We need to run init-index first to set up the run
    result = runner.invoke(app, [
        'init-index',
        '--input', str(csv_corpus),
        '--work-dir', str(work_dir),
        '--run-name', 'test_run',
        '--collection', 'notes',
        '--backend', 'mock',
        '--model-path', 'mock-model'
    ])
    assert result.exit_code == 0

    # Now try to retrieve. retrieve_notes uses reconstruct_text_if_needed which uses CorpusIndexSqlite
    result = runner.invoke(app, [
        'retrieve-notes',
        '--seeds', 'Väinämöinen',
        '--work-dir', str(work_dir),
        '--run-name', 'test_run',
        '--top-k', '1'
    ])
    assert result.exit_code == 0

    # Check results
    results_path = work_dir / 'runs' / 'test_run' / 'artifacts' / 'results_notes.jsonl'
    assert results_path.exists()
    with open(results_path, 'r', encoding='utf8') as f:
        line = f.readline()
        res = json.loads(line)
        assert isinstance(res.get('text'), str) and len(res['text']) > 0
        assert res.get('doc_id') in {'1', '2', '3', '4'}


def test_csv_corpus_retrieval_integration(tmp_path):
    work_dir = tmp_path / 'work'
    csv_path = work_dir / 'corpus.csv'
    db_path = work_dir / 'corpus.db'
    work_dir.mkdir(parents=True, exist_ok=True)
    create_csv_corpus(csv_path)
    create_sqlite_corpus(db_path)

    # 1. init-index with CSV
    # We must provide some model_path because it's required.
    result = runner.invoke(app, [
        'init-index',
        '--input', str(csv_path),
        '--work-dir', str(work_dir),
        '--run-name', 'csv_test_run',
        '--collection', 'notes',
        '--backend', 'mock',
        '--model-path', 'dummy',
    ])
    assert result.exit_code == 0

    # 2. Manually update config.toml of the collection to include sqlite_path
    store_path = work_dir / 'runs' / 'csv_test_run' / 'vectorstore' / 'notes'
    config_path = store_path / 'config.toml'

    with open(config_path, 'a', encoding='utf8') as fh:
        # We use a relative path to work_dir as expected by _get_corpus_backend_from_config
        rel_db_path = db_path.relative_to(work_dir)
        fh.write(f'sqlite_path = "{str(rel_db_path).replace("\\", "\\\\")}"\n')

    # 3. Retrieve: _get_corpus_backend_from_config, which should now find 'sqlite_path', CsvCorpusBackend
    result = runner.invoke(app, [
        'retrieve-notes',
        '--work-dir', str(work_dir),
        '--run-name', 'csv_test_run',
        '--seeds', 'Väinämöinen',
        '--backend', 'mock',
        '--model-path', 'dummy'
    ])
    assert result.exit_code == 0

    # 4. Verify that text was resolved (which means CsvCorpusBackend.get_text worked via SQLite)
    results_path = work_dir / 'runs' / 'csv_test_run' / 'artifacts' / 'results_notes.jsonl'
    assert results_path.exists()
    with open(results_path, 'r', encoding='utf8') as fh:
        found_text = False
        for line in fh:
            res = json.loads(line)
            # CsvCorpusBackend.get_text('1') should return 'Väinämöinen, old and steadfast.'
            if 'text' in res and 'Väinämöinen' in res['text']:
                found_text = True
                break
        assert found_text
