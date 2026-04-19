import json
import sqlite3
from pathlib import Path

from typer.testing import CliRunner

from pattern_nursery.cli.main import app

runner = CliRunner()


def create_sqlite_corpus(path: Path, table_name='documents'):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute(f"CREATE TABLE {table_name} (id TEXT PRIMARY KEY, text TEXT)")
    docs = [
        ('1', 'Väinämöinen, old and steadfast.'),
        ('2', 'Ilmarinen, smith of the sky.'),
        ('3', 'Lemminkäinen, the reckless hero.'),
        ('4', 'Louhi, mistress of the North.')
    ]
    cursor.executemany(f"INSERT INTO {table_name} VALUES (?, ?)", docs)
    conn.commit()
    conn.close()


def test_sqlite_corpus_validation_and_sharding(tmp_path):
    corpus_path = tmp_path / 'corpus.db'
    work_dir = tmp_path / 'work'
    create_sqlite_corpus(corpus_path)

    # 1. Run index-corpus with no total_shards
    result = runner.invoke(app, [
        'index-corpus',
        '--input', str(corpus_path),
        '--work-dir', str(work_dir)
    ])
    assert result.exit_code == 0
    # assert 'SQLite corpus validated; shard_id not requested' in result.output

    # check shard_id column absent
    conn = sqlite3.connect(corpus_path)
    cursor = conn.cursor()
    cursor.execute('PRAGMA table_info(documents)')
    cols = [row[1] for row in cursor.fetchall()]
    assert 'shard_id' not in cols
    conn.close()

    # 2. Run index-corpus --total-shards 4
    result = runner.invoke(app, [
        'index-corpus',
        '--input', str(corpus_path),
        '--work-dir', str(work_dir),
        '--total-shards', '4'
    ])
    assert result.exit_code == 0

    conn = sqlite3.connect(corpus_path)
    cursor = conn.cursor()
    cursor.execute('PRAGMA table_info(documents)')
    cols = [row[1] for row in cursor.fetchall()]
    assert 'shard_id' in cols

    cursor.execute("SELECT shard_id FROM documents")
    shards = [row[0] for row in cursor.fetchall()]
    assert all(s is not None for s in shards)
    assert all(0 <= s < 4 for s in shards)
    conn.close()

    # 3. Run index-corpus --total-shards 2 (mismatch)
    result = runner.invoke(app, [
        "index-corpus",
        "--input", str(corpus_path),
        "--work-dir", str(work_dir),
        "--total-shards", "2"
    ])
    assert result.exit_code != 0
    # assert "Total shards mismatch" in result.output

    # 4. Run with --force-rebuild-shards
    result = runner.invoke(app, [
        "index-corpus",
        "--input", str(corpus_path),
        "--work-dir", str(work_dir),
        "--total-shards", "2",
        "--force-rebuild-shards"
    ])
    assert result.exit_code == 0

    conn = sqlite3.connect(corpus_path)
    cursor = conn.cursor()
    cursor.execute("SELECT shard_id FROM documents")
    shards = [row[0] for row in cursor.fetchall()]
    assert all(0 <= s < 2 for s in shards)
    conn.close()


def test_sqlite_sharded_indexing_and_retrieval(tmp_path):
    corpus_path = tmp_path / 'corpus.db'
    work_dir = tmp_path / 'work'
    create_sqlite_corpus(corpus_path)

    # 1. Shard corpus
    runner.invoke(app, [
        'index-corpus',
        '--input', str(corpus_path),
        '--work-dir', str(work_dir),
        '--total-shards', '2'
    ])

    # 2. Init shards
    for sid in [0, 1]:
        result = runner.invoke(app, [
            'init-index-shard',
            '--input', str(corpus_path),
            '--work-dir', str(work_dir),
            '--run-name', 'test_run',
            '--shard-id', str(sid),
            '--total-shards', '2',
            '--collection', 'notes',
            '--collection', 'sentences',
            '--backend', 'mock',
            '--model-path', 'dummy'
        ])
        assert result.exit_code == 0

    # Manually create run_config.json
    run_dir = work_dir / 'runs' / 'test_run'
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'run_config.json', 'w', encoding='utf8') as f:
        json.dump({
            'input': str(corpus_path.absolute()),
            'id_col': 'id',
            'text_col': 'text',
            'model_path': 'dummy',
            'backend': 'mock',
            'table_name': 'documents'
        }, f)

    # 3. Retrieve
    result = runner.invoke(app, [
        "retrieve-notes",
        "--work-dir", str(work_dir),
        "--run-name", "test_run",
        "--seeds", "Väinämöinen",
        "--backend", "mock",
        "--model-path", "dummy"
    ])
    assert result.exit_code == 0

    # Assert output JSONL includes text
    results_path = work_dir / "runs" / "test_run" / "artifacts" / "results_notes.jsonl"
    assert results_path.exists()
    with open(results_path, 'r', encoding='utf8') as f:
        found_text = False
        for line in f:
            res = json.loads(line)
            if 'text' in res and len(res['text']) > 0:
                found_text = True
                break
        assert found_text
