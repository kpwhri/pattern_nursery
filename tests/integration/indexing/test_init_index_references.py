import json
from pattern_nursery.cli.main import app
from pattern_nursery.io.corpus_index import CorpusIndexSqlite
from typer.testing import CliRunner

runner = CliRunner()


def test_init_index_stores_references(tmp_path):
    corpus_path = tmp_path / 'corpus.jsonl'
    docs = [
        {'id': '1', 'text': 'Väinämöinen soittaa kannelta.'},
        {'id': '2', 'text': 'Ilmarinen takoo sammon.'}
    ]
    with open(corpus_path, 'w', encoding='utf8') as f:
        for d in docs:
            f.write(json.dumps(d) + '\n')

    work_dir = tmp_path / 'work'

    # 1. Pre-index corpus
    result = runner.invoke(app, [
        'index-corpus',
        '--input', str(corpus_path),
        '--work-dir', str(work_dir)
    ])
    assert result.exit_code == 0

    # 2. Run init-index (non-sharded)
    result = runner.invoke(app, [
        'init-index',
        '--input', str(corpus_path),
        '--work-dir', str(work_dir),
        '--run-name', 'ref_test',
        '--collection', 'notes',
        '--collection', 'sentences',
        '--backend', 'mock',
        '--model-path', 'dummy'
    ])
    assert result.exit_code == 0

    # 3. Check metadata for notes
    notes_meta_path = work_dir / 'runs' / 'ref_test' / 'vectorstore' / 'notes' / 'metadata.jsonl'
    assert notes_meta_path.exists()
    with open(notes_meta_path, 'r', encoding='utf8') as f:
        meta = [json.loads(line) for line in f]

    assert len(meta) == 2
    for m in meta:
        assert 'doc_id' in m
        assert 'doc_byte_offset' in m
        assert 'text' not in m  # Critical: text must be omitted

    # 4. Check metadata for sentences
    sent_meta_path = work_dir / 'runs' / 'ref_test' / 'vectorstore' / 'sentences' / 'metadata.jsonl'
    assert sent_meta_path.exists()
    with open(sent_meta_path, 'r', encoding='utf8') as f:
        meta = [json.loads(line) for line in f]

    assert len(meta) == 2  # 1 sentence per doc in this case
    for m in meta:
        assert 'doc_id' in m
        assert 'doc_byte_offset' in m
        assert 'sent_id' in m
        assert 'start_char' in m
        assert 'end_char' in m
        assert 'text' not in m  # Critical: text must be omitted

    # 5. Verify doc_byte_offset points to correct doc
    index = CorpusIndexSqlite(CorpusIndexSqlite.get_index_path(work_dir, corpus_path, 'id'))
    for m in meta:
        doc = index.get_doc_by_offset(corpus_path, m['doc_byte_offset'])
        assert doc['id'] == m['doc_id']


def test_init_index_csv(tmp_path):
    import csv
    corpus_path = tmp_path / 'corpus.csv'
    docs = [
        {'id': '1', 'text': 'Väinämöinen soittaa kannelta.'},
        {'id': '2', 'text': 'Ilmarinen takoo sammon.'}
    ]
    with open(corpus_path, 'w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'text'])
        writer.writeheader()
        writer.writerows(docs)

    work_dir = tmp_path / 'work'

    # run init-index
    from loguru import logger
    from io import StringIO
    log_capture = StringIO()
    handler_id = logger.add(log_capture, level='INFO')

    try:
        result = runner.invoke(app, [
            'init-index',
            '--input', str(corpus_path),
            '--work-dir', str(work_dir),
            '--run-name', 'csv_test',
            '--collection', 'notes',
            '--backend', 'mock',
            '--model-path', 'dummy'
        ], catch_exceptions=False)
        assert result.exit_code == 0

        log_output = log_capture.getvalue()
        assert 'Using CSV corpus backend' in log_output
    finally:
        logger.remove(handler_id)

    # Check metadata for notes
    notes_meta_path = work_dir / 'runs' / 'csv_test' / 'vectorstore' / 'notes' / 'metadata.jsonl'
    assert notes_meta_path.exists()
    with open(notes_meta_path, 'r', encoding='utf8') as f:
        meta = [json.loads(line) for line in f]

    assert len(meta) == 2
    for m in meta:
        assert 'doc_id' in m
        assert 'text' in m  # CSV should have full text by default since it has no offset index


def test_init_index_shards_csv(tmp_path):
    import csv
    import sqlite3
    corpus_path = tmp_path / 'corpus.csv'
    sqlite_path = tmp_path / 'corpus.db'
    docs = [
        {'id': '1', 'text': 'Väinämöinen soittaa kannelta.'},
        {'id': '2', 'text': 'Ilmarinen takoo sammon.'},
        {'id': '3', 'text': 'Lemminkäinen seikkailee.'},
        {'id': '4', 'text': 'Louhi hallitsee Pohjolaa.'}
    ]
    with open(corpus_path, 'w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'text'])
        writer.writeheader()
        writer.writerows(docs)

    # Pre-create sqlite corpus for CSV sharding
    conn = sqlite3.connect(sqlite_path)
    conn.execute('CREATE TABLE documents (id TEXT PRIMARY KEY, text TEXT)')
    conn.executemany('INSERT INTO documents (id, text) VALUES (?, ?)', [(d['id'], d['text']) for d in docs])
    conn.commit()
    conn.close()

    work_dir = tmp_path / 'work'

    # 1. Shard corpus (it will use the .db companion)
    result = runner.invoke(app, [
        'index-corpus',
        '--input', str(corpus_path),
        '--work-dir', str(work_dir),
        '--total-shards', '2'
    ])
    assert result.exit_code == 0

    # 2. Run init-index-shard for CSV
    from loguru import logger
    from io import StringIO
    for sid in [0, 1]:
        log_capture = StringIO()
        handler_id = logger.add(log_capture, level='INFO')
        try:
            result = runner.invoke(app, [
                'init-index-shard',
                '--input', str(corpus_path),
                '--work-dir', str(work_dir),
                '--run-name', 'csv_shard_test',
                '--shard-id', str(sid),
                '--total-shards', '2',
                '--collection', 'notes',
                '--backend', 'mock',
                '--model-path', 'dummy'
            ], catch_exceptions=False)
            assert result.exit_code == 0

            log_output = log_capture.getvalue()
            assert f'Indexing CSV shard {sid}/2 using SQLite backend' in log_output
        finally:
            logger.remove(handler_id)

    # Check metadata
    for sid in [0, 1]:
        meta_path = work_dir / 'runs' / 'csv_shard_test' / 'vectorstore' / 'notes' / 'shards' / f'shard_{sid:05d}' / 'metadata.jsonl'
        assert meta_path.exists()
        with open(meta_path, 'r', encoding='utf8') as f:
            meta = [json.loads(line) for line in f]
            assert len(meta) == 2  # 4 docs / 2 shards = 2 docs per shard
