import json

from typer.testing import CliRunner

from pattern_nursery.cli.main import app

runner = CliRunner()


def test_sharded_retrieval_and_reconstruction(tmp_path):
    # 1. Create corpus
    corpus_path = tmp_path / 'corpus.jsonl'
    docs = [
        {'id': f'doc_{i}', 'text': f'This is document {i}. It contains some special terms like Väinämöinen.'}
        for i in range(10)
    ]
    with open(corpus_path, 'w', encoding='utf8') as f:
        for doc in docs:
            f.write(json.dumps(doc) + '\n')

    work_dir = tmp_path / 'work'

    # 2. Build corpus index with 2 shards
    result = runner.invoke(app, [
        'index-corpus',
        '--input', str(corpus_path),
        '--work-dir', str(work_dir),
        '--total-shards', '2'
    ])
    assert result.exit_code == 0

    # 3. Build shards
    for shard_id in [0, 1]:
        result = runner.invoke(app, [
            'init-index-shard',
            '--input', str(corpus_path),
            '--work-dir', str(work_dir),
            '--run-name', 'sharded_run',
            '--total-shards', '2',
            '--shard-id', str(shard_id),
            '--collection', 'notes',
            '--collection', 'sentences',
            '--backend', 'mock',
            '--model-path', 'dummy'
        ])
        assert result.exit_code == 0

    # Create run_config.json manually since init-index-shard doesn't do it
    run_dir = work_dir / 'runs' / 'sharded_run'
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'run_config.json', 'w', encoding='utf8') as f:
        json.dump({
            'input': str(corpus_path.absolute()),
            'id_col': 'id',
            'text_col': 'text',
            'model_path': 'dummy',
            'backend': 'mock'
        }, f)

    # 4. Retrieve notes
    result = runner.invoke(app, [
        'retrieve-notes',
        '--work-dir', str(work_dir),
        '--run-name', 'sharded_run',
        '--seeds', 'document',
        '--backend', 'mock',
        '--model-path', 'dummy'
    ])
    assert result.exit_code == 0

    # Check results
    res_path = work_dir / 'runs' / 'sharded_run' / 'artifacts' / 'results_notes.jsonl'
    assert res_path.exists()
    with open(res_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        assert len(lines) > 0
        first_res = json.loads(lines[0])
        assert 'text' in first_res
        assert 'doc_byte_offset' in first_res
        # Text should be reconstructed from corpus
        assert 'This is document' in first_res['text']

    # 5. Retrieve sentences (should aggregate shards and reconstruct spans)
    result = runner.invoke(app, [
        'retrieve-sentences',
        '--work-dir', str(work_dir),
        '--run-name', 'sharded_run',
        '--seeds', 'Väinämöinen',
        '--backend', 'mock',
        '--model-path', 'dummy'
    ])
    assert result.exit_code == 0

    res_path = work_dir / 'runs' / 'sharded_run' / 'artifacts' / 'results_sentences.jsonl'
    assert res_path.exists()
    with open(res_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        assert len(lines) > 0
        # Check that we found at least one result with text reconstructed
        any_text = any('text' in json.loads(line) for line in lines)
        assert any_text
