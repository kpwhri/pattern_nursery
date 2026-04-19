import json

from typer.testing import CliRunner

from pattern_nursery.cli.main import app
from pattern_nursery.embeddings.vectorstores.faiss_store import FaissStore
from pattern_nursery.index import paths

runner = CliRunner()


def test_init_index_shard_flow(tmp_path):
    # 1. create corpus
    corpus_path = tmp_path / 'corpus.jsonl'
    kalevala_names = ['Vainamoinen', 'Ilmarinen', 'Lemminkainen', 'Vipunen', 'Aino', 'Louhi', 'Sampo', 'Kullervo',
                      'Untamo', 'Joukahainen']
    docs = [
        {'id': f'doc_{name}', 'text': f'{name} sang a song of ancient times. This is sentence two.'}
        for name in kalevala_names
    ]
    with open(corpus_path, 'w', encoding='utf8') as f:
        for doc in docs:
            f.write(json.dumps(doc) + '\n')

    work_dir = tmp_path / 'work'

    # 2. run index-corpus with total-shards
    result = runner.invoke(app, [
        'index-corpus',
        '--input', str(corpus_path),
        '--work-dir', str(work_dir),
        '--total-shards', '1'
    ])
    assert result.exit_code == 0

    # 3. run init-index-shard (single shard) with mock backend
    result = runner.invoke(app, [
        'init-index-shard',
        '--input', str(corpus_path),
        '--work-dir', str(work_dir),
        '--run-name', 'test_run',
        '--total-shards', '1',
        '--shard-id', '0',
        '--collection', 'notes',
        '--backend', 'mock',
        '--model-path', 'dummy'
    ], catch_exceptions=False)
    assert result.exit_code == 0

    # 4. check manifest
    manifest_path = paths.get_global_vectorstore_path(
        work_dir, 'test_run',
        'notes') / 'shards' / 'shard_00000' / 'manifest.json'
    assert manifest_path.exists()
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest['status'] == 'done'
    assert manifest['collection'] == 'notes'
    assert manifest['counts']['docs_seen'] == 10
    assert manifest['counts']['vectors_added'] == 10
    assert 'timing' in manifest
    assert 'corpus' in manifest
    assert manifest['shard_boundary']['start_byte_offset'] == 0

    # 5. failure case when index is missing
    work_dir_2 = tmp_path / 'work_no_index'
    result = runner.invoke(app, [
        'init-index-shard',
        '--input', str(corpus_path),
        '--work-dir', str(work_dir_2),
        '--run-name', 'test_run',
        '--model-path', 'dummy'
    ])
    assert result.exit_code == 1


def test_faiss_ivf_training(tmp_path):
    import numpy as np
    dim = 64
    store = FaissStore(dimension=dim, index_type='ivfflatip', nlist=10)
    assert not store.index.is_trained

    # train
    data = np.random.random((100, dim)).astype('float32')
    store.train(data)
    assert store.index.is_trained

    # add
    meta = [{'id': i} for i in range(100)]
    store.add(data, meta)
    assert store.count() == 100

    # query
    results = store.query(data[:1], top_k=5, nprobe=2)
    assert len(results[0]) == 5


def test_init_index_shard_vainamoinen(tmp_path):
    work_dir = tmp_path / 'work'
    corpus_path = tmp_path / 'corpus.jsonl'

    # create a small corpus
    with open(corpus_path, 'w', encoding='utf8') as f:
        f.write(json.dumps({'id': '1', 'text': 'Väinämöinen played the kantele.'}) + '\n')

    # 1. index-corpus to create boundaries
    result = runner.invoke(app, [
        'index-corpus',
        '--input', str(corpus_path),
        '--work-dir', str(work_dir),
        '--total-shards', '1'
    ])
    assert result.exit_code == 0

    # 2. init-index-shard
    result = runner.invoke(app, [
        'init-index-shard',
        '--input', str(corpus_path),
        '--work-dir', str(work_dir),
        '--run-name', 'test_run',
        '--shard-id', '0',
        '--total-shards', '1',
        '--model-path', 'dummy',
        '--backend', 'mock'
    ])

    assert result.exit_code == 0
