import json

from typer.testing import CliRunner

from pattern_nursery.cli.main import app
from pattern_nursery.io.corpus_index import CorpusIndexSqlite

runner = CliRunner()


def test_shard_boundaries_correctness(tmp_path):
    corpus_path = tmp_path / 'corpus.jsonl'
    # 5 documents
    docs = [
        {'id': '1', 'text': 'Väinämöinen'},
        {'id': '2', 'text': 'Ilmarinen'},
        {'id': '3', 'text': 'Lemminkäinen'},
        {'id': '4', 'text': 'Joukahainen'},
        {'id': '5', 'text': 'Louhi'}
    ]
    with open(corpus_path, 'w', encoding='utf8') as f:
        for d in docs:
            f.write(json.dumps(d) + '\n')

    work_dir = tmp_path / 'work'

    # run index-corpus with 2 shards
    result = runner.invoke(app, [
        'index-corpus',
        '--input', str(corpus_path),
        '--work-dir', str(work_dir),
        '--total-shards', '2'
    ])
    assert result.exit_code == 0

    db_path = CorpusIndexSqlite.get_index_path(work_dir, corpus_path, 'id')
    assert db_path.exists()

    index = CorpusIndexSqlite(db_path)
    assert index.has_shard_boundaries(2)

    s0_start, s0_end = index.get_shard_boundary(2, 0)
    s1_start, s1_end = index.get_shard_boundary(2, 1)

    # doc_count = 5. total_shards = 2.
    # cut_indices = [floor(0*5/2), floor(1*5/2), 5] = [0, 2, 5]
    # shard 0: docs 0, 1. shard 1: docs 2, 3, 4.

    file_size = corpus_path.stat().st_size
    assert s0_start == 0
    assert s0_end == s1_start
    assert s1_end == file_size

    # check if boundaries are line starts
    with open(corpus_path, 'rb') as f:
        f.seek(s1_start)
        line = f.readline()
        doc = json.loads(line.decode('utf8'))
        assert doc['id'] == '3'  # third doc
