import json

from typer.testing import CliRunner

from pattern_nursery.cli.main import app
from pattern_nursery.io.corpus_index import CorpusIndexSqlite

runner = CliRunner()


def get_shard_ids_from_db(corpus_path, work_dir, total_shards, shard_id):
    db_path = CorpusIndexSqlite.get_index_path(work_dir, corpus_path, 'id')
    index = CorpusIndexSqlite(db_path)
    start_offset, end_offset = index.get_shard_boundary(total_shards, shard_id)

    doc_ids = []
    with open(corpus_path, 'rb') as f:
        f.seek(start_offset)
        while f.tell() < end_offset:
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            doc = json.loads(line.decode('utf8'))
            doc_ids.append(doc['id'])
    return doc_ids


def test_byte_range_correctness(tmp_path):
    corpus_path = tmp_path / 'corpus.jsonl'
    work_dir = tmp_path / 'work'
    all_doc_ids = [f'doc_{i:03d}' for i in range(100)]
    # mix of short and long lines
    with open(corpus_path, 'wb') as f:
        for i, doc_id in enumerate(all_doc_ids):
            text = 'Short.' if i % 10 != 0 else 'Long ' * 100
            line = json.dumps({'id': doc_id, 'text': text}) + '\n'
            f.write(line.encode('utf8'))

    total_shards = 4

    # Pre-index
    result = runner.invoke(app, [
        'index-corpus',
        '--input', str(corpus_path),
        '--work-dir', str(work_dir),
        '--total-shards', str(total_shards)
    ])
    assert result.exit_code == 0

    collected_ids = []
    shard_results = []
    for s_id in range(total_shards):
        ids = get_shard_ids_from_db(corpus_path, work_dir, total_shards, s_id)
        shard_results.append(set(ids))
        collected_ids.extend(ids)

    # assert union equals all
    assert sorted(collected_ids) == sorted(all_doc_ids)

    # assert intersections are empty
    for i in range(total_shards):
        for j in range(i + 1, total_shards):
            intersection = shard_results[i].intersection(shard_results[j])
            assert not intersection, f'Shard {i} and {j} overlap: {intersection}'


def test_one_very_long_line_at_boundary(tmp_path):
    corpus_path = tmp_path / 'corpus.jsonl'
    work_dir = tmp_path / 'work'

    doc_0 = {'id': 'd0', 'text': 'a' * 10}
    doc_1 = {'id': 'd1', 'text': 'b' * 1000}
    doc_2 = {'id': 'd2', 'text': 'c' * 10}

    with open(corpus_path, 'w', encoding='utf8') as f:
        f.write(json.dumps(doc_0) + '\n')
        f.write(json.dumps(doc_1) + '\n')
        f.write(json.dumps(doc_2) + '\n')

    total_shards = 2
    # pre-index
    result = runner.invoke(app, [
        'index-corpus',
        '--input', str(corpus_path),
        '--work-dir', str(work_dir),
        '--total-shards', str(total_shards)
    ])
    assert result.exit_code == 0

    s0_ids = get_shard_ids_from_db(corpus_path, work_dir, total_shards, 0)
    s1_ids = get_shard_ids_from_db(corpus_path, work_dir, total_shards, 1)

    assert sorted(s0_ids + s1_ids) == ['d0', 'd1', 'd2']
    assert not set(s0_ids).intersection(set(s1_ids))
