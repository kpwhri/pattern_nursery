import json

import pytest

from pattern_nursery.io.corpus_index import CorpusIndexSqlite


def test_corpus_index_build_lookup(tmp_path):
    corpus_path = tmp_path / 'corpus.jsonl'
    docs = [
        {'id': 'v_vainamoinen', 'text': 'Vainamoinen played his kantele.'},
        {'id': 'i_ilmarinen', 'text': 'Ilmarinen forged the Sampo.'},
        {'id': 'l_louhi', 'text': 'Louhi was the mistress of Pohjola.'}
    ]
    with open(corpus_path, 'w', encoding='utf8') as f:
        for doc in docs:
            f.write(json.dumps(doc) + '\n')

    db_path = tmp_path / 'index.sqlite'
    index = CorpusIndexSqlite(db_path)
    count = index.build_from_jsonl(corpus_path, id_col='id', text_col='text')
    assert count == 3

    for doc in docs:
        offset = index.get_offset(doc['id'])
        assert offset is not None

        retrieved_doc = index.get_doc_by_offset(corpus_path, offset)
        assert retrieved_doc == doc

        retrieved_doc_by_id = index.get_doc_by_id(corpus_path, doc['id'])
        assert retrieved_doc_by_id == doc


def test_corpus_index_validation(tmp_path):
    corpus_path = tmp_path / 'corpus.jsonl'
    doc = {'id': 'v', 'text': 't'}
    with open(corpus_path, 'w', encoding='utf8') as f:
        f.write(json.dumps(doc) + '\n')

    db_path = tmp_path / 'index.sqlite'
    index = CorpusIndexSqlite(db_path)
    index.build_from_jsonl(corpus_path, id_col='id', text_col='text')

    # validate should pass
    index.validate_corpus(corpus_path)

    # modify corpus
    import time
    time.sleep(0.01)  # ensure mtime change
    with open(corpus_path, 'a', encoding='utf8') as f:
        f.write(json.dumps({'id': 'v2', 'text': 't2'}) + '\n')

    with pytest.raises(ValueError):
        index.validate_corpus(corpus_path)


def test_duplicate_id_raises_error(tmp_path):
    corpus_path = tmp_path / 'corpus.jsonl'
    docs = [
        {'id': 'v', 'text': 't1'},
        {'id': 'v', 'text': 't2'}
    ]
    with open(corpus_path, 'w', encoding='utf8') as f:
        for doc in docs:
            f.write(json.dumps(doc) + '\n')

    db_path = tmp_path / 'index.sqlite'
    index = CorpusIndexSqlite(db_path)
    with pytest.raises(ValueError, match='Duplicate doc_id encountered'):
        index.build_from_jsonl(corpus_path, id_col='id', text_col='text')
