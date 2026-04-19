import json

from pattern_nursery.io.corpus_index import CorpusIndexSqlite


def test_corpus_index_skip_duplicate_id_jsonl(tmp_path):
    corpus_path = tmp_path / 'corpus.jsonl'
    # Väinämöinen appears twice with different text, but same ID
    docs = [
        {'id': 'v_vainamoinen', 'text': 'Vainamoinen played his kantele.'},
        {'id': 'v_vainamoinen', 'text': 'Vainamoinen sang his songs.'},
        {'id': 'i_ilmarinen', 'text': 'Ilmarinen forged the Sampo.'}
    ]
    with open(corpus_path, 'w', encoding='utf8') as f:
        for doc in docs:
            f.write(json.dumps(doc) + '\n')

    db_path = tmp_path / 'index.sqlite'
    index = CorpusIndexSqlite(db_path)

    # use on_duplicate_id='skip'
    count = index.build_from_jsonl(corpus_path, id_col='id', text_col='text', on_duplicate_id='skip')

    assert count == 3  # number of lines processed

    # Check that only the FIRST one is indexed (standard SQL INSERT behavior if we skip duplicates)
    offset = index.get_offset('v_vainamoinen')
    assert offset is not None

    doc = index.get_doc_by_offset(corpus_path, offset)
    assert doc['text'] == 'Vainamoinen played his kantele.'

    # Check ilmarinen
    offset_i = index.get_offset('i_ilmarinen')
    assert offset_i is not None
    doc_i = index.get_doc_by_offset(corpus_path, offset_i)
    assert doc_i['text'] == 'Ilmarinen forged the Sampo.'


def test_corpus_index_skip_duplicate_id_csv(tmp_path):
    corpus_path = tmp_path / 'corpus.csv'
    content = (
        'id,text\n'
        'v,Vainamoinen\n'
        'v,Duplicate\n'
        'i,Ilmarinen\n'
    )
    corpus_path.write_text(content, encoding='utf8')

    db_path = tmp_path / 'index.sqlite'
    index = CorpusIndexSqlite(db_path)

    count = index.build_from_csv(corpus_path, id_col='id', text_col='text', on_duplicate_id='skip')
    assert count == 3

    offset = index.get_offset('v')
    doc = index.get_doc_by_offset(corpus_path, offset)
    assert doc['text'] == 'Vainamoinen'
