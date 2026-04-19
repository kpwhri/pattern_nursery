import csv
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pattern_nursery.io.corpus_backend import CsvCorpusBackend

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


def test_csv_corpus_backend_direct(tmp_path):
    csv_path = tmp_path / 'corpus.csv'
    db_path = tmp_path / 'corpus.db'
    create_csv_corpus(csv_path)
    create_sqlite_corpus(db_path)

    backend = CsvCorpusBackend(
        csv_path=csv_path,
        sqlite_db_path=db_path,
        table_name='documents',
        id_col='id',
        text_col='text'
    )

    # test validation
    backend.validate()

    # test iter_docs (from CSV)
    docs = list(backend.iter_docs())
    assert len(docs) == 4
    assert docs[0] == ('1', 'Väinämöinen, old and steadfast.')
    assert docs[3] == ('4', 'Louhi, mistress of the North.')

    # test get_text (from sqlite)
    assert backend.get_text('2') == 'Ilmarinen, smith of the sky.'
    with pytest.raises(KeyError):
        backend.get_text('999')

    # test iter_docs_shard (from sqlite)
    shard_0 = list(backend.iter_docs_shard(total_shards=2, shard_id=0))
    assert len(shard_0) == 2
    # doc_id, byte_offset (None for sqlite), text
    assert shard_0[0][0] == '1'
    assert shard_0[1][0] == '3'


def test_csv_detection():
    from pattern_nursery.io.corpus_types import detect_corpus_kind
    assert detect_corpus_kind(Path('foo.csv')) == 'csv'
