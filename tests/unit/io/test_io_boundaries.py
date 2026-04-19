import json

import pytest

from pattern_nursery.io import reader


def test_corpus_iter_empty_jsonl(tmp_path):
    """Ensure empty jsonl returns nothing."""
    p = tmp_path / 'empty.jsonl'
    p.write_text('', encoding='utf8')
    items = list(reader.corpus_iter(p, 'id', 'text'))
    assert len(items) == 0


def test_corpus_iter_malformed_jsonl(tmp_path):
    """Ensure malformed line is handled (currently json.loads will raise)."""
    # based on current implementation, it doesn't try-except
    p = tmp_path / 'bad.jsonl'
    p.write_text('{"id": "1", "text": "ok"}\n{broken json}\n', encoding='utf8')
    with pytest.raises(json.JSONDecodeError):
        list(reader.corpus_iter(p, 'id', 'text'))


def test_sentence_iter_empty_text(regex_engine):
    """Ensure empty text document yields no sentences."""
    docs = [('d1', '')]
    sentences = list(reader.sentence_iter(regex_engine, iter(docs)))
    assert len(sentences) == 0


def test_paragraph_iter_sentences_overlap_edge(regex_engine):
    """Test overlapping window logic."""
    docs = [('d1', 'S1. S2. S3. S4.')]
    # window 2, overlap 1 -> [S1, S2], [S2, S3], [S3, S4]
    paras = list(reader.paragraph_iter_sentences(regex_engine, iter(docs), window_size=2, overlap=1))
    assert len(paras) == 3
    assert paras[0][2] == 'S1. S2.'
    assert paras[1][2] == 'S2. S3.'
    assert paras[2][2] == 'S3. S4.'


def test_paragraph_iter_sentences_small_doc(regex_engine):
    """Test doc smaller than window."""
    docs = [('d1', 'S1. S2.')]
    paras = list(reader.paragraph_iter_sentences(regex_engine, iter(docs), window_size=5, overlap=1))
    assert len(paras) == 1
    assert paras[0][2] == 'S1. S2.'


def test_paragraph_iter_tokens_overlap_edge(regex_engine):
    """Test token window overlap."""
    docs = [('d1', 'one two three four five')]
    # window 3, overlap 2 -> [one two three], [two three four], [three four five]
    paras = list(reader.paragraph_iter_tokens(regex_engine, iter(docs), token_window=3, token_overlap=2))
    assert len(paras) == 3
    # Note: RegexEngine tokenizer uses \b\w+\b: 'one two three'
    assert paras[0][2] == 'one two three'
    assert paras[1][2] == 'two three four'
    assert paras[2][2] == 'three four five'
