from pattern_nursery.io import reader
from pattern_nursery.nlp.regex_engine import RegexEngine


def test_paragraph_iter_sentences():
    engine = RegexEngine()
    docs = [("doc1", "S1. S2. S3. S4. S5.")]

    # window_size=3, overlap=1 -> step=2
    # para1: S1, S2, S3
    # para2: S3, S4, S5
    paras = list(reader.paragraph_iter_sentences(engine, iter(docs), window_size=3, overlap=1))

    assert len(paras) == 2
    assert paras[0][2] == "S1. S2. S3."
    assert paras[1][2] == "S3. S4. S5."
    assert paras[0][5] == 0
    assert paras[0][6] == 2
    assert paras[1][5] == 2
    assert paras[1][6] == 4


def test_paragraph_iter_tokens():
    engine = RegexEngine()
    docs = [("doc1", "one two three four five six seven eight nine ten")]

    # window=5, overlap=1 -> step=4
    # para1: one two three four five
    # para2: five six seven eight nine
    # para3: nine ten
    paras = list(reader.paragraph_iter_tokens(engine, iter(docs), token_window=5, token_overlap=1))

    assert len(paras) == 3
    assert paras[0][2] == "one two three four five"
    assert paras[1][2] == "five six seven eight nine"
    assert paras[2][2] == "nine ten"
