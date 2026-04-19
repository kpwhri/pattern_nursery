from pattern_nursery.nlp.regex_engine import RegexEngine


def test_regex_engine_spans():
    engine = RegexEngine()
    text = "Hello world. This is a test."

    sentences = engine.split_sentences_with_spans(text)
    assert len(sentences) == 2
    assert sentences[0][0] == "Hello world."
    assert sentences[0][1] == 0
    assert sentences[0][2] == 12

    assert sentences[1][0] == "This is a test."
    assert sentences[1][1] == 13
    assert sentences[1][2] == 28

    # check if spans align to text
    for sent, start, end in sentences:
        assert text[start:end] == sent


def test_regex_engine_tokens():
    engine = RegexEngine()
    text = "One, two; three."
    tokens = engine.tokenize_with_spans(text)
    assert len(tokens) == 3
    assert tokens[0][0] == "One"
    assert tokens[1][0] == "two"
    assert tokens[2][0] == "three"

    for tok, start, end in tokens:
        assert text[start:end] == tok
