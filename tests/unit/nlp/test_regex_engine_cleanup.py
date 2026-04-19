from pattern_nursery.nlp.regex_engine import RegexEngine


def test_regex_engine_splitting_finnish_gods():
    engine = RegexEngine()

    # test splitting by punctuation and newlines
    text = "Väinämöinen lauloi. \nIlmarinen takoi Sammon.  \n  \nLouhi emäntä Pohjolan."
    sentences = engine.split_sentences(text)

    assert len(sentences) == 3
    assert sentences[0] == "Väinämöinen lauloi."
    assert sentences[1] == "Ilmarinen takoi Sammon."
    assert sentences[2] == "Louhi emäntä Pohjolan."


def test_regex_engine_newline_noise_cleanup():
    engine = RegexEngine()

    # test cases reported in the issue
    text = "Confidential \n This is a test.  \n"
    sentences = engine.split_sentences(text)

    assert len(sentences) == 2
    assert sentences[0] == "Confidential"
    assert sentences[1] == "This is a test."

    # test just noise
    assert engine.split_sentences("  \n  ") == []
    assert engine.tokenize("  \n  ") == []


def test_regex_engine_multiple_newlines():
    engine = RegexEngine()
    text = "Lemminkäinen ratsastaa\n\nhiiden hirven perässä."
    sentences = engine.split_sentences(text)

    assert len(sentences) == 2
    assert sentences[0] == "Lemminkäinen ratsastaa"
    assert sentences[1] == "hiiden hirven perässä."


def test_regex_engine_spans_with_newlines():
    engine = RegexEngine()
    text = "Aino \n hukkui."
    spans = engine.split_sentences_with_spans(text)

    # check newlines
    assert len(spans) == 2
    assert spans[0][0] == "Aino"
    assert spans[1][0] == "hukkui."
