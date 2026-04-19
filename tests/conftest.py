import json

import pytest

from pattern_nursery.nlp.regex_engine import RegexEngine


@pytest.fixture
def kalevala_corpus(tmp_path):
    """Create a small JSONL corpus with Kalevala entities."""
    corpus_path = tmp_path / 'kalevala.jsonl'
    data = [
        {'id': 'v_001', 'text': 'Väinämöinen played the kantele. The music echoed through the forest.'},
        {'id': 'i_002', 'text': 'Ilmarinen forged the Sampo. It was a task of great skill.'},
        {'id': 'l_003', 'text': 'Lemminkäinen traveled to Tuonela. He sought the swan of death.'},
        {'id': 'a_004', 'text': 'Aino wept by the sea. She did not wish to marry the old sage.'}
    ]
    with open(corpus_path, 'w', encoding='utf8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    return corpus_path


@pytest.fixture
def regex_engine():
    """Return a RegexEngine instance."""
    return RegexEngine()


@pytest.fixture
def tmp_workdir(tmp_path):
    """Return a temporary work directory."""
    return tmp_path / 'work_dir'
