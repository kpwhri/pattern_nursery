import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from typer.testing import CliRunner

from pattern_nursery.cli.main import app

runner = CliRunner()


@pytest.fixture
def mock_backend():
    with patch('pattern_nursery.cli.main.get_embedding_backend') as mock:
        backend = MagicMock()
        backend.dimension = 384
        backend.model_name = 'mock_model'

        # ensure different sentences get different embeddings for sorting
        def side_effect(texts):
            if isinstance(texts, str):
                texts = [texts]
            return np.random.rand(len(texts), 384).astype(np.float32)

        backend.embed.side_effect = side_effect
        mock.return_value = backend
        yield mock


def test_retrieve_notes_segmentation(tmp_path, mock_backend):
    corpus_path = Path('tests/data/corpus.jsonl')
    work_dir = tmp_path / 'runs'

    # 1. Initialize index
    runner.invoke(app, [
        'init-index',
        '--input', str(corpus_path),
        '--work-dir', str(work_dir),
        '--run-name', 'test_seg',
        '--model-path', 'mock_model',
        '--collection', 'notes',
        '--engine', 'regex'
    ])

    # 2. Retrieve with segmentation
    result = runner.invoke(app, [
        'retrieve-notes',
        '--work-dir', str(work_dir),
        '--run-name', 'test_seg',
        '--seeds', 'Väinämöinen',
        '--segment-sentences',
        '--top-k-sentences', '3',
        '--top-k', '2'
    ])

    assert result.exit_code == 0

    results_path = work_dir / 'runs' / 'test_seg' / 'artifacts' / 'results_notes.jsonl'
    assert results_path.exists()

    with open(results_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        assert len(lines) > 0
        for line in lines:
            data = json.loads(line)
            assert 'relevant_sentences' in data
            assert len(data['relevant_sentences']) <= 3
            for sent in data['relevant_sentences']:
                assert 'text' in sent
                assert 'score' in sent

    # verify html report exists
    html_path = work_dir / 'runs' / 'test_seg' / 'reports' / 'notes.html'
    assert html_path.exists()
    html_content = html_path.read_text(encoding='utf8')
    assert 'Relevant Sentences:' in html_content
