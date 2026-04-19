import time
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
        backend.embed.side_effect = lambda texts: np.random.rand(len(texts), 384).astype(np.float32)
        mock.return_value = backend
        yield mock


def test_logging_init_index(tmp_path, mock_backend):
    """Test that init-index produces an app log with expected markers."""
    corpus_path = Path('tests/data/corpus.jsonl')
    work_dir = tmp_path
    run_name = 'kantele_run'

    result = runner.invoke(app, [
        'init-index',
        '--input', str(corpus_path),
        '--work-dir', str(work_dir),
        '--run-name', run_name,
        '--model-path', 'mock_model',
        '--backend', 'mock',
        '--collection', 'sentences'
    ])

    assert result.exit_code == 0

    # wait for enqueued logs to be written
    time.sleep(0.5)

    log_file = work_dir / 'runs' / run_name / 'logs' / 'app.log'
    assert log_file.exists()

    log_content = log_file.read_text(encoding='utf8')
    assert 'command_start' in log_content
    assert 'command_done' in log_content
    assert "'command': 'init_index'" in log_content
    assert 'elapsed_sec' in log_content
    assert 'pid=' in log_content


def test_logging_retrieve_sentences(tmp_path, mock_backend):
    """Test that retrieve-sentences logs seed count and output path."""
    corpus_path = Path('tests/data/corpus.jsonl')
    work_dir = tmp_path
    run_name = 'sampo_run'

    # 1. init
    runner.invoke(app, [
        'init-index',
        '--input', str(corpus_path),
        '--work-dir', str(work_dir),
        '--run-name', run_name,
        '--model-path', 'mock_model',
        '--backend', 'mock',
        '--collection', 'sentences'
    ])

    # 2. retrieve
    result = runner.invoke(app, [
        'retrieve-sentences',
        '--work-dir', str(work_dir),
        '--run-name', run_name,
        '--seeds', 'Louhi',
        '--seeds', 'Ilmarinen',
        '--model-path', 'mock_model',
        '--backend', 'mock',
        '--top-k', '2'
    ])

    assert result.exit_code == 0

    # wait for enqueued logs to be written
    time.sleep(0.5)

    log_file = work_dir / 'runs' / run_name / 'logs' / 'app.log'
    assert log_file.exists()

    log_content = log_file.read_text(encoding='utf8')
    assert "'command': 'retrieve_sentences'" in log_content
    assert 'command_start' in log_content
    assert 'command_done' in log_content
    assert "'seeds_count': 2" in log_content
    assert 'results_sentences.jsonl' in log_content
