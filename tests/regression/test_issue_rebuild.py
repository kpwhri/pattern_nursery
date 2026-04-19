import tomllib
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
        backend.model_name = 'mock_model_name'
        backend.embed.side_effect = lambda texts: np.random.rand(len(texts), 384).astype(np.float32)
        mock.return_value = backend
        yield mock


def test_init_index_no_rebuild_and_metadata(tmp_path, mock_backend):
    corpus_path = Path('tests/data/corpus.jsonl')
    work_dir = tmp_path / 'runs'

    # 1st run: use capsys to capture stderr if needed, or just check log files
    log_dir = work_dir / 'runs' / 'run1' / 'logs'
    log_file = log_dir / 'app.log'

    with patch('pattern_nursery.cli.main.setup_logging') as mock_setup:
        # manual log file for test
        from loguru import logger
        log_dir.mkdir(parents=True, exist_ok=True)
        handler_id = logger.add(log_file)

        result = runner.invoke(app, [
            'init-index',
            '--input', str(corpus_path),
            '--work-dir', str(work_dir),
            '--run-name', 'run1',
            '--model-path', 'mock_model',
            '--collection', 'sentences',
            '--collection', 'paragraphs',
            '--paragraph-mode', 'sentences',
            '--window-size', '5'
        ], catch_exceptions=False)
        logger.remove(handler_id)

    assert result.exit_code == 0
    log_content = log_file.read_text()
    assert 'Indexing sentences...' in log_content
    assert 'Indexing paragraphs (mode=sentences)...' in log_content

    # check config.toml for metadata
    config_path = work_dir / 'runs' / 'run1' / 'vectorstore' / 'paragraphs' / 'config.toml'
    assert config_path.exists()
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)
        assert config['embedding_model'] == 'mock_model_name'
        assert config['nlp_engine'] == 'RegexEngine'
        assert config['mode'] == 'sentences'
        assert config['window_size'] == 5

    # 2nd run: different run_name but same work_dir
    log_dir2 = work_dir / 'runs' / 'run2' / 'logs'
    log_file2 = log_dir2 / 'app.log'

    with patch('pattern_nursery.cli.main.setup_logging'):
        from loguru import logger
        log_dir2.mkdir(parents=True, exist_ok=True)
        handler_id = logger.add(log_file2)

        result2 = runner.invoke(app, [
            'init-index',
            '--input', str(corpus_path),
            '--work-dir', str(work_dir),
            '--run-name', 'run1',
            '--model-path', 'mock_model',
            '--collection', 'sentences',
            '--collection', 'paragraphs'
        ], catch_exceptions=False)
        logger.remove(handler_id)

    assert result2.exit_code == 0
    log_content2 = log_file2.read_text()
    assert 'Sentences index already exists. Skipping...' in log_content2
    assert 'Paragraphs index already exists. Skipping...' in log_content2
    assert 'Indexing sentences...' not in log_content2
