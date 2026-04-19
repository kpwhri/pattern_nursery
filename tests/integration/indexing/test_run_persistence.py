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
        backend.model_name = 'mock_model_name'
        backend.embed.side_effect = lambda texts: np.random.rand(len(texts), 384).astype(np.float32)
        mock.return_value = backend
        yield mock


def test_run_config_persistence_and_relocation(tmp_path, mock_backend):
    corpus_path = Path('tests/data/corpus.jsonl')
    work_dir = tmp_path / 'runs'
    run_name = 'test_persistence'
    model_path = 'local/path/to/model'

    # 1. Init index with all params
    result = runner.invoke(app, [
        'init-index',
        '--input', str(corpus_path),
        '--work-dir', str(work_dir),
        '--run-name', run_name,
        '--model-path', model_path,
        '--collection', 'sentences'
    ])
    assert result.exit_code == 0

    # verify relocation: vectorstore should be INSIDE the run directory
    expected_store_path = work_dir / 'runs' / run_name / 'vectorstore' / 'sentences'
    assert expected_store_path.exists()
    assert (expected_store_path / 'config.toml').exists()

    # verify run_config.json exists
    config_path = work_dir / 'runs' / run_name / 'run_config.json'
    assert config_path.exists()
    with open(config_path, 'r') as f:
        saved_config = json.load(f)
        assert saved_config['model_path'] == model_path
        assert saved_config['run_name'] == run_name

    # 2. Run retrieval WITHOUT model_path or backend - it should be reused
    with patch('pattern_nursery.cli.main.setup_logging'):
        result = runner.invoke(app, [
            'retrieve-sentences',
            '--work-dir', str(work_dir),
            '--run-name', run_name,
            '--seeds', 'Sampo'
        ], catch_exceptions=False)

    assert result.exit_code == 0
    # TODO: verify it used the saved model_path by checking logs??


def test_log_run_info(tmp_path, mock_backend):
    corpus_path = Path('tests/data/corpus.jsonl')
    work_dir = tmp_path / 'runs'
    run_name = 'test_logs'
    model_path = 'some/model'

    log_file = work_dir / 'runs' / run_name / 'logs' / 'app.log'

    with patch('pattern_nursery.cli.main.setup_logging'):
        from loguru import logger
        log_file.parent.mkdir(parents=True, exist_ok=True)
        hid = logger.add(log_file)

        runner.invoke(app, [
            'init-index',
            '--input', str(corpus_path),
            '--work-dir', str(work_dir),
            '--run-name', run_name,
            '--model-path', model_path,
            '--collection', 'sentences'
        ])
        logger.remove(hid)

    # force flush if possible
    logger.remove()

    log_content = log_file.read_text()
    assert 'command_start' in log_content
    # TODO: what else should be here?
