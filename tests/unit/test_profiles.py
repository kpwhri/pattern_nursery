import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from typer.testing import CliRunner

from pattern_nursery.cli.main import app

runner = CliRunner()


@pytest.fixture
def temp_config_dir(tmp_path):
    config_dir = tmp_path / 'config'
    os.environ['PATTERN_NURSERY_CONFIG_DIR'] = str(config_dir)
    yield config_dir
    if 'PATTERN_NURSERY_CONFIG_DIR' in os.environ:
        del os.environ['PATTERN_NURSERY_CONFIG_DIR']


@pytest.fixture
def mock_backend():
    with patch('pattern_nursery.cli.main.get_embedding_backend') as mock:
        backend = MagicMock()
        backend.dimension = 384
        backend.model_name = 'mock_model'
        backend.embed.side_effect = lambda texts: np.random.rand(len(texts), 384).astype(np.float32)
        mock.return_value = backend
        yield mock


def test_profile_written_on_init(temp_config_dir, mock_backend, tmp_path):
    corpus_path = Path('tests/data/corpus.jsonl')
    work_dir = tmp_path / 'runs'

    result = runner.invoke(app, [
        'init-index',
        '--input', str(corpus_path),
        '--work-dir', str(work_dir),
        '--run-name', 'run_p1',
        '--profile', 'p1',
        '--model-path', 'mock_model',
        '--collection', 'vocab_terms'
    ])

    assert result.exit_code == 0
    profiles_file = temp_config_dir / 'profiles.json'
    assert profiles_file.exists()

    with open(profiles_file, 'r', encoding='utf8') as f:
        data = json.load(f)
        assert 'p1' in data['profiles']
        assert data['profiles']['p1']['work_dir'] == str(work_dir)
        assert data['profiles']['p1']['run_name'] == 'run_p1'


def test_multi_profile_separation(temp_config_dir, mock_backend, tmp_path):
    # set profile A
    runner.invoke(app, [
        'use-index',
        '--profile', 'A',
        '--work-dir', str(tmp_path / 'wa'),
        '--run-name', 'ra'
    ])

    # set profile B
    runner.invoke(app, [
        'use-index',
        '--profile', 'B',
        '--work-dir', str(tmp_path / 'wb'),
        '--run-name', 'rb'
    ])

    profiles_file = temp_config_dir / 'profiles.json'
    with open(profiles_file, 'r', encoding='utf8') as f:
        data = json.load(f)
        assert data['profiles']['A']['run_name'] == 'ra'
        assert data['profiles']['B']['run_name'] == 'rb'


def test_profile_defaults_reading(temp_config_dir, mock_backend, tmp_path):
    work_dir = tmp_path / 'w_def'
    run_name = 'r_def'

    # Pre-seed profile
    profiles_file = temp_config_dir / 'profiles.json'
    temp_config_dir.mkdir(parents=True, exist_ok=True)
    with open(profiles_file, 'w', encoding='utf8') as f:
        json.dump({
            'version': 1,
            'profiles': {
                'p_def': {
                    'work_dir': str(work_dir),
                    'run_name': run_name
                }
            }
        }, f)

    # mock load_store/manager
    with patch('pattern_nursery.cli.main.IndexManager') as mock_mgr_cls:
        mock_mgr = MagicMock()
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(app, [
            'retrieve-sentences',
            '--profile', 'p_def',
            '--seeds', 'test',
            '--model-path', 'mock_model'
        ])

        # check that IndexManager was called with defaults from profile
        mock_mgr_cls.assert_called_with(work_dir, run_name)
