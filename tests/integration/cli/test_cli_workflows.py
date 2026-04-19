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


def test_cli_init_index(tmp_path, mock_backend, kalevala_corpus):
    corpus_path = kalevala_corpus
    work_dir = tmp_path / 'runs'

    result = runner.invoke(app, [
        'init-index',
        '--input', str(corpus_path),
        '--work-dir', str(work_dir),
        '--run-name', 'test_run',
        '--model-path', 'mock_model',
        '--backend', 'mock',
        '--collection', 'vocab_terms',
        '--collection', 'sentences',
        '--collection', 'paragraphs'
    ])

    assert result.exit_code == 0

    # check structure
    assert (work_dir / 'runs' / 'test_run' / 'vectorstore' / 'vocab_terms' / 'config.toml').exists()
    assert (work_dir / 'runs' / 'test_run' / 'vectorstore' / 'sentences' / 'config.toml').exists()
    assert (work_dir / 'runs' / 'test_run' / 'vectorstore' / 'paragraphs' / 'config.toml').exists()
    assert (work_dir / 'runs' / 'test_run' / 'artifacts' / 'vocab.txt').exists()
    assert (work_dir / 'runs' / 'test_run' / 'logs' / 'app.log').exists()


def test_cli_retrieve_paragraphs(tmp_path, mock_backend):
    # first init
    corpus_path = Path('tests/data/corpus.jsonl')
    work_dir = tmp_path / 'runs'
    runner.invoke(app, [
        'init-index',
        '--input', str(corpus_path),
        '--work-dir', str(work_dir),
        '--run-name', 'test_run',
        '--model-path', 'mock_model',
        '--backend', 'mock',
        '--collection', 'paragraphs'
    ])

    # then retrieve
    result = runner.invoke(app, [
        'retrieve-paragraphs',
        '--work-dir', str(work_dir),
        '--run-name', 'test_run',
        '--seeds', 'Väinämöinen',
        '--model-path', 'mock_model',
        '--backend', 'mock',
        '--top-k', '5'
    ])

    assert result.exit_code == 0
    assert (work_dir / 'runs' / 'test_run' / 'artifacts' / 'results_paragraphs.jsonl').exists()
    assert (work_dir / 'runs' / 'test_run' / 'reports' / 'paragraphs.html').exists()


def test_cli_expand_from_regex(tmp_path, mock_backend):
    corpus_path = Path('tests/data/corpus.jsonl')
    work_dir = tmp_path / 'runs'
    runner.invoke(app, [
        'init-index',
        '--input', str(corpus_path),
        '--work-dir', str(work_dir),
        '--run-name', 'test_run',
        '--model-path', 'mock_model',
        '--backend', 'mock',
        '--collection', 'vocab_terms',
        '--collection', 'sentences'
    ])

    result = runner.invoke(app, [
        'expand-from-regex',
        '--input', str(corpus_path),
        '--work-dir', str(work_dir),
        '--run-name', 'test_run',
        '--regex', 'Väinä.*',
        '--model-path', 'mock_model',
        '--backend', 'mock'
    ])

    assert result.exit_code == 0
    assert (work_dir / 'runs' / 'test_run' / 'artifacts' / 'regex_expansion.jsonl').exists()
    assert (work_dir / 'runs' / 'test_run' / 'reports' / 'regex_expansion.html').exists()


def test_cli_retrieve_notes(tmp_path, mock_backend):
    corpus_path = Path('tests/data/corpus.jsonl')
    work_dir = tmp_path / 'runs'
    runner.invoke(app, [
        'init-index',
        '--input', str(corpus_path),
        '--work-dir', str(work_dir),
        '--run-name', 'test_run',
        '--model-path', 'mock_model',
        '--backend', 'mock',
        '--collection', 'notes'
    ])

    result = runner.invoke(app, [
        'retrieve-notes',
        '--work-dir', str(work_dir),
        '--run-name', 'test_run',
        '--seeds', 'Sampo',
        '--model-path', 'mock_model',
        '--backend', 'mock'
    ])
    assert result.exit_code == 0
    assert (work_dir / 'runs' / 'test_run' / 'artifacts' / 'results_notes.jsonl').exists()
    assert (work_dir / 'runs' / 'test_run' / 'reports' / 'notes.html').exists()


def test_cli_retrieve_sentences(tmp_path, mock_backend):
    corpus_path = Path('tests/data/corpus.jsonl')
    work_dir = tmp_path / 'runs'
    runner.invoke(app, [
        'init-index',
        '--input', str(corpus_path),
        '--work-dir', str(work_dir),
        '--run-name', 'test_run',
        '--model-path', 'mock_model',
        '--backend', 'mock',
        '--collection', 'sentences'
    ])

    result = runner.invoke(app, [
        'retrieve-sentences',
        '--work-dir', str(work_dir),
        '--run-name', 'test_run',
        '--seeds', 'Ilmarinen',
        '--model-path', 'mock_model',
        '--backend', 'mock'
    ])
    assert result.exit_code == 0
    assert (work_dir / 'runs' / 'test_run' / 'artifacts' / 'results_sentences.jsonl').exists()
    assert (work_dir / 'runs' / 'test_run' / 'reports' / 'sentences.html').exists()


def test_cli_retrieve_phrases(tmp_path, mock_backend):
    corpus_path = Path('tests/data/corpus.jsonl')
    work_dir = tmp_path / 'runs'
    runner.invoke(app, [
        'init-index',
        '--input', str(corpus_path),
        '--work-dir', str(work_dir),
        '--run-name', 'test_run',
        '--model-path', 'mock_model',
        '--backend', 'mock',
        '--collection', 'phrases'
    ])

    result = runner.invoke(app, [
        'retrieve-phrases',
        '--work-dir', str(work_dir),
        '--run-name', 'test_run',
        '--seeds', 'Louhi',
        '--model-path', 'mock_model',
        '--backend', 'mock'
    ])
    assert result.exit_code == 0
    assert (work_dir / 'runs' / 'test_run' / 'vectorstore' / 'phrases' / 'config.toml').exists()
    assert (work_dir / 'runs' / 'test_run' / 'artifacts' / 'results_phrases.jsonl').exists()
    assert (work_dir / 'runs' / 'test_run' / 'reports' / 'phrases.html').exists()


def test_cli_expand_terms(tmp_path, mock_backend):
    corpus_path = Path('tests/data/corpus.jsonl')
    work_dir = tmp_path / 'runs'
    runner.invoke(app, [
        'init-index',
        '--input', str(corpus_path),
        '--work-dir', str(work_dir),
        '--run-name', 'test_run',
        '--model-path', 'mock_model',
        '--backend', 'mock',
        '--collection', 'vocab_terms'
    ])

    result = runner.invoke(app, [
        'expand-terms',
        '--work-dir', str(work_dir),
        '--run-name', 'test_run',
        '--seeds', 'Pohjola',
        '--model-path', 'mock_model',
        '--backend', 'mock'
    ])
    assert result.exit_code == 0
    assert (work_dir / 'runs' / 'test_run' / 'artifacts' / 'expansion_terms.jsonl').exists()
    assert (work_dir / 'runs' / 'test_run' / 'reports' / 'expansion_terms.html').exists()
