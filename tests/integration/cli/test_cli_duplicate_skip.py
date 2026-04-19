from typer.testing import CliRunner
from pattern_nursery.cli.main import app
import json

runner = CliRunner()

def test_cli_index_corpus_skip_duplicates(tmp_path):
    corpus_path = tmp_path / 'corpus.jsonl'
    docs = [
        {'id': 'v', 'text': 'first'},
        {'id': 'v', 'text': 'second'},
    ]
    with open(corpus_path, 'w') as f:
        for d in docs:
            f.write(json.dumps(d) + '\n')
            
    work_dir = tmp_path / 'work'
    work_dir.mkdir()
    
    # try without skip - should fail
    result = runner.invoke(app, [
        'index-corpus',
        '--input', str(corpus_path),
        '--work-dir', str(work_dir),
        '--on-duplicate-id', 'raise'
    ])
    assert result.exit_code != 0
    
    # try with skip - should succeed
    result = runner.invoke(app, [
        'index-corpus',
        '--input', str(corpus_path),
        '--work-dir', str(work_dir),
        '--on-duplicate-id', 'skip',
        '--force-rebuild'
    ])
    assert result.exit_code == 0  # TODO: database is locked exit code?
    
    # TODO: Check that it works in init-index too
