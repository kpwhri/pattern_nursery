from pattern_nursery.index import paths


def test_paths_structure(tmp_path):
    """Verify run directory structure."""
    run_name = 'test_run'
    paths.ensure_run_dirs(tmp_path, run_name)

    run_dir = tmp_path / 'runs' / run_name
    assert run_dir.exists()
    assert (run_dir / 'artifacts').exists()
    assert (run_dir / 'logs').exists()
    assert (run_dir / 'reports').exists()


def test_global_store_path(tmp_path):
    """Verify global store path."""
    p = paths.get_global_vectorstore_path(tmp_path, 'test_run', 'vocab')
    assert p == tmp_path / 'runs' / 'test_run' / 'vectorstore' / 'vocab'
