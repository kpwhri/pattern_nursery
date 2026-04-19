from pathlib import Path


def get_global_vectorstore_path(work_dir: Path, run_name: str, collection: str) -> Path:
    """Return path to vectorstore for a collection within a run."""
    return get_run_dir(work_dir, run_name) / 'vectorstore' / collection


def get_run_dir(work_dir: Path, run_name: str) -> Path:
    """Return path to specific run directory."""
    if run_name is None:
        return work_dir
    return work_dir / 'runs' / run_name


def get_run_artifacts_path(work_dir: Path, run_name: str) -> Path:
    """Return path to run artifacts."""
    return get_run_dir(work_dir, run_name) / 'artifacts'


def get_run_logs_path(work_dir: Path, run_name: str) -> Path:
    """Return path to run logs."""
    return get_run_dir(work_dir, run_name) / 'logs'


def get_run_reports_path(work_dir: Path, run_name: str) -> Path:
    """Return path to run reports."""
    return get_run_dir(work_dir, run_name) / 'reports'


def ensure_run_dirs(work_dir: Path, run_name: str) -> None:
    """Create run directories if they do not exist."""
    get_run_artifacts_path(work_dir, run_name).mkdir(parents=True, exist_ok=True)
    get_run_logs_path(work_dir, run_name).mkdir(parents=True, exist_ok=True)
    get_run_reports_path(work_dir, run_name).mkdir(parents=True, exist_ok=True)
