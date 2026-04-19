from pathlib import Path


def detect_corpus_kind(path: Path) -> str:
    """Detect corpus kind from file extension."""
    suffix = path.suffix.lower()
    if suffix == '.jsonl':
        return 'jsonl'
    elif suffix in ('.sqlite', '.sqlite3', '.db'):
        return 'sqlite'
    elif suffix == '.csv':
        return 'csv'
    else:
        supported = '.jsonl, .sqlite, .sqlite3, .db, .csv'
        raise ValueError(f'Unsupported corpus extension: {suffix}. Supported: {supported}')
