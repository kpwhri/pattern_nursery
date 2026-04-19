import json
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any

from loguru import logger


class CorpusIndexSqlite:
    """SQLite-backed index for document byte offsets in a JSONL corpus."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._ensure_schema()
        return self._conn

    def _ensure_schema(self):
        conn = self._get_conn()
        conn.execute('''
                     CREATE TABLE IF NOT EXISTS meta
                     (
                         key TEXT PRIMARY KEY,
                         value TEXT NOT NULL
                     )
                     ''')
        conn.execute('''
                     CREATE TABLE IF NOT EXISTS doc_offsets
                     (
                         doc_id TEXT PRIMARY KEY,
                         byte_offset INTEGER NOT NULL
                     )
                     ''')
        conn.execute('''
                     CREATE TABLE IF NOT EXISTS shards
                     (
                         total_shards INTEGER NOT NULL,
                         shard_id INTEGER NOT NULL,
                         start_byte_offset INTEGER NOT NULL,
                         end_byte_offset INTEGER NOT NULL,
                         PRIMARY KEY (total_shards,shard_id)
                         )
                     ''')
        conn.commit()

    @staticmethod
    def get_index_path(work_dir: Path, corpus_path: Path, id_col: str) -> Path:
        """Standardize where DB lives."""
        return work_dir / 'corpus_index' / f'{corpus_path.stem}.{id_col}.sqlite3'

    def set_shard_boundaries(self, total_shards: int, boundaries: list[tuple[int, int, int]]) -> None:
        """Store shard boundaries."""
        conn = self._get_conn()
        conn.executemany(
            'INSERT OR REPLACE INTO shards (total_shards, shard_id, start_byte_offset, end_byte_offset) VALUES (?, ?, ?, ?)',
            [(total_shards, s_id, start, end) for s_id, start, end in boundaries]
        )
        conn.commit()

    def get_shard_boundary(self, total_shards: int, shard_id: int) -> tuple[int, int]:
        """Get shard boundary. Raise ValueError if missing."""
        conn = self._get_conn()
        row = conn.execute(
            'SELECT start_byte_offset, end_byte_offset FROM shards WHERE total_shards = ? AND shard_id = ?',
            (total_shards, shard_id)
        ).fetchone()
        if row is None:
            raise ValueError(f'No shard boundaries found for total_shards={total_shards}, shard_id={shard_id}')
        return row['start_byte_offset'], row['end_byte_offset']

    def has_shard_boundaries(self, total_shards: int) -> bool:
        """Check if boundaries exist for given total_shards."""
        conn = self._get_conn()
        row = conn.execute(
            'SELECT 1 FROM shards WHERE total_shards = ? LIMIT 1',
            (total_shards,)
        ).fetchone()
        return row is not None

    def build_from_jsonl(
            self,
            corpus_path: Path,
            id_col: str,
            text_col: str,
            encoding: str = 'utf8',
            batch_size: int = 50000,
            on_duplicate_id: str = 'raise'
    ) -> int:
        """Build index from JSONL file."""
        return self._build_from_file(corpus_path, id_col, text_col, 'jsonl', encoding, batch_size, on_duplicate_id)

    def build_from_csv(
            self,
            corpus_path: Path,
            id_col: str,
            text_col: str,
            encoding: str = 'utf8',
            batch_size: int = 50000,
            on_duplicate_id: str = 'raise'
    ) -> int:
        """Build index from CSV file."""
        return self._build_from_file(corpus_path, id_col, text_col, 'csv', encoding, batch_size, on_duplicate_id)

    def _build_from_file(
            self,
            corpus_path: Path,
            id_col: str,
            text_col: str,
            kind: str,
            encoding: str = 'utf8',
            batch_size: int = 50000,
            on_duplicate_id: str = 'raise'
    ) -> int:
        if not corpus_path.exists():
            raise FileNotFoundError(f'Corpus not found: {corpus_path}')

        conn = self._get_conn()
        # clear existing data if any (or rely on cli's force_rebuild?)
        conn.execute('DELETE FROM meta')
        conn.execute('DELETE FROM doc_offsets')

        stats = corpus_path.stat()

        count = 0
        batch = []

        if kind == 'jsonl':
            with open(corpus_path, 'rb') as f:
                offset = 0
                for line in f:
                    line_len = len(line)

                    if line.strip():
                        try:
                            obj = json.loads(line.decode(encoding))
                            doc_id = str(obj[id_col])
                            batch.append((doc_id, offset))
                            count += 1
                        except Exception as e:
                            raise ValueError(f'Error parsing JSONL at offset {offset}: {e}. Ensure it is valid JSONL.')

                    offset += line_len

                    if len(batch) >= batch_size:
                        self._insert_batch(batch, on_duplicate_id)
                        batch = []
                        logger.info(f'Indexed {count:,} documents...')
        elif kind == 'csv':
            import csv
            import io
            # assumes no multi-line fields in csv
            with open(corpus_path, 'rb') as f:
                header_line = f.readline()
                offset = len(header_line)

                header_reader = csv.reader(io.TextIOWrapper(io.BytesIO(header_line), encoding=encoding, newline=''))
                try:
                    header = next(header_reader)
                except StopIteration:
                    raise ValueError('Empty CSV file')

                try:
                    id_idx = header.index(id_col)
                except ValueError:
                    raise ValueError(f"ID column '{id_col}' not found in CSV header: {header}")

                for line in f:
                    line_len = len(line)
                    if line.strip():
                        try:
                            # Parse single line as CSV
                            row_reader = csv.reader(io.TextIOWrapper(io.BytesIO(line), encoding=encoding, newline=''))
                            row = next(row_reader)
                            doc_id = str(row[id_idx])
                            batch.append((doc_id, offset))
                            count += 1
                        except Exception as e:
                            logger.warning(f'Error parsing CSV at offset {offset}: {e}')

                    offset += line_len

                    if len(batch) >= batch_size:
                        self._insert_batch(batch, on_duplicate_id)
                        batch = []
                        logger.info(f'Indexed {count:,} documents...')

        if batch:
            self._insert_batch(batch, on_duplicate_id)

        meta = {
            'corpus_path': str(corpus_path.absolute()),
            'corpus_size_bytes': str(stats.st_size),
            'corpus_mtime_ns': str(stats.st_mtime_ns),
            'id_col': id_col,
            'text_col': text_col,
        }
        conn.executemany('INSERT INTO meta (key, value) VALUES (?, ?)', list(meta.items()))
        conn.commit()

        return count

    def _insert_batch(self, batch: list, on_duplicate_id: str = 'raise'):
        conn = self._get_conn()
        try:
            conn.executemany('INSERT INTO doc_offsets (doc_id, byte_offset) VALUES (?, ?)', batch)
        except sqlite3.IntegrityError as e:
            if 'UNIQUE constraint failed: doc_offsets.doc_id' in str(e):
                if on_duplicate_id == 'skip':
                    # insert one by one, skipping duplicates
                    for doc_id, offset in batch:
                        try:
                            conn.execute('INSERT INTO doc_offsets (doc_id, byte_offset) VALUES (?, ?)', (doc_id, offset))
                        except sqlite3.IntegrityError:
                            logger.warning(f'Skipping duplicate doc_id: {doc_id}')
                            continue
                    return

                # find which one is duplicate
                ids = [b[0] for b in batch]
                seen = set()
                for i in ids:
                    if i in seen:
                        raise ValueError(f'Duplicate doc_id encountered: {i}')
                    seen.add(i)
                # if not in batch, it's already in DB (shouldn't happen with DELETE FROM)
                raise ValueError(f'Duplicate doc_id encountered in corpus: {e}')
            raise e

    def get_offset(self, doc_id: str) -> Optional[int]:
        conn = self._get_conn()
        row = conn.execute('SELECT byte_offset FROM doc_offsets WHERE doc_id = ?', (doc_id,)).fetchone()
        return row['byte_offset'] if row else None

    def get_doc_by_offset(self, corpus_path: Path, byte_offset: int, encoding: str = 'utf8') -> Dict[str, Any]:
        with open(corpus_path, 'rb') as f:
            f.seek(byte_offset)
            line = f.readline()
            decoded = line.decode(encoding)
            if decoded.strip().startswith('{'):
                return json.loads(decoded)
            else:
                # assume CSV - need the header to return a dict
                import csv
                import io
                f.seek(0)
                header_line = f.readline().decode(encoding)
                header_reader = csv.reader(io.StringIO(header_line))
                header = next(header_reader)

                row_reader = csv.reader(io.StringIO(decoded))
                row = next(row_reader)
                return dict(zip(header, row))

    def get_doc_by_id(self, corpus_path: Path, doc_id: str, encoding: str = 'utf8') -> Dict[str, Any]:
        offset = self.get_offset(doc_id)
        if offset is None:
            raise KeyError(f'doc_id {doc_id} not found in index')
        return self.get_doc_by_offset(corpus_path, offset, encoding)

    def get_text_span_by_offset(
            self,
            corpus_path: Path,
            byte_offset: int,
            start: int,
            end: int,
            text_col: str,
            encoding: str = 'utf8'
    ) -> str:
        doc = self.get_doc_by_offset(corpus_path, byte_offset, encoding)
        return doc[text_col][start:end]

    def validate_corpus(self, corpus_path: Path) -> None:
        """Compare current size + mtime_ns against meta."""
        conn = self._get_conn()
        meta = dict(conn.execute('SELECT key, value FROM meta').fetchall())

        if not meta:
            raise ValueError('Corpus index is empty or missing metadata.')

        stats = corpus_path.stat()

        if str(stats.st_size) != meta['corpus_size_bytes']:
            raise ValueError(
                f'Corpus size mismatch: current={stats.st_size}, indexed={meta["corpus_size_bytes"]}. '
                'Run index-corpus again.'
            )

        if str(stats.st_mtime_ns) != meta['corpus_mtime_ns']:
            # some filesystems might have issues
            raise ValueError(
                f'Corpus mtime mismatch. Run index-corpus again.'
            )

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
