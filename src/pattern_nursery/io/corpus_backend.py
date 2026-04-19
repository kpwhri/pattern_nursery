import csv
import json
import sqlite3
from pathlib import Path
from typing import Iterator, Tuple, Optional

from pattern_nursery.io.corpus_index import CorpusIndexSqlite


class CorpusBackend:
    """Base class for corpus backends."""

    def validate(self) -> None:
        """Validate the corpus."""
        raise NotImplementedError()

    def get_text(self, doc_id: str) -> str:
        """Get full text for a document ID."""
        raise NotImplementedError()

    def iter_docs(self) -> Iterator[Tuple[str, str]]:
        """Iterate over all documents (id, text)."""
        raise NotImplementedError()

    def iter_docs_shard(self, total_shards: int, shard_id: int) -> Iterator[Tuple[str, Optional[int], str]]:
        """Iterate over documents in a specific shard. Returns (doc_id, byte_offset, text)."""
        raise NotImplementedError()


class JsonlCorpusBackend(CorpusBackend):
    """Corpus backend for JSONL files with a SQLite offset index."""

    def __init__(self, corpus_path: Path, index_db_path: Path, id_col: str, text_col: str):
        self.corpus_path = corpus_path
        self.index_db_path = index_db_path
        self.id_col = id_col
        self.text_col = text_col
        self._index: Optional[CorpusIndexSqlite] = None

    @property
    def index(self) -> CorpusIndexSqlite:
        if self._index is None:
            self._index = CorpusIndexSqlite(self.index_db_path)
        return self._index

    def validate(self) -> None:
        self.index.validate_corpus(self.corpus_path)

    def get_text(self, doc_id: str) -> str:
        # get_doc_by_id returns the whole dict
        doc = self.index.get_doc_by_id(self.corpus_path, doc_id)
        return doc[self.text_col]

    def iter_docs(self) -> Iterator[Tuple[str, str]]:
        with open(self.corpus_path, 'r', encoding='utf8') as f:
            for line in f:
                doc = json.loads(line)
                yield str(doc[self.id_col]), doc[self.text_col]

    def iter_docs_shard(self, total_shards: int, shard_id: int) -> Iterator[Tuple[str, Optional[int], str]]:
        if not self.index.has_shard_boundaries(total_shards):
            raise ValueError(f'Shard boundaries missing for total_shards={total_shards}')

        start_offset, end_offset = self.index.get_shard_boundary(total_shards, shard_id)

        with open(self.corpus_path, 'rb') as f:
            f.seek(start_offset)
            while f.tell() < end_offset:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    doc = json.loads(line.decode('utf8'))
                    yield str(doc[self.id_col]), offset, doc[self.text_col]
                except Exception:
                    continue


class CsvCorpusBackend(CorpusBackend):
    """Corpus backend for CSV files.

    Supports two modes:
    - Legacy: delegates random access and sharding to a companion SQLite corpus (sqlite_db_path + table_name).
    - Index: uses a CorpusIndexSqlite DB with byte offsets (index_db_path) for random access/sharding.
    """

    def __init__(
            self,
            csv_path: Path,
            sqlite_db_path: Optional[Path] = None,
            table_name: Optional[str] = None,
            id_col: str = 'id',
            text_col: str = 'text',
            index_db_path: Optional[Path] = None
    ):
        self.csv_path = csv_path
        self.id_col = id_col
        self.text_col = text_col
        self._sqlite_backend: Optional[SqliteCorpusBackend] = None
        self._index: Optional[CorpusIndexSqlite] = None

        if sqlite_db_path is not None and table_name is not None:
            # legacy mode
            self._sqlite_backend = SqliteCorpusBackend(sqlite_db_path, table_name, id_col, text_col)
        elif index_db_path is not None:
            # index mode
            self._index_path = index_db_path
        else:
            # stream-only mode
            self._index_path = None

    @property
    def index(self) -> Optional[CorpusIndexSqlite]:
        if self._index is None and hasattr(self, '_index_path') and self._index_path is not None:
            self._index = CorpusIndexSqlite(self._index_path)
        return self._index

    def validate(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f'CSV corpus not found: {self.csv_path}')
        if self._sqlite_backend is not None:
            self._sqlite_backend.validate()
        elif self.index is not None:
            self.index.validate_corpus(self.csv_path)

    def get_text(self, doc_id: str) -> str:
        if self._sqlite_backend is not None:
            return self._sqlite_backend.get_text(doc_id)
        if self.index is not None:
            doc = self.index.get_doc_by_id(self.csv_path, doc_id)
            return doc[self.text_col]
        # fallback: stream and find
        with open(self.csv_path, 'r', encoding='utf8', newline='') as f:
            reader_obj = csv.DictReader(f)
            for row in reader_obj:
                if str(row[self.id_col]) == str(doc_id):
                    return row[self.text_col]
        raise KeyError(f'doc_id {doc_id} not found')

    def iter_docs(self) -> Iterator[Tuple[str, str]]:
        with open(self.csv_path, 'r', encoding='utf8', newline='') as f:
            reader_obj = csv.DictReader(f)
            for row in reader_obj:
                yield str(row[self.id_col]), row[self.text_col]

    def iter_docs_shard(self, total_shards: int, shard_id: int) -> Iterator[Tuple[str, Optional[int], str]]:
        if self._sqlite_backend is not None:
            yield from self._sqlite_backend.iter_docs_shard(total_shards, shard_id)
            return

        if self.index is None or not self.index.has_shard_boundaries(total_shards):
            raise ValueError(f'Shard boundaries missing for total_shards={total_shards}')

        start_offset, end_offset = self.index.get_shard_boundary(total_shards, shard_id)

        with open(self.csv_path, 'rb') as f:
            if start_offset == 0:
                # skip header
                f.readline()
            else:
                f.seek(start_offset)

            import io
            # To know the column indices, we read the header
            curr_pos = f.tell()
            f.seek(0)
            header_line = f.readline().decode('utf8')
            header_reader = csv.reader(io.StringIO(header_line))
            header = next(header_reader)
            id_idx = header.index(self.id_col)
            text_idx = header.index(self.text_col)
            f.seek(curr_pos)

            while f.tell() < end_offset:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    row_reader = csv.reader(io.TextIOWrapper(io.BytesIO(line), encoding='utf8', newline=''))
                    row = next(row_reader)
                    yield str(row[id_idx]), offset, row[text_idx]
                except Exception:
                    continue


class SqliteCorpusBackend(CorpusBackend):
    """Corpus backend for SQLite databases."""

    def __init__(self, db_path: Path, table_name: str, id_col: str, text_col: str):
        self.db_path = db_path
        self.table_name = table_name
        self.id_col = id_col
        self.text_col = text_col

    def validate(self) -> None:
        if not self.db_path.exists():
            raise FileNotFoundError(f'SQLite corpus not found: {self.db_path}')

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # check table
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (self.table_name,))
            if not cursor.fetchone():
                raise ValueError(f"Table '{self.table_name}' not found in {self.db_path}")

            # check columns
            cursor.execute(f'SELECT * FROM {self.table_name} LIMIT 0')
            cols = [description[0] for description in cursor.description]
            if self.id_col not in cols:
                raise ValueError(f"ID column '{self.id_col}' not found in table '{self.table_name}'")
            if self.text_col not in cols:
                raise ValueError(f"Text column '{self.text_col}' not found in table '{self.table_name}'")
        finally:
            conn.close()

    def get_text(self, doc_id: str) -> str:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(f'SELECT {self.text_col} FROM {self.table_name} WHERE {self.id_col} = ?', (doc_id,))
            row = cursor.fetchone()
            if not row:
                raise KeyError(f'Document not found: {doc_id}')
            return row[0]
        finally:
            conn.close()

    def iter_docs(self) -> Iterator[Tuple[str, str]]:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # use ORDER BY for stability
            cursor.execute(f'SELECT {self.id_col}, {self.text_col} FROM {self.table_name} ORDER BY {self.id_col}')
            while True:
                rows = cursor.fetchmany(1000)
                if not rows:
                    break
                for row in rows:
                    yield str(row[0]), row[1]
        finally:
            conn.close()

    def iter_docs_shard(self, total_shards: int, shard_id: int) -> Iterator[Tuple[str, Optional[int], str]]:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # check if shard_id column exists
            cursor.execute(f'PRAGMA table_info({self.table_name})')
            columns = [row[1] for row in cursor.fetchall()]
            if 'shard_id' not in columns:
                raise ValueError(f'SQLite corpus missing shard_id; run index-corpus --total-shards {total_shards}')

            cursor.execute(
                f'SELECT {self.id_col}, {self.text_col} FROM {self.table_name} WHERE shard_id = ? ORDER BY {self.id_col}',
                (shard_id,)
            )
            while True:
                rows = cursor.fetchmany(1000)
                if not rows:
                    break
                for row in rows:
                    yield str(row[0]), None, row[1]
        finally:
            conn.close()
