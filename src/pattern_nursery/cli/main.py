import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import typer
from loguru import logger

from pattern_nursery import logging_utils
from pattern_nursery.cli import profiles
from pattern_nursery.embeddings.fasttext_backend import FastTextBackend
from pattern_nursery.embeddings.sentence_transformers_backend import SentenceTransformersBackend
from pattern_nursery.embeddings.sparse.bm25_backend import BM25Backend
from pattern_nursery.embeddings.vectorstores.faiss_store import FaissStore
from pattern_nursery.embeddings.word2vec_backend import Word2VecBackend
from pattern_nursery.expansion import term_guided, regex_guided_sentences, model_guided
from pattern_nursery.index import paths
from pattern_nursery.index.manager import IndexManager
from pattern_nursery.io import reader
from pattern_nursery.io.corpus_backend import JsonlCorpusBackend, SqliteCorpusBackend, CorpusBackend, CsvCorpusBackend
from pattern_nursery.io.corpus_index import CorpusIndexSqlite
from pattern_nursery.io.corpus_types import detect_corpus_kind
from pattern_nursery.io.sharding import stable_shard_id
from pattern_nursery.io.snippets import resolve_text_for_result
from pattern_nursery.nlp.nltk_engine import NLTKEngine
from pattern_nursery.nlp.regex_engine import RegexEngine
from pattern_nursery.nlp.spacy_engine import SpaCyEngine
from pattern_nursery.reports import html as reports_html

app = typer.Typer(
    help='Hybrid clinical NLP library using embeddings to augment keyword/regex methods',
    pretty_exceptions_enable=os.environ.get('TYPER_PRETTY_EXCEPTIONS', 'true').lower() == 'true'
)


def get_nlp_engine(engine_name: str):
    if engine_name == 'regex':
        return RegexEngine()
    elif engine_name == 'nltk':
        return NLTKEngine()
    elif engine_name == 'spacy':
        return SpaCyEngine()
    raise ValueError(f'Unknown engine: {engine_name}')


def get_embedding_backend(backend_name: str, model_path: str):
    if backend_name == 'sbert':
        return SentenceTransformersBackend(model_path)
    elif backend_name == 'fasttext':
        return FastTextBackend(model_path)
    elif backend_name == 'word2vec':
        return Word2VecBackend(model_path)
    elif backend_name == 'bm25':
        return BM25Backend()
    elif backend_name == 'mock':
        # only for testing
        from pattern_nursery.embeddings.mock_backend import MockBackend
        return MockBackend(model_path)
    raise ValueError(f'Unknown backend: {backend_name}')


def setup_logging(work_dir: Path, run_name: str, level: str = 'INFO'):
    log_file = paths.get_run_logs_path(work_dir, run_name) / 'app.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)

    log_format = '{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | pid={process} | {extra} | {message}'

    logger.add(
        str(log_file),
        level=level,
        format=log_format,
        rotation='10 MB',
        enqueue=True,
        backtrace=True,
        diagnose=False
    )


def save_run_config(work_dir: Path, run_name: str, config: dict):
    """Save run configuration to run directory."""
    config_path = paths.get_run_dir(work_dir, run_name) / 'run_config.json'
    # load existing if any
    if config_path.exists():
        with open(config_path, 'r', encoding='utf8') as f:
            existing = json.load(f)
            existing.update(config)
            config = existing

    with open(config_path, 'w', encoding='utf8') as f:
        json.dump(config, f, indent=2)


def load_run_config(work_dir: Path, run_name: Optional[str]) -> dict:
    """Load run configuration from run directory."""
    if not run_name:
        return {}
    config_path = paths.get_run_dir(work_dir, run_name) / 'run_config.json'
    if config_path.exists():
        with open(config_path, 'r', encoding='utf8') as f:
            return json.load(f)
    return {}


def get_effective_params(
        work_dir: Optional[Path],
        run_name: Optional[str],
        model_path: Optional[str] = None,
        backend: Optional[str] = None,
        profile: str = 'default'
):
    """Get parameters from args or saved config."""
    # 1. resolve work_dir and run_name from profile if not provided
    prof = profiles.get_profile(profile)

    eff_work_dir = work_dir or (Path(prof['work_dir']) if prof.get('work_dir') else Path('runs'))
    eff_run_name = run_name or prof.get('run_name')

    # 2. load run config from resolved run directory
    config = load_run_config(eff_work_dir, eff_run_name)

    # 3. handle run_name if still None
    if not eff_run_name:
        eff_run_name = config.get('run_name')
    if not eff_run_name:
        eff_run_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # 4. handle model_path and backend
    eff_model_path = model_path or config.get('model_path')
    eff_backend = backend or config.get('backend')

    return eff_work_dir, eff_run_name, eff_model_path, eff_backend


def log_run_info(work_dir: Path, run_name: str, model_path: Optional[str]):
    logger.info(f'Run Context: work_dir={work_dir}, run_name={run_name}')
    if model_path:
        logger.info(f'Model Path: {model_path}')


def reconstruct_text_if_needed(meta: Dict[str, Any], work_dir: Path, input_path: Optional[Path], id_col: str,
                               text_col: str) -> str:
    """Fetch text from corpus if it's missing but we have doc_byte_offset."""
    if 'text' in meta:
        return meta['text']

    if 'doc_byte_offset' in meta and input_path and input_path.exists():
        db_path = CorpusIndexSqlite.get_index_path(work_dir, input_path, id_col)
        if db_path.exists():
            index_db = CorpusIndexSqlite(db_path)
            try:
                # if we have start_char and end_char, we can be more precise
                if 'start_char' in meta and 'end_char' in meta:
                    return index_db.get_text_span_by_offset(
                        input_path, meta['doc_byte_offset'], meta['start_char'], meta['end_char'], text_col
                    )
                else:
                    doc = index_db.get_doc_by_offset(input_path, meta['doc_byte_offset'])
                    return doc.get(text_col, '')
            except Exception as e:
                logger.warning(f'Failed to reconstruct text: {e}')

    return ''


@app.command()
def init_index(
        input: Path = typer.Option(..., help='Path to corpus (jsonl, sqlite, csv, parquet)'),
        work_dir: Optional[Path] = typer.Option(None, help='Work directory'),
        run_name: Optional[str] = typer.Option(None, help='Run name (default: timestamp)'),
        profile: str = typer.Option('default', help='Profile name for saving defaults'),
        id_col: str = typer.Option('id', help='ID column name'),
        text_col: str = typer.Option('text', help='Text column name'),
        table_name: str = typer.Option('documents', help='SQLite table name (SQLite corpora only)'),
        engine: str = typer.Option('regex', help='NLP engine'),
        collection: List[str] = typer.Option(['vocab_terms', 'sentences', 'notes', 'paragraphs', 'phrases'],
                                             help='Collections to index'),
        phrase_types: List[str] = typer.Option(['noun_chunks', 'bigrams'], help='Types of phrases to index'),
        backend: str = typer.Option('sbert', help='Embedding backend'),
        model_path: str = typer.Option(..., help='Path to model'),
        vector_store: str = typer.Option('faiss', help='Vector store type'),
        paragraph_mode: str = typer.Option('sentences', help='Paragraph mode'),
        window_size: int = typer.Option(3, help='Sentence window size'),
        overlap: int = typer.Option(1, help='Sentence window overlap'),
        token_window: int = typer.Option(50, help='Token window size'),
        token_overlap: int = typer.Option(10, help='Token window overlap'),
        batch_size: int = typer.Option(10000, help='Batch size for indexing (documents)'),
        num_workers: int = typer.Option(1, help='Number of worker processes for indexing'),
        log_level: str = typer.Option('INFO', help='Log level'),
        on_duplicate_id: str = typer.Option('raise', help='Action on duplicate doc_id: raise or skip')
):
    work_dir, run_name, _, _ = get_effective_params(work_dir, run_name, profile=profile)

    setup_logging(work_dir, run_name, log_level)
    local_logger = logger.bind(work_dir=str(work_dir), run_name=run_name, command='init_index')
    logging_utils.log_command_start(
        local_logger, 'init_index',
        engine=engine, backend=backend, model_path=model_path,
        collections=collection, batch_size=batch_size, num_workers=num_workers
    )

    t0 = time.perf_counter()
    # save config for future use
    save_run_config(work_dir, run_name, {
        'model_path': model_path,
        'backend': backend,
        'run_name': run_name,
        'engine': engine,
        'num_workers': num_workers,
        'batch_size': batch_size,
        'input': str(input.absolute()),
        'id_col': id_col,
        'text_col': text_col,
        'table_name': table_name
    })

    # save to profile
    profiles.set_profile(profile, work_dir, run_name)

    logger.info(f'Parameters: engine={engine}, backend={backend}, model={model_path}, collections={collection}')

    nlp_engine = get_nlp_engine(engine)
    embed_backend = get_embedding_backend(backend, model_path)
    manager = IndexManager(work_dir, run_name, vector_store, nlp_engine)

    # detect corpus kind
    try:
        kind = detect_corpus_kind(input)
    except ValueError:
        kind = 'other'

    corpus_index_db = None
    if kind == 'jsonl':
        db_path = CorpusIndexSqlite.get_index_path(work_dir, input, id_col)
        if db_path.exists():
            index_db = CorpusIndexSqlite(db_path)
            try:
                index_db.validate_corpus(input)
                logger.info('Corpus index validated for JSONL reference storage.')
                corpus_index_db = str(db_path.relative_to(work_dir))

                def get_it():
                    return reader.corpus_iter_jsonl_with_offsets(input, id_col, text_col)
            except ValueError as e:
                logger.warning(f'Corpus validation failed: {e}. Falling back to full-text storage.')

                def get_it():
                    return reader.corpus_iter(input, id_col, text_col)
        else:
            logger.info('Corpus index missing. Falling back to full-text storage.')

            def get_it():
                return reader.corpus_iter(input, id_col, text_col)
    elif kind == 'sqlite':
        logger.info(f'Using SQLite corpus backend for table {table_name}')

        def get_it():
            backend = SqliteCorpusBackend(input, table_name, id_col, text_col)
            return backend.iter_docs()
    elif kind == 'csv':
        logger.info('Using CSV corpus backend')
        db_path = CorpusIndexSqlite.get_index_path(work_dir, input, id_col)
        if db_path.exists():
            index_db = CorpusIndexSqlite(db_path)
            try:
                index_db.validate_corpus(input)
                logger.info('Corpus index validated for CSV reference storage.')
                corpus_index_db = str(db_path.relative_to(work_dir))

                def get_it():
                    return reader.corpus_iter(input, id_col, text_col)
            except ValueError as e:
                logger.warning(f'Corpus validation failed: {e}. Falling back to full-text storage.')

                def get_it():
                    return reader.corpus_iter(input, id_col, text_col)
        else:
            logger.info('Corpus index missing for CSV. Falling back to full-text storage.')

            def get_it():
                return reader.corpus_iter(input, id_col, text_col)
    else:
        def get_it():
            return reader.corpus_iter(input, id_col, text_col)

    manager.set_corpus_info(
        corpus_path=input,
        corpus_kind=kind,
        id_col=id_col,
        text_col=text_col,
        table_name=table_name if kind == 'sqlite' else None,
        corpus_index_db=corpus_index_db
    )

    if 'vocab_terms' in collection:
        if manager.exists('vocab_terms'):
            logger.info('Vocabulary index already exists. Skipping...')
        else:
            logger.info('Building vocabulary...')
            it = reader.corpus_iter(input, id_col, text_col)
            vocab_list = manager.build_and_persist_vocabulary(it)

            # Create (term_id, term_text) iterator for embed_and_store
            vocab_iter = ((f'v_{i}', term) for i, term in enumerate(vocab_list))

            logger.info(f'Embedding {len(vocab_list)} vocabulary terms...')
            manager.embed_and_store(vocab_iter, embed_backend, 'vocab_terms', batch_size=batch_size,
                                    num_workers=num_workers)

    if 'sentences' in collection:
        if manager.exists('sentences'):
            logger.info('Sentences index already exists. Skipping...')
        else:
            logger.info('Indexing sentences...')
            it = get_it()
            manager.index_sentences(it, embed_backend, batch_size=batch_size, num_workers=num_workers)

    if 'notes' in collection:
        if manager.exists('notes'):
            logger.info('Notes index already exists. Skipping...')
        else:
            logger.info('Indexing notes...')
            it = get_it()
            manager.index_notes(it, embed_backend, batch_size=batch_size, num_workers=num_workers)
            # Make sure we persist it after full indexing
            store_path = paths.get_global_vectorstore_path(work_dir, run_name, 'notes')
            if (store_path / 'index.faiss').exists():
                logger.info(f'Persisted notes index to {store_path}')
            else:
                # In some cases (single batch) it might already be persisted by _add_to_store
                pass

    if 'paragraphs' in collection:
        if manager.exists('paragraphs'):
            logger.info('Paragraphs index already exists. Skipping...')
        else:
            logger.info(f'Indexing paragraphs (mode={paragraph_mode})...')
            it = get_it()
            manager.index_paragraphs(
                it, embed_backend, mode=paragraph_mode,
                window_size=window_size, overlap=overlap,
                token_window=token_window, token_overlap=token_overlap,
                batch_size=batch_size, num_workers=num_workers
            )

    if 'phrases' in collection:
        if manager.exists('phrases'):
            logger.info('Phrases index already exists. Skipping...')
        else:
            logger.info(f'Indexing phrases (types={phrase_types})...')
            it = get_it()
            manager.index_phrases(it, embed_backend, types=phrase_types, batch_size=batch_size, num_workers=num_workers)

    elapsed = time.perf_counter() - t0
    logging_utils.log_command_done(local_logger, 'init_index', elapsed)
    logger.info('Index initialization complete.')


@app.command()
def index_corpus(
        input: Path = typer.Option(..., help='Path to corpus (jsonl, csv, sqlite)'),
        work_dir: Path = typer.Option(..., help='Work directory'),
        id_col: str = typer.Option('id', help='ID column name'),
        text_col: str = typer.Option('text', help='Text column name'),
        table_name: str = typer.Option('documents', help='SQLite table name (SQLite corpora only)'),
        total_shards: Optional[int] = typer.Option(None, help='Compute and store record-aligned shard boundaries'),
        force_rebuild: bool = typer.Option(False, help='Drop and recreate index/shard info'),
        force_rebuild_shards: bool = typer.Option(False, help='Recompute shard_id for SQLite corpora'),
        log_level: str = typer.Option('INFO', help='Log level'),
        on_duplicate_id: str = typer.Option('raise', help='Action on duplicate doc_id: raise or skip')
):
    """Build a SQLite mapping doc_id -> byte_offset for the JSONL file OR validate/shard SQLite corpus."""
    setup_logging(work_dir, 'corpus_indexing', log_level)

    local_logger = logger.bind(work_dir=str(work_dir), run_name='corpus_indexing', command='index_corpus')
    logging_utils.log_command_start(
        local_logger, 'index_corpus',
        id_col=id_col, text_col=text_col, table_name=table_name,
        total_shards=total_shards, force_rebuild=force_rebuild, force_rebuild_shards=force_rebuild_shards
    )
    t0 = time.perf_counter()

    try:
        kind = detect_corpus_kind(input)
    except ValueError as e:
        logger.error(str(e))
        raise typer.Exit(code=1)

    if kind == 'jsonl':
        _index_corpus_jsonl(input, work_dir, id_col, text_col, total_shards, force_rebuild, on_duplicate_id)
    elif kind == 'csv':
        _index_corpus_csv(input, work_dir, id_col, text_col, total_shards, force_rebuild, on_duplicate_id)
    elif kind == 'sqlite':
        _index_corpus_sqlite(input, table_name, id_col, text_col, total_shards, force_rebuild_shards)

    elapsed = time.perf_counter() - t0
    logging_utils.log_command_done(local_logger, 'index_corpus', elapsed)


def _index_corpus_jsonl(input_path, work_dir, id_col, text_col, total_shards, force_rebuild, on_duplicate_id):
    db_path = CorpusIndexSqlite.get_index_path(work_dir, input_path, id_col)
    index = CorpusIndexSqlite(db_path)

    skip_build = False
    if not force_rebuild and db_path.exists():
        try:
            index.validate_corpus(input_path)
            logger.info('Corpus index matches corpus.')
            skip_build = True
        except ValueError as e:
            logger.warning(f'Validation failed: {e}. Rebuilding...')

    if not skip_build:
        logger.info(f'Building corpus index for {input_path}...')
        count = index.build_from_jsonl(input_path, id_col, text_col, on_duplicate_id=on_duplicate_id)
        logger.info(f'Successfully indexed {count:,} documents.')

    if total_shards:
        if index.has_shard_boundaries(total_shards) and not force_rebuild and skip_build:
            logger.info(f'Shard boundaries for total_shards={total_shards} already exist. Skipping.')
        else:
            logger.info(f'Computing record-aligned boundaries for total_shards={total_shards}...')
            # pass 1: count lines
            doc_count = 0
            file_size = input_path.stat().st_size
            with open(input_path, 'rb') as f:
                for line in f:
                    if line.strip():
                        doc_count += 1

            logger.info(f'Total records: {doc_count}')

            # compute cutpoints
            import math
            cut_indices = [math.floor(k * doc_count / total_shards) for k in range(total_shards)]
            cut_indices.append(doc_count)

            # pass 2: find offsets for cutpoints
            cut_offsets = []
            current_doc_idx = 0
            next_cut_idx_ptr = 0

            with open(input_path, 'rb') as f:
                offset = 0
                while next_cut_idx_ptr < len(cut_indices):
                    target_idx = cut_indices[next_cut_idx_ptr]
                    if current_doc_idx == target_idx:
                        cut_offsets.append(offset)
                        next_cut_idx_ptr += 1
                        if next_cut_idx_ptr == len(cut_indices):
                            break

                    line = f.readline()
                    if not line:
                        # Should only happen at end of file
                        while next_cut_idx_ptr < len(cut_indices):
                            cut_offsets.append(offset)
                            next_cut_idx_ptr += 1
                        break

                    if line.strip():
                        current_doc_idx += 1
                    offset += len(line)

            # convert to boundaries
            boundaries = []
            for k in range(total_shards):
                start = cut_offsets[k]
                end = cut_offsets[k + 1] if k + 1 < len(cut_offsets) else file_size
                boundaries.append((k, start, end))

            # validate invariants
            assert boundaries[0][1] == 0
            assert boundaries[-1][2] == file_size
            for k in range(total_shards - 1):
                assert boundaries[k][2] == boundaries[k + 1][1]

            index.set_shard_boundaries(total_shards, boundaries)

            docs_per_shard = [cut_indices[k + 1] - cut_indices[k] for k in range(total_shards)]
            logger.info(
                f'Stored boundaries for {total_shards} shards. Docs per shard: min={min(docs_per_shard)}, max={max(docs_per_shard)}, mean={sum(docs_per_shard) / len(docs_per_shard):.1f}')


def _index_corpus_csv(input_corpus, work_dir, id_col, text_col, total_shards, force_rebuild, on_duplicate_id):
    db_path = CorpusIndexSqlite.get_index_path(work_dir, input_corpus, id_col)
    index = CorpusIndexSqlite(db_path)

    skip_build = False
    if not force_rebuild and db_path.exists():
        try:
            index.validate_corpus(input_corpus)
            logger.info('Corpus index matches corpus.')
            skip_build = True
        except ValueError as e:
            logger.warning(f'Validation failed: {e}. Rebuilding...')

    if not skip_build:
        logger.info(f'Building corpus index for {input_corpus}...')
        count = index.build_from_csv(input_corpus, id_col, text_col, on_duplicate_id=on_duplicate_id)
        logger.info(f'Successfully indexed {count:,} documents.')

    if total_shards:
        if index.has_shard_boundaries(total_shards) and not force_rebuild and skip_build:
            logger.info(f'Shard boundaries for total_shards={total_shards} already exist. Skipping.')
        else:
            logger.info(f'Computing record-aligned boundaries for total_shards={total_shards}...')
            # pass 1: count records
            doc_count = 0
            file_size = input_corpus.stat().st_size
            with open(input_corpus, 'rb') as f:
                # skip header
                f.readline()
                for line in f:
                    if line.strip():
                        doc_count += 1

            logger.info(f'Total records: {doc_count}')

            # compute cutpoints
            import math
            cut_indices = [math.floor(k * doc_count / total_shards) for k in range(total_shards)]
            cut_indices.append(doc_count)

            # pass 2: find offsets for cutpoints
            cut_offsets = []
            current_doc_idx = 0
            next_cut_idx_ptr = 0

            with open(input_corpus, 'rb') as f:
                offset = 0
                # header
                header_line = f.readline()

                while next_cut_idx_ptr < len(cut_indices):
                    target_idx = cut_indices[next_cut_idx_ptr]
                    if current_doc_idx == target_idx:
                        cut_offsets.append(offset)
                        next_cut_idx_ptr += 1
                        if next_cut_idx_ptr == len(cut_indices):
                            break

                    line = f.readline()
                    if not line:
                        while next_cut_idx_ptr < len(cut_indices):
                            cut_offsets.append(offset)
                            next_cut_idx_ptr += 1
                        break

                    if line.strip():
                        current_doc_idx += 1
                    offset += len(line)

            # convert to boundaries
            boundaries = []
            for k in range(total_shards):
                start = cut_offsets[k]
                end = cut_offsets[k + 1] if k + 1 < len(cut_offsets) else file_size
                boundaries.append((k, start, end))

            index.set_shard_boundaries(total_shards, boundaries)
            logger.info(f'Stored boundaries for {total_shards} shards.')


def _index_corpus_sqlite(db_path: Path, table_name: str, id_col: str, text_col: str, total_shards: Optional[int],
                         force_rebuild_shards: bool):
    import sqlite3

    backend = SqliteCorpusBackend(db_path, table_name, id_col, text_col)
    try:
        backend.validate()
    except ValueError as e:
        logger.error(f'SQLite corpus validation failed: {e}')
        raise typer.Exit(code=1)

    if total_shards is None:
        logger.info('SQLite corpus validated; shard_id not requested; skipping shard checks.')
        return

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        # 1. ensure shard_id column exists
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = {row[1]: row for row in cursor.fetchall()}
        if 'shard_id' not in columns:
            logger.info(f"Adding 'shard_id' column to table '{table_name}'")
            cursor.execute(f'ALTER TABLE {table_name} ADD COLUMN shard_id INTEGER')
            needs_compute = True
        else:
            # check if any NULLs
            cursor.execute(f'SELECT COUNT(*) FROM {table_name} WHERE shard_id IS NULL')
            null_count = cursor.fetchone()[0]
            needs_compute = null_count > 0

        # 2. check meta for total_shards mismatch
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='meta'")
        has_meta = cursor.fetchone() is not None
        if has_meta:
            cursor.execute("SELECT value FROM meta WHERE key='total_shards'")
            row = cursor.fetchone()
            if row:
                stored_shards = int(row[0])
                if stored_shards != total_shards:
                    if not force_rebuild_shards:
                        logger.error(
                            f'Total shards mismatch: requested {total_shards}, but corpus has {stored_shards}. Use --force-rebuild-shards to recompute.')
                        raise typer.Exit(code=1)
                    else:
                        logger.info(f'Forcing rebuild of shards (stored {stored_shards} -> requested {total_shards})')
                        needs_compute = True

        if force_rebuild_shards:
            needs_compute = True

        if needs_compute:
            logger.info(f'Computing shard_id for total_shards={total_shards}...')
            # fetch doc ids
            if force_rebuild_shards:
                cursor.execute(f'SELECT {id_col} FROM {table_name}')
            else:
                cursor.execute(f'SELECT {id_col} FROM {table_name} WHERE shard_id IS NULL')

            all_ids = [row[0] for row in cursor.fetchall()]
            logger.info(f'Processing {len(all_ids):,} documents...')

            updates = []
            for doc_id in all_ids:
                sid = stable_shard_id(str(doc_id), total_shards)
                updates.append((sid, doc_id))

            # batch update
            cursor.executemany(f'UPDATE {table_name} SET shard_id = ? WHERE {id_col} = ?', updates)

            # index
            cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_shard_id ON {table_name}(shard_id)')

            # meta
            if not has_meta:
                cursor.execute('CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)')

            meta_data = [
                ('total_shards', str(total_shards)),
                ('id_col', id_col),
                ('text_col', text_col),
                ('table_name', table_name),
                ('updated_at', datetime.now().isoformat())
            ]
            cursor.executemany('INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)', meta_data)

            conn.commit()
            logger.info('Shard assignment complete.')
        else:
            logger.info(f'SQLite corpus already sharded for total_shards={total_shards}.')

    finally:
        conn.close()


@app.command()
def init_index_shard(
        input: Path = typer.Option(..., help='Path to corpus (jsonl or sqlite)'),
        work_dir: Path = typer.Option(..., help='Work directory'),
        run_name: str = typer.Option(..., help='Run name'),
        profile: str = typer.Option('default', help='Profile name for saving defaults'),
        shard_id: int = typer.Option(0, help='Shard ID (0 to total-shards - 1)'),
        total_shards: int = typer.Option(1, help='Total number of shards'),
        collection: List[str] = typer.Option(['notes', 'sentences', 'paragraphs', 'phrases'],
                                             help='Collections to index'),
        engine: str = typer.Option('regex', help='NLP engine'),
        backend: str = typer.Option('sbert', help='Embedding backend'),
        model_path: str = typer.Option(..., help='Path to model'),
        id_col: str = typer.Option('id', help='ID column name'),
        text_col: str = typer.Option('text', help='Text column name'),
        table_name: str = typer.Option('documents', help='SQLite table name (SQLite corpora only)'),
        embed_batch_size: int = typer.Option(2048, help='Batch size for embedding'),
        sentences_nlist: int = typer.Option(2048, help='NLIST for sentences IVF'),
        sentences_nprobe: int = typer.Option(16, help='NPROBE for sentences IVF'),
        paragraphs_nlist: int = typer.Option(1024, help='NLIST for paragraphs IVF'),
        paragraphs_nprobe: int = typer.Option(16, help='NPROBE for paragraphs IVF'),
        max_phrases_per_sentence: int = typer.Option(5, help='Max phrases per sentence'),
        min_phrase_tokens: int = typer.Option(2, help='Min tokens per phrase'),
        log_level: str = typer.Option('INFO', help='Log level'),
):
    """Index one shard into FAISS shard artifacts."""
    work_dir, run_name, _, _ = get_effective_params(work_dir, run_name, profile=profile)
    setup_logging(work_dir, f'shard_{shard_id:05d}', log_level)
    local_logger = logger.bind(
        work_dir=str(work_dir), run_name=run_name, command='init_index_shard',
        shard_id=shard_id, total_shards=total_shards
    )
    logging_utils.log_command_start(
        local_logger, 'init_index_shard',
        shard_id=shard_id, total_shards=total_shards, engine=engine,
        backend=backend, model_path=model_path, collections=collection
    )
    t0 = time.perf_counter()
    profiles.set_profile(profile, work_dir, run_name)

    # detect corpus kind
    try:
        kind = detect_corpus_kind(input)
    except ValueError as e:
        logger.error(str(e))
        raise typer.Exit(code=1)

    corpus_index_db = None
    start_offset, end_offset = 0, 0
    if kind == 'jsonl':
        # precondition: corpus index must exist and match corpus
        db_path = CorpusIndexSqlite.get_index_path(work_dir, input, id_col)
        if not db_path.exists():
            logger.error(f'Corpus index missing: {db_path}. Run index-corpus first.')
            raise typer.Exit(code=1)

        index_db = CorpusIndexSqlite(db_path)
        try:
            index_db.validate_corpus(input)
        except ValueError as e:
            logger.error(f'Corpus validation failed: {e}')
            raise typer.Exit(code=1)

        if not index_db.has_shard_boundaries(total_shards):
            logger.error(
                f'Shard boundaries missing for total_shards={total_shards}. Run index-corpus --total-shards {total_shards}')
            raise typer.Exit(code=1)

        corpus_index_db = str(db_path.relative_to(work_dir))
        # create backend
        corpus_backend = JsonlCorpusBackend(input, db_path, id_col, text_col)
        start_offset, end_offset = index_db.get_shard_boundary(total_shards, shard_id)
        logger.info(f'Indexing JSONL shard {shard_id}/{total_shards} (byte range {start_offset}-{end_offset})')
    elif kind == 'sqlite':
        corpus_backend = SqliteCorpusBackend(input, table_name, id_col, text_col)
        try:
            corpus_backend.validate()
        except Exception as e:
            logger.error(f'SQLite corpus validation failed: {e}')
            raise typer.Exit(code=1)
        logger.info(f'Indexing SQLite shard {shard_id}/{total_shards}')
    elif kind == 'csv':
        # precondition: corpus index must exist and match corpus
        db_path = CorpusIndexSqlite.get_index_path(work_dir, input, id_col)
        if not db_path.exists():
            logger.error(f'Corpus index missing: {db_path}. Run index-corpus first.')
            raise typer.Exit(code=1)
        index_db = CorpusIndexSqlite(db_path)
        try:
            index_db.validate_corpus(input)
        except ValueError as e:
            logger.error(f'Corpus validation failed: {e}')
            raise typer.Exit(code=1)

        if not index_db.has_shard_boundaries(total_shards):
            logger.error(
                f'Shard boundaries missing for total_shards={total_shards}. Run index-corpus --total-shards {total_shards}')
            raise typer.Exit(code=1)

        corpus_index_db = str(db_path.relative_to(work_dir))

        corpus_backend = CsvCorpusBackend(csv_path=input, index_db_path=db_path, id_col=id_col, text_col=text_col)

        start_offset, end_offset = index_db.get_shard_boundary(total_shards, shard_id)
        logger.info(f'Indexing CSV shard {shard_id}/{total_shards} using SQLite backend')
        logger.info(f'Indexing CSV shard {shard_id}/{total_shards} (byte range {start_offset}-{end_offset})')
    else:
        logger.error(f'Unsupported corpus kind for sharded indexing: {kind}')
        raise typer.Exit(code=1)

    start_time = datetime.now()
    nlp_engine = get_nlp_engine(engine)
    embed_backend = get_embedding_backend(backend, model_path)

    manager = IndexManager(work_dir, run_name, 'faiss', nlp_engine)
    manager.set_corpus_info(
        corpus_path=input,
        corpus_kind=kind,
        id_col=id_col,
        text_col=text_col,
        table_name=table_name if kind == 'sqlite' else None,
        corpus_index_db=corpus_index_db
    )

    # Save run config
    save_run_config(work_dir, run_name, {
        'model_path': model_path,
        'backend': backend,
        'run_name': run_name,
        'engine': engine,
        'input': str(input.absolute()),
        'id_col': id_col,
        'text_col': text_col,
        'table_name': table_name,
        'corpus_kind': kind,
        'corpus_index_db': corpus_index_db
    })

    # stores for this shard
    stores = {}
    for coll in collection:
        dim = embed_backend.dimension
        if coll == 'sentences':
            stores[coll] = FaissStore(dimension=dim, index_type='ivfflatip', nlist=sentences_nlist)
        elif coll == 'paragraphs':
            stores[coll] = FaissStore(dimension=dim, index_type='ivfflatip', nlist=paragraphs_nlist)
        else:
            stores[coll] = FaissStore(dimension=dim, index_type='flatip')

    # read shard
    docs_seen = 0
    units_indexed = {coll: 0 for coll in collection}

    success = False
    error_msg = None
    try:
        # Buffers for batch processing
        buffers = {coll: {'texts': [], 'metas': []} for coll in collection}

        def flush_buffer(coll):
            if not buffers[coll]['texts']:
                return

            vecs = embed_backend.embed(buffers[coll]['texts'])

            if hasattr(stores[coll], 'train') and not stores[coll].index.is_trained:
                # train IVF if possible; otherwise downgrade to flat
                if hasattr(stores[coll].index, 'nlist') and len(vecs) < stores[coll].index.nlist:
                    dim = embed_backend.dimension
                    stores[coll] = FaissStore(dimension=dim, index_type='flatip')
                else:
                    stores[coll].train(vecs)

            stores[coll].add(vecs, buffers[coll]['metas'])
            units_indexed[coll] += len(vecs)
            buffers[coll]['texts'] = []
            buffers[coll]['metas'] = []

        for doc_id, current_line_offset, text in corpus_backend.iter_docs_shard(total_shards, shard_id):
            docs_seen += 1
            if docs_seen % 1000 == 0:
                logger.info(f'Processed {docs_seen:,} documents...')

            # 1. Notes
            if 'notes' in collection:
                meta = {'doc_id': doc_id}
                if current_line_offset is not None:
                    meta['doc_byte_offset'] = current_line_offset

                buffers['notes']['texts'].append(text)
                buffers['notes']['metas'].append(meta)
                if len(buffers['notes']['texts']) >= embed_batch_size:
                    flush_buffer('notes')

            # 2. Sentences
            if 'sentences' in collection:
                for sent_id, (sent_text, start, end) in enumerate(nlp_engine.split_sentences_with_spans(text)):
                    meta = {
                        'doc_id': doc_id,
                        'sent_id': sent_id,
                        'start_char': start,
                        'end_char': end
                    }
                    if current_line_offset is not None:
                        meta['doc_byte_offset'] = current_line_offset

                    buffers['sentences']['texts'].append(sent_text)
                    buffers['sentences']['metas'].append(meta)
                    if len(buffers['sentences']['texts']) >= embed_batch_size:
                        flush_buffer('sentences')

            # 3. Paragraphs
            if 'paragraphs' in collection:
                sents = list(nlp_engine.split_sentences_with_spans(text))
                window_size_local = 3
                overlap_local = 1
                step = max(1, window_size_local - overlap_local)
                para_id = 0
                for i in range(0, len(sents), step):
                    window = sents[i: i + window_size_local]
                    if not window: break

                    para_text = ' '.join(s[0] for s in window)
                    meta = {
                        'doc_id': doc_id,
                        'para_id': para_id,
                        'start_char': window[0][1],
                        'end_char': window[-1][2],
                        'start_sent_id': i,
                        'end_sent_id': i + len(window) - 1
                    }
                    if current_line_offset is not None:
                        meta['doc_byte_offset'] = current_line_offset

                    buffers['paragraphs']['texts'].append(para_text)
                    buffers['paragraphs']['metas'].append(meta)
                    if len(buffers['paragraphs']['texts']) >= embed_batch_size:
                        flush_buffer('paragraphs')
                    para_id += 1
                    if i + window_size_local >= len(sents):
                        break

            # 4. Phrases
            if 'phrases' in collection and hasattr(nlp_engine, 'extract_phrases'):
                phrases = nlp_engine.extract_phrases(text)
                for p_idx, p in enumerate(phrases):
                    meta = {
                        'doc_id': doc_id,
                        'phrase_id': p_idx,
                        'start_char': p[1],
                        'end_char': p[2],
                        'phrase_type': p[3]
                    }
                    if current_line_offset is not None:
                        meta['doc_byte_offset'] = current_line_offset

                    buffers['phrases']['texts'].append(p[0])
                    buffers['phrases']['metas'].append(meta)
                    if len(buffers['phrases']['texts']) >= embed_batch_size:
                        flush_buffer('phrases')

        for coll in collection:
            flush_buffer(coll)

        success = True
    except Exception as e:
        logger.exception(f'Error during indexing: {e}')
        error_msg = str(e)

    finished_at = datetime.now()
    elapsed = (finished_at - start_time).total_seconds()

    # try to load corpus meta for manifest
    meta = {}
    if (kind == 'jsonl' or kind == 'csv') and corpus_index_db:
        try:
            index_db = CorpusIndexSqlite(work_dir / corpus_index_db)
            conn = index_db._get_conn()
            meta_rows = conn.execute('SELECT key, value FROM meta').fetchall()
            meta = {row[0]: row[1] for row in meta_rows}
        except Exception:
            pass

    for coll in collection:
        shard_dir = paths.get_global_vectorstore_path(work_dir, run_name, coll) / 'shards' / f'shard_{shard_id:05d}'
        shard_dir.mkdir(parents=True, exist_ok=True)
        if stores.get(coll) is not None and stores[coll].count() > 0:
            stores[coll].persist(shard_dir)

            # Write mini config for the shard
            manager._write_config(coll, embed_backend.dimension, shard_dir, backend=embed_backend)

        # counts
        vectors_added = stores[coll].count() if stores.get(coll) is not None else 0
        units = units_indexed.get(coll, 0)

        # manifest per shard/collection
        dim = embed_backend.dimension
        manifest = {
            'status': 'done' if success else 'failed',
            'collection': coll,
            'store_type': 'faiss',
            'sharded': True,
            'total_shards': total_shards,
            'shard_id': shard_id,
            'corpus': {
                'path': str(input.absolute()),
                'kind': kind,
                'id_col': id_col,
                'text_col': text_col,
                'size_bytes': int(meta.get('corpus_size_bytes', input.stat().st_size)),
                'mtime_ns': int(meta.get('corpus_mtime_ns', input.stat().st_mtime_ns)),
            },
            'shard_boundary': {
                'start_byte_offset': start_offset,
                'end_byte_offset': end_offset,
                'start_is_line_boundary': True,
                'end_is_line_boundary': True,
            },
            'counts': {
                'docs_seen': docs_seen,
                'units_indexed': units,
                'vectors_added': vectors_added,
            },
            'performance': {
                'duration_sec': elapsed,
                'docs_per_sec': docs_seen / elapsed if elapsed > 0 else 0,
                'units_per_sec': units / elapsed if elapsed > 0 else 0,
            },
            'embedding': {
                'backend': backend,
                'model': model_path,
                'dim': dim,
                'dtype': 'float32',
                'normalized': True,
                'similarity': 'cosine_ip'
            },
            'faiss': {
                'index_type': 'ivfflatip' if coll in ['sentences', 'paragraphs'] else 'flatip',
                'metric': 'ip',
                'trained': getattr(stores[coll].index, 'is_trained', True) if stores.get(coll) is not None else False,
                'nlist': getattr(stores[coll].index, 'nlist', None) if stores.get(coll) is not None else None,
                'nprobe': sentences_nprobe if coll == 'sentences' else (
                    paragraphs_nprobe if coll == 'paragraphs' else None)
            },
            'nlp': {
                'engine': engine,
                'paragraph_mode': 'sentences',
                'window_size': 3,
                'overlap': 1,
                'token_window': None,
                'token_overlap': None,
                'phrase_types': ['noun_chunks', 'bigrams'],
                'max_phrases_per_sentence': max_phrases_per_sentence,
                'min_phrase_tokens': min_phrase_tokens
            },
            'timing': {
                'started_at': start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'finished_at': finished_at.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'elapsed_seconds': elapsed
            },
            'version': {
                'app_version': '0.1.0'
            },
            'timestamp': finished_at.isoformat(),
            'error': error_msg if not success else None
        }
        with open(shard_dir / 'manifest.json', 'w', encoding='utf8') as f_out:
            json.dump(manifest, f_out, indent=2)

    elapsed = time.perf_counter() - t0
    logging_utils.log_command_done(local_logger, 'init_index_shard', elapsed)
    logger.info(f'Shard {shard_id} complete in {elapsed:.1f}s. Indexed {docs_seen} docs.')
    if not success:
        raise typer.Exit(code=1)


@app.command()
def use_index(
        work_dir: Optional[Path] = typer.Option(None, help='Work directory'),
        run_name: Optional[str] = typer.Option(None, help='Run name'),
        profile: str = typer.Option('default', help='Profile name for saving defaults'),
        log_level: str = typer.Option('INFO', help='Log level')
):
    """Set session defaults for a profile."""
    work_dir, run_name, model_path, backend = get_effective_params(work_dir, run_name, profile=profile)

    local_logger = logger.bind(work_dir=str(work_dir), run_name=run_name, command='use_index')
    logging_utils.log_command_start(local_logger, 'use_index', profile=profile)
    t0 = time.perf_counter()

    # Persist to profile
    profiles.set_profile(profile, work_dir, run_name)

    setup_logging(work_dir, run_name, log_level)
    logger.info(f'Using profile "{profile}": work_dir={work_dir}, run_name={run_name}')
    if model_path:
        logger.info(f'Resolved model_path={model_path}, backend={backend}')

    elapsed = time.perf_counter() - t0
    logging_utils.log_command_done(local_logger, 'use_index', elapsed)


def _get_corpus_backend_from_config(config: Dict[str, Any], work_dir: Path) -> Optional[CorpusBackend]:
    """Reconstruct CorpusBackend from store config."""
    corpus_kind = config.get('corpus_kind')
    if not corpus_kind:
        return None

    corpus_path = Path(config['corpus_path'])
    id_col = config.get('id_col', 'id')
    text_col = config.get('text_col', 'text')

    if corpus_kind == 'jsonl':
        index_db_rel = config.get('corpus_index_db')
        if not index_db_rel:
            return None
        return JsonlCorpusBackend(corpus_path, work_dir / index_db_rel, id_col, text_col)
    elif corpus_kind == 'sqlite':
        table_name = config.get('table_name', 'documents')
        return SqliteCorpusBackend(corpus_path, table_name, id_col, text_col)
    elif corpus_kind == 'csv':
        # for csv, we require an accompanying index db with byte offsets
        index_db_rel = config.get('corpus_index_db')
        if not index_db_rel:
            return None
        return CsvCorpusBackend(corpus_path, work_dir / index_db_rel, id_col, text_col)

    return None


@app.command()
def retrieve_paragraphs(
        seeds: List[str] = typer.Option(..., help='Seed terms'),
        work_dir: Optional[Path] = typer.Option(None, help='Work directory'),
        run_name: Optional[str] = typer.Option(None, help='Run name'),
        profile: str = typer.Option('default', help='Profile name'),
        top_k: int = typer.Option(15, help='Top K results'),
        backend: Optional[str] = typer.Option(None, help='Embedding backend'),
        model_path: Optional[str] = typer.Option(None, help='Path to model'),
        log_level: str = typer.Option('INFO', help='Log level')
):
    work_dir, run_name, model_path, backend = get_effective_params(work_dir, run_name, model_path, backend,
                                                                   profile=profile)
    if not model_path:
        raise typer.BadParameter('model_path must be specified or saved in run config')
    if not backend:
        logger.warning(f'No backend specified, defaulting to sbert.')
        backend = 'sbert'

    setup_logging(work_dir, run_name, log_level)
    local_logger = logger.bind(work_dir=str(work_dir), run_name=run_name, command='retrieve_paragraphs')
    logging_utils.log_command_start(
        local_logger, 'retrieve_paragraphs',
        seeds_count=len(seeds), top_k=top_k, backend=backend
    )
    t0 = time.perf_counter()

    manager = IndexManager(work_dir, run_name)
    store = manager.load_store('paragraphs')
    embed_backend = get_embedding_backend(backend, model_path)

    # load store config for corpus info
    store_path = paths.get_global_vectorstore_path(work_dir, run_name, 'paragraphs')
    corpus_backend = None
    config_toml_path = store_path / 'config.toml'
    if config_toml_path.exists():
        with open(config_toml_path, 'rb') as f:
            import tomllib
            store_config = tomllib.load(f)
        corpus_backend = _get_corpus_backend_from_config(store_config, work_dir)
    else:
        # try to see if it is sharded and we can get corpus info from one of the shards
        shard_config_path = store_path / 'shards' / 'shard_00000' / 'config.toml'
        if shard_config_path.exists():
            with open(shard_config_path, 'rb') as f:
                import tomllib
                store_config = tomllib.load(f)
            corpus_backend = _get_corpus_backend_from_config(store_config, work_dir)

    # fallback info for old indexes
    run_config = load_run_config(work_dir, run_name)
    input_path = Path(run_config.get('input')) if run_config.get('input') else None
    id_col_eff = run_config.get('id_col', 'id')
    text_col_eff = run_config.get('text_col', 'text')

    query_vecs = embed_backend.embed(seeds)
    results = store.query(query_vecs, top_k=top_k)

    formatted_results = []
    for i, seed in enumerate(seeds):
        for idx, score, meta in results[i]:
            res = {
                'seed': seed,
                'score': score,
                **meta,
            }
            if 'text' not in res:
                if corpus_backend:
                    res['text'] = resolve_text_for_result(corpus_backend, meta)
                else:
                    res['text'] = reconstruct_text_if_needed(meta, work_dir, input_path, id_col_eff, text_col_eff)
            formatted_results.append(res)

    # save jsonlines
    out_path = paths.get_run_artifacts_path(work_dir, run_name) / 'results_paragraphs.jsonl'
    with open(out_path, 'w', encoding='utf8') as f:
        for res in formatted_results:
            f.write(json.dumps(res) + '\n')

    # save html
    html_path = paths.get_run_reports_path(work_dir, run_name) / 'paragraphs.html'
    reports_html.render_results_html('Retrieve Paragraphs', run_name, backend, formatted_results, str(html_path))

    elapsed = time.perf_counter() - t0
    logging_utils.log_command_done(
        local_logger, 'retrieve_paragraphs', elapsed,
        results_count=len(formatted_results), output_path=str(out_path)
    )
    logger.info(f'Retrieved {len(formatted_results)} paragraphs. Results saved to {out_path} and {html_path}')


@app.command()
def retrieve_notes(
        seeds: List[str] = typer.Option(..., help='Seed terms'),
        work_dir: Optional[Path] = typer.Option(None, help='Work directory'),
        run_name: Optional[str] = typer.Option(None, help='Run name'),
        profile: str = typer.Option('default', help='Profile name'),
        top_k: int = typer.Option(15, help='Top K results'),
        segment_sentences: bool = typer.Option(False, help='Segment notes into sentences and offer most relevant ones'),
        top_k_sentences: int = typer.Option(5, help='Top K relevant sentences per note if segmenting'),
        backend: Optional[str] = typer.Option(None, help='Embedding backend'),
        model_path: Optional[str] = typer.Option(None, help='Path to model'),
        log_level: str = typer.Option('INFO', help='Log level')
):
    work_dir, run_name, model_path, backend = get_effective_params(work_dir, run_name, model_path, backend,
                                                                   profile=profile)
    if not model_path:
        raise typer.BadParameter('model_path must be specified or saved in run config')
    if not backend:
        backend = 'sbert'

    setup_logging(work_dir, run_name, log_level)
    local_logger = logger.bind(work_dir=str(work_dir), run_name=run_name, command='retrieve_notes')
    logging_utils.log_command_start(
        local_logger, 'retrieve_notes',
        seeds_count=len(seeds), top_k=top_k, segment_sentences=segment_sentences,
        top_k_sentences=top_k_sentences, backend=backend
    )
    t0 = time.perf_counter()

    manager = IndexManager(work_dir, run_name)
    store = manager.load_store('notes')
    embed_backend = get_embedding_backend(backend, model_path)

    # load store config for corpus info
    store_path = paths.get_global_vectorstore_path(work_dir, run_name, 'notes')
    corpus_backend = None
    config_toml_path = store_path / 'config.toml'
    if config_toml_path.exists():
        with open(config_toml_path, 'rb') as f:
            import tomllib
            store_config = tomllib.load(f)
        corpus_backend = _get_corpus_backend_from_config(store_config, work_dir)
    else:
        # try to see if it is sharded and we can get corpus info from one of the shards
        shard_config_path = store_path / 'shards' / 'shard_00000' / 'config.toml'
        if shard_config_path.exists():
            with open(shard_config_path, 'rb') as f:
                import tomllib
                store_config = tomllib.load(f)
            corpus_backend = _get_corpus_backend_from_config(store_config, work_dir)

    # Fallback info for old indexes
    run_config = load_run_config(work_dir, run_name)
    input_path = Path(run_config.get('input')) if run_config.get('input') else None
    id_col_eff = run_config.get('id_col', 'id')
    text_col_eff = run_config.get('text_col', 'text')

    query_vecs = embed_backend.embed(seeds)
    results = store.query(query_vecs, top_k=top_k)

    # get NLP engine if segmenting
    nlp_engine = None
    if segment_sentences:
        engine_name = run_config.get('engine', 'regex')
        nlp_engine = get_nlp_engine(engine_name)

    formatted_results = []
    for i, seed in enumerate(seeds):
        seed_vec = query_vecs[i].reshape(1, -1)
        for idx, score, meta in results[i]:
            res = {
                'seed': seed,
                'score': score,
                **meta
            }
            if 'text' not in res:
                if corpus_backend:
                    res['text'] = resolve_text_for_result(corpus_backend, meta)
                else:
                    res['text'] = reconstruct_text_if_needed(meta, work_dir, input_path, id_col_eff, text_col_eff)

            if segment_sentences and nlp_engine:
                text = res.get('text', '')
                sentences = nlp_engine.split_sentences(text)
                if sentences:
                    sent_vecs = embed_backend.embed(sentences)

                    import numpy as np
                    norm_sent = np.linalg.norm(sent_vecs, axis=1, keepdims=True)
                    norm_seed = np.linalg.norm(seed_vec)
                    # avoid division by zero
                    norm_sent[norm_sent == 0] = 1.0
                    if norm_seed == 0: norm_seed = 1.0

                    sims = (sent_vecs @ seed_vec.T).flatten() / (norm_sent.flatten() * norm_seed)

                    top_sent_indices = np.argsort(sims)[::-1][:top_k_sentences]
                    relevant_sentences = []
                    for s_idx in top_sent_indices:
                        relevant_sentences.append({
                            'text': sentences[s_idx],
                            'score': float(sims[s_idx])
                        })
                    res['relevant_sentences'] = relevant_sentences

            formatted_results.append(res)

    out_path = paths.get_run_artifacts_path(work_dir, run_name) / 'results_notes.jsonl'
    with open(out_path, 'w', encoding='utf8') as f:
        for res in formatted_results:
            f.write(json.dumps(res) + '\n')

    html_path = paths.get_run_reports_path(work_dir, run_name) / 'notes.html'
    reports_html.render_results_html('Retrieve Notes', run_name, backend, formatted_results, str(html_path))

    elapsed = time.perf_counter() - t0
    logging_utils.log_command_done(
        local_logger, 'retrieve_notes', elapsed,
        results_count=len(formatted_results), output_path=str(out_path)
    )
    logger.info(f'Retrieved {len(formatted_results)} notes. Results saved to {out_path} and {html_path}')


@app.command()
def retrieve_sentences(
        seeds: List[str] = typer.Option(..., help='Seed terms'),
        work_dir: Optional[Path] = typer.Option(None, help='Work directory'),
        run_name: Optional[str] = typer.Option(None, help='Run name'),
        profile: str = typer.Option('default', help='Profile name'),
        top_k: int = typer.Option(15, help='Top K results'),
        backend: Optional[str] = typer.Option(None, help='Embedding backend'),
        model_path: Optional[str] = typer.Option(None, help='Path to model'),
        log_level: str = typer.Option('INFO', help='Log level')
):
    work_dir, run_name, model_path, backend = get_effective_params(work_dir, run_name, model_path, backend,
                                                                   profile=profile)
    if not model_path:
        raise typer.BadParameter('model_path must be specified or saved in run config')
    if not backend:
        backend = 'sbert'

    setup_logging(work_dir, run_name, log_level)
    local_logger = logger.bind(work_dir=str(work_dir), run_name=run_name, command='retrieve_sentences')
    logging_utils.log_command_start(
        local_logger, 'retrieve_sentences',
        seeds_count=len(seeds), top_k=top_k, backend=backend
    )
    t0 = time.perf_counter()

    manager = IndexManager(work_dir, run_name)
    store = manager.load_store('sentences')
    embed_backend = get_embedding_backend(backend, model_path)

    # load store config for corpus info
    store_path = paths.get_global_vectorstore_path(work_dir, run_name, 'sentences')
    corpus_backend = None
    config_toml_path = store_path / 'config.toml'
    if config_toml_path.exists():
        with open(config_toml_path, 'rb') as f:
            import tomllib
            store_config = tomllib.load(f)
        corpus_backend = _get_corpus_backend_from_config(store_config, work_dir)
    else:
        # try to see if it is sharded and we can get corpus info from one of the shards
        shard_config_path = store_path / 'shards' / 'shard_00000' / 'config.toml'
        if shard_config_path.exists():
            with open(shard_config_path, 'rb') as f:
                import tomllib
                store_config = tomllib.load(f)
            corpus_backend = _get_corpus_backend_from_config(store_config, work_dir)

    # fallback info for old indexes
    run_config = load_run_config(work_dir, run_name)
    input_path = Path(run_config.get('input')) if run_config.get('input') else None
    id_col_eff = run_config.get('id_col', 'id')
    text_col_eff = run_config.get('text_col', 'text')

    query_vecs = embed_backend.embed(seeds)
    results = store.query(query_vecs, top_k=top_k)

    formatted_results = []
    for i, seed in enumerate(seeds):
        for idx, score, meta in results[i]:
            res = {
                'seed': seed,
                'score': score,
                **meta
            }
            if 'text' not in res:
                if corpus_backend:
                    res['text'] = resolve_text_for_result(corpus_backend, meta)
                else:
                    res['text'] = reconstruct_text_if_needed(meta, work_dir, input_path, id_col_eff, text_col_eff)
            formatted_results.append(res)

    out_path = paths.get_run_artifacts_path(work_dir, run_name) / 'results_sentences.jsonl'
    with open(out_path, 'w', encoding='utf8') as f:
        for res in formatted_results:
            f.write(json.dumps(res) + '\n')

    html_path = paths.get_run_reports_path(work_dir, run_name) / 'sentences.html'
    reports_html.render_results_html('Retrieve Sentences', run_name, backend, formatted_results, str(html_path))

    elapsed = time.perf_counter() - t0
    logging_utils.log_command_done(
        local_logger, 'retrieve_sentences', elapsed,
        results_count=len(formatted_results), output_path=str(out_path)
    )
    logger.info(f'Retrieved {len(formatted_results)} sentences. Results saved to {out_path} and {html_path}')


@app.command()
def retrieve_phrases(
        seeds: List[str] = typer.Option(..., help='Seed terms'),
        work_dir: Optional[Path] = typer.Option(None, help='Work directory'),
        run_name: Optional[str] = typer.Option(None, help='Run name'),
        profile: str = typer.Option('default', help='Profile name'),
        top_k: int = typer.Option(15, help='Top K results'),
        backend: Optional[str] = typer.Option(None, help='Embedding backend'),
        model_path: Optional[str] = typer.Option(None, help='Path to model'),
        log_level: str = typer.Option('INFO', help='Log level')
):
    work_dir, run_name, model_path, backend = get_effective_params(work_dir, run_name, model_path, backend,
                                                                   profile=profile)
    if not model_path:
        raise typer.BadParameter('model_path must be specified or saved in run config')
    if not backend:
        backend = 'sbert'

    setup_logging(work_dir, run_name, log_level)
    local_logger = logger.bind(work_dir=str(work_dir), run_name=run_name, command='retrieve_phrases')
    logging_utils.log_command_start(
        local_logger, 'retrieve_phrases',
        seeds_count=len(seeds), top_k=top_k, backend=backend
    )
    t0 = time.perf_counter()

    manager = IndexManager(work_dir, run_name)
    store = manager.load_store('phrases')
    embed_backend = get_embedding_backend(backend, model_path)

    # load store config for corpus info
    store_path = paths.get_global_vectorstore_path(work_dir, run_name, 'phrases')
    corpus_backend = None
    if (store_path / 'config.toml').exists():
        with open(store_path / 'config.toml', 'rb') as f:
            import tomllib
            store_config = tomllib.load(f)
        corpus_backend = _get_corpus_backend_from_config(store_config, work_dir)

    # fallback info for old indexes
    run_config = load_run_config(work_dir, run_name)
    input_path = Path(run_config.get('input')) if run_config.get('input') else None
    id_col_eff = run_config.get('id_col', 'id')
    text_col_eff = run_config.get('text_col', 'text')

    # query vecs
    query_vecs = embed_backend.embed(seeds)
    results = store.query(query_vecs, top_k=top_k)

    formatted_results = []
    for i, seed in enumerate(seeds):
        for idx, score, meta in results[i]:
            res = {
                'seed': seed,
                'score': score,
                **meta
            }
            if 'text' not in res:
                if corpus_backend:
                    res['text'] = resolve_text_for_result(corpus_backend, meta)
                else:
                    res['text'] = reconstruct_text_if_needed(meta, work_dir, input_path, id_col_eff, text_col_eff)
            formatted_results.append(res)

    out_path = paths.get_run_artifacts_path(work_dir, run_name) / 'results_phrases.jsonl'
    with open(out_path, 'w', encoding='utf8') as f:
        for res in formatted_results:
            f.write(json.dumps(res) + '\n')

    html_path = paths.get_run_reports_path(work_dir, run_name) / 'phrases.html'
    reports_html.render_results_html('Retrieve Phrases', run_name, backend, formatted_results, str(html_path))

    elapsed = time.perf_counter() - t0
    logging_utils.log_command_done(
        local_logger, 'retrieve_phrases', elapsed,
        results_count=len(formatted_results), output_path=str(out_path)
    )
    logger.info(f'Retrieved {len(formatted_results)} phrases. Results saved to {out_path} and {html_path}')


@app.command()
def expand_terms_model(
        seeds: List[str] = typer.Option(..., help='Seed terms'),
        work_dir: Optional[Path] = typer.Option(None, help='Work directory'),
        run_name: Optional[str] = typer.Option(None, help='Run name'),
        profile: str = typer.Option('default', help='Profile name'),
        top_k: int = typer.Option(10, help='Top K results'),
        threshold: float = typer.Option(0.0, help='Similarity threshold'),
        backend: Optional[str] = typer.Option(None, help='Embedding backend'),
        model_path: Optional[str] = typer.Option(None, help='Path to model'),
        log_level: str = typer.Option('INFO', help='Log level')
):
    """Expand terms using the model's internal vocabulary (unrelated to corpus)."""
    work_dir, run_name, model_path, backend = get_effective_params(work_dir, run_name, model_path, backend,
                                                                   profile=profile)
    if not model_path:
        raise typer.BadParameter('model_path must be specified or saved in run config')
    if not backend:
        backend = 'fasttext'  # Default to fasttext as it's more likely to support this

    setup_logging(work_dir, run_name, log_level)
    local_logger = logger.bind(work_dir=str(work_dir), run_name=run_name, command='expand_terms_model')
    logging_utils.log_command_start(
        local_logger, 'expand_terms_model',
        seeds_count=len(seeds), top_k=top_k, threshold=threshold, backend=backend
    )
    t0 = time.perf_counter()

    embed_backend = get_embedding_backend(backend, model_path)

    expanded_results = model_guided.expand_terms_with_model(
        seeds, embed_backend, top_k=top_k, threshold=threshold
    )

    if not expanded_results:
        logger.warning(
            'No expansion results found. Ensure the backend supports model-guided expansion and terms are in vocabulary.')

    out_path = paths.get_run_artifacts_path(work_dir, run_name) / 'expansion_model.jsonl'
    with open(out_path, 'w', encoding='utf8') as f:
        for res in expanded_results:
            f.write(json.dumps(res) + '\n')

    html_path = paths.get_run_reports_path(work_dir, run_name) / 'expansion_model.html'
    reports_html.render_results_html('Expand Terms (Model)', run_name, backend, expanded_results, str(html_path))

    elapsed = time.perf_counter() - t0
    logging_utils.log_command_done(
        local_logger, 'expand_terms_model', elapsed,
        results_count=len(expanded_results), output_path=str(out_path)
    )
    logger.info(
        f'Expanded {len(seeds)} terms to {len(expanded_results)} candidates via model. Results saved to {out_path} and {html_path}')


@app.command()
def expand_terms(
        seeds: List[str] = typer.Option(..., help='Seed terms'),
        work_dir: Optional[Path] = typer.Option(None, help='Work directory'),
        run_name: Optional[str] = typer.Option(None, help='Run name'),
        profile: str = typer.Option('default', help='Profile name'),
        top_k: int = typer.Option(10, help='Top K results'),
        threshold: float = typer.Option(0.0, help='Similarity threshold'),
        backend: Optional[str] = typer.Option(None, help='Embedding backend'),
        model_path: Optional[str] = typer.Option(None, help='Path to model'),
        log_level: str = typer.Option('INFO', help='Log level')
):
    work_dir, run_name, model_path, backend = get_effective_params(work_dir, run_name, model_path, backend,
                                                                   profile=profile)
    if not model_path:
        raise typer.BadParameter('model_path must be specified or saved in run config')
    if not backend:
        backend = 'sbert'

    setup_logging(work_dir, run_name, log_level)
    local_logger = logger.bind(work_dir=str(work_dir), run_name=run_name, command='expand_terms')
    logging_utils.log_command_start(
        local_logger, 'expand_terms',
        seeds_count=len(seeds), top_k=top_k, threshold=threshold, backend=backend
    )
    t0 = time.perf_counter()

    manager = IndexManager(work_dir, run_name)
    store = manager.load_store('vocab_terms')
    embed_backend = get_embedding_backend(backend, model_path)

    expanded_results = term_guided.expand_terms(seeds, store, embed_backend, top_k=top_k, threshold=threshold)

    out_path = paths.get_run_artifacts_path(work_dir, run_name) / 'expansion_terms.jsonl'
    with open(out_path, 'w', encoding='utf8') as f:
        for res in expanded_results:
            f.write(json.dumps(res) + '\n')

    html_path = paths.get_run_reports_path(work_dir, run_name) / 'expansion_terms.html'
    reports_html.render_results_html('Expand Terms', run_name, backend, expanded_results, str(html_path))

    elapsed = time.perf_counter() - t0
    logging_utils.log_command_done(
        local_logger, 'expand_terms', elapsed,
        results_count=len(expanded_results), output_path=str(out_path)
    )
    logger.info(
        f'Expanded {len(seeds)} terms to {len(expanded_results)} candidates. Results saved to {out_path} and {html_path}')


@app.command()
def expand_from_regex(
        input: Path = typer.Option(..., help='Path to corpus'),
        work_dir: Optional[Path] = typer.Option(None, help='Work directory'),
        run_name: Optional[str] = typer.Option(None, help='Run name'),
        profile: str = typer.Option('default', help='Profile name'),
        id_col: str = typer.Option('id', help='ID column name'),
        text_col: str = typer.Option('text', help='Text column name'),
        regex: List[str] = typer.Option(..., help='Regexes to match'),
        engine: str = typer.Option('regex', help='NLP engine'),
        use_noun_chunks: bool = typer.Option(False, help='Use noun chunks for expansion'),
        backend: Optional[str] = typer.Option(None, help='Embedding backend'),
        model_path: Optional[str] = typer.Option(None, help='Path to model'),
        top_k_terms: int = typer.Option(50, help='Top K terms'),
        top_k_sentences: int = typer.Option(50, help='Top K sentences'),
        threshold: float = typer.Option(0.6, help='Similarity threshold'),
        include_paragraphs: bool = typer.Option(False, help='Include paragraph results'),
        log_level: str = typer.Option('INFO', help='Log level')
):
    work_dir, run_name, model_path, backend = get_effective_params(work_dir, run_name, model_path, backend,
                                                                   profile=profile)
    if not model_path:
        raise typer.BadParameter('model_path must be specified or saved in run config')
    if not backend:
        backend = 'sbert'

    setup_logging(work_dir, run_name, log_level)
    local_logger = logger.bind(work_dir=str(work_dir), run_name=run_name, command='expand_from_regex')
    logging_utils.log_command_start(
        local_logger, 'expand_from_regex',
        regex_count=len(regex), engine=engine, use_noun_chunks=use_noun_chunks,
        threshold=threshold, backend=backend, include_paragraphs=include_paragraphs
    )
    t0 = time.perf_counter()

    nlp_engine = get_nlp_engine(engine)
    embed_backend = get_embedding_backend(backend, model_path)
    manager = IndexManager(work_dir, run_name, nlp_engine=nlp_engine)

    vocab_store = manager.load_store('vocab_terms')
    sent_store = manager.load_store('sentences')
    para_store = manager.load_store('paragraphs') if include_paragraphs else None

    corpus_it = reader.corpus_iter(input, id_col, text_col)

    results = regex_guided_sentences.expand_from_regex(
        corpus_it, regex, nlp_engine,
        vocab_store=vocab_store,
        sentence_store=sent_store,
        paragraph_store=para_store,
        backend=embed_backend,
        use_noun_chunks=use_noun_chunks,
        top_k_terms=top_k_terms,
        top_k_sentences=top_k_sentences,
        threshold=threshold,
        include_paragraphs=include_paragraphs
    )

    # post-process results for reconstruction
    for res in results:
        if 'expanded_term' in res and 'doc_byte_offset' in res:
            # We use expanded_term field for sentences/paragraphs in regex expansion
            if res.get('expanded_term') is None:
                meta = {
                    'doc_id': res.get('doc_id'),
                    'doc_byte_offset': res.get('doc_byte_offset'),
                    'start_char': res.get('start_char'),
                    'end_char': res.get('end_char')
                }
                res['expanded_term'] = reconstruct_text_if_needed(meta, work_dir, input, id_col, text_col)

    # save jsonlines
    out_path = paths.get_run_artifacts_path(work_dir, run_name) / 'regex_expansion.jsonl'
    with open(out_path, 'w', encoding='utf8') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')

    # save html
    html_path = paths.get_run_reports_path(work_dir, run_name) / 'regex_expansion.html'
    reports_html.render_results_html('Regex Expansion', run_name, backend, results, str(html_path))

    elapsed = time.perf_counter() - t0
    logging_utils.log_command_done(
        local_logger, 'expand_from_regex', elapsed,
        results_count=len(results), output_path=str(out_path)
    )
    logger.info(f'Regex expansion complete. Results saved to {out_path} and {html_path}')


if __name__ == '__main__':
    app()
