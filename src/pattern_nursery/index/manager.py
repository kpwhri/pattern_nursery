import itertools
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union, Iterable, Tuple

from loguru import logger

from pattern_nursery.embeddings.vectorstores.chroma_store import ChromaStore
from pattern_nursery.embeddings.vectorstores.faiss_store import FaissStore
from pattern_nursery.index import paths
from pattern_nursery.io import reader


def _chunk_it(iterable: Iterable, size: int) -> Iterable[Tuple]:
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


def _worker_process_notes(chunk, backend_params):
    from pattern_nursery.cli.main import get_embedding_backend
    backend = get_embedding_backend(**backend_params)

    # chunk can be (doc_id, text) or (doc_id, doc_byte_offset, text)
    if chunk and len(chunk[0]) == 3:
        texts = [c[2] for c in chunk]
        metadata = [{'doc_id': str(c[0]), 'doc_byte_offset': c[1]} for c in chunk]
    else:
        texts = [c[1] for c in chunk]
        metadata = [{'doc_id': str(c[0]), 'text': c[1]} for c in chunk]

    vecs = backend.embed(texts)
    return vecs, metadata


def _worker_process_sentences(chunk, backend_params, nlp_engine_name):
    from pattern_nursery.cli.main import get_embedding_backend, get_nlp_engine
    backend = get_embedding_backend(**backend_params)
    nlp_engine = get_nlp_engine(nlp_engine_name)

    all_texts = []
    all_meta = []

    for item in chunk:
        if len(item) == 3:
            doc_id, doc_byte_offset, text = item
        else:
            doc_id, text = item
            doc_byte_offset = None

        for sent_id, (sentence, start, end) in enumerate(nlp_engine.split_sentences_with_spans(text)):
            all_texts.append(sentence)
            meta = {
                'doc_id': str(doc_id),
                'sent_id': sent_id,
                'start_char': start,
                'end_char': end
            }
            if doc_byte_offset is not None:
                meta['doc_byte_offset'] = doc_byte_offset
            else:
                meta['text'] = sentence
            all_meta.append(meta)

    if not all_texts:
        return None

    vecs = backend.embed(all_texts)
    return vecs, all_meta


def _worker_process_paragraphs(chunk, backend_params, nlp_engine_name, para_params):
    from pattern_nursery.cli.main import get_embedding_backend, get_nlp_engine
    from pattern_nursery.io import reader
    backend = get_embedding_backend(**backend_params)
    nlp_engine = get_nlp_engine(nlp_engine_name)

    mode = para_params.get('mode', 'sentences')
    window_size = para_params.get('window_size', 3)
    overlap = para_params.get('overlap', 1)
    token_window = para_params.get('token_window', 50)
    token_overlap = para_params.get('token_overlap', 10)

    all_texts = []
    all_meta = []

    for item in chunk:
        if len(item) == 3:
            doc_id, doc_byte_offset, text = item
            docs_iter = [(doc_id, text)]
        else:
            doc_id, text = item
            doc_byte_offset = None
            docs_iter = [(doc_id, text)]

        if mode == 'sentences':
            para_iter = reader.paragraph_iter_sentences(nlp_engine, docs_iter, window_size, overlap)
        else:
            para_iter = reader.paragraph_iter_tokens(nlp_engine, docs_iter, token_window, token_overlap)

        for p_item in para_iter:
            p_doc_id, para_id, para_text, start_char, end_char = p_item[:5]
            meta = {
                'doc_id': str(p_doc_id),
                'para_id': para_id,
                'start_char': start_char,
                'end_char': end_char
            }
            if doc_byte_offset is not None:
                meta['doc_byte_offset'] = doc_byte_offset
            else:
                meta['text'] = para_text

            if mode == 'sentences':
                meta['start_sent_id'] = p_item[5]
                meta['end_sent_id'] = p_item[6]
            else:
                meta['start_tok_idx'] = p_item[5]
                meta['end_tok_idx'] = p_item[6]

            all_texts.append(para_text)
            all_meta.append(meta)

    if not all_texts:
        return None

    vecs = backend.embed(all_texts)
    return vecs, all_meta


class IndexManager:
    def __init__(self, work_dir: Union[str, Path], run_name: str, vector_store_type: str = 'faiss', nlp_engine=None):
        self.work_dir = Path(work_dir)
        self.run_name = run_name
        self.vector_store_type = vector_store_type
        self.nlp_engine = nlp_engine
        self.corpus_info = {}

    def set_corpus_info(self, corpus_path: Path, corpus_kind: str, id_col: str, text_col: str,
                        table_name: Optional[str] = None, corpus_index_db: Optional[str] = None):
        """Set corpus information to be stored in collection configs."""
        self.corpus_info = {
            'corpus_path': str(corpus_path.absolute()),
            'corpus_kind': corpus_kind,
            'id_col': id_col,
            'text_col': text_col,
        }
        if table_name:
            self.corpus_info['table_name'] = table_name
        if corpus_index_db:
            self.corpus_info['corpus_index_db'] = corpus_index_db

        paths.ensure_run_dirs(self.work_dir, self.run_name)

    def exists(self, collection: str) -> bool:
        """Check if a collection already exists in this run."""
        store_path = paths.get_global_vectorstore_path(self.work_dir, self.run_name, collection)
        config_path = store_path / 'config.toml'
        if not config_path.exists():
            return False

        if self.vector_store_type == 'faiss':
            return (store_path / 'index.faiss').exists()
        elif self.vector_store_type == 'chroma':
            # check if directory has files
            return any(store_path.iterdir())
        return False

    def build_and_persist_vocabulary(self, texts_iter, lower: bool = True):
        """Build vocab from texts and persist."""
        vocab = set()
        for _, text in texts_iter:
            tokens = self.nlp_engine.tokenize(text, lower=lower)
            # ensure each token is stripped and not empty
            vocab.update(t.strip() for t in tokens if t.strip())

        vocab_list = sorted(list(vocab))
        artifacts_path = paths.get_run_artifacts_path(self.work_dir, self.run_name)
        artifacts_path.mkdir(parents=True, exist_ok=True)
        vocab_path = artifacts_path / 'vocab.txt'
        with open(vocab_path, 'w', encoding='utf8') as f:
            for term in vocab_list:
                f.write(term + '\n')

        logger.info(f'Built vocabulary with {len(vocab_list)} terms (lower={lower}).')
        return vocab_list

    def embed_and_store(self, texts_iter, backend, collection='vocab_terms', batch_size=10000, num_workers=1):
        """Embed texts and store in global vectorstore."""
        if num_workers > 1:
            self._parallel_index(texts_iter, backend, collection, _worker_process_notes, batch_size, num_workers)
            self._write_config(collection, backend.dimension,
                               paths.get_global_vectorstore_path(self.work_dir, self.run_name, collection),
                               backend=backend,
                               extra={'lower': True} if collection == 'vocab_terms' else None)
            return

        texts = []
        metadata = []
        for item in texts_iter:
            if len(item) == 3:
                doc_id, doc_byte_offset, text = item
                texts.append(text)
                metadata.append({'doc_id': str(doc_id), 'doc_byte_offset': doc_byte_offset})
            else:
                doc_id, text = item
                texts.append(text)
                metadata.append({'doc_id': str(doc_id), 'text': text})

        if not texts:
            return

        # batch embed
        vecs = backend.embed(texts)

        store_path = paths.get_global_vectorstore_path(self.work_dir, self.run_name, collection)
        store = self._get_or_create_store(collection, backend.dimension, store_path)
        store.add(vecs, metadata)
        store.persist(store_path)

        self._write_config(collection, backend.dimension, store_path, backend=backend,
                           extra={'lower': True} if collection == 'vocab_terms' else None)
        logger.info(f'Stored {len(texts)} items in collection {collection}.')

    def index_sentences(self, docs_iter, backend, batch_size=10000, num_workers=1):
        """Index sentences from documents."""
        if num_workers > 1:
            worker_kwargs = {'nlp_engine_name': self.nlp_engine.name.lower().replace('engine', '')}
            self._parallel_index(docs_iter, backend, 'sentences', _worker_process_sentences,
                                 batch_size, num_workers, worker_kwargs)
            self._write_config('sentences', backend.dimension,
                               paths.get_global_vectorstore_path(self.work_dir, self.run_name, 'sentences'),
                               backend=backend)
            return

        metadata = []
        internal_batch_size = 1000
        batch_texts = []
        docs_seen = 0
        total_units = 0
        t0 = time.perf_counter()

        for item in docs_iter:
            docs_seen += 1
            if len(item) == 3:
                doc_id, doc_byte_offset, text = item
            else:
                doc_id, text = item
                doc_byte_offset = None

            for sent_id, (sentence, start, end) in enumerate(self.nlp_engine.split_sentences_with_spans(text)):
                batch_texts.append(sentence)
                total_units += 1
                meta = {
                    'doc_id': str(doc_id),
                    'sent_id': sent_id,
                    'start_char': start,
                    'end_char': end
                }
                if doc_byte_offset is not None:
                    meta['doc_byte_offset'] = doc_byte_offset
                else:
                    meta['text'] = sentence
                metadata.append(meta)

                if len(batch_texts) >= internal_batch_size:
                    self._add_to_store('sentences', backend, batch_texts, metadata)
                    batch_texts = []
                    metadata = []

            if docs_seen % 1000 == 0:
                elapsed = time.perf_counter() - t0
                logger.info(f'Indexing progress (sentences): {docs_seen} docs, {total_units} units, {elapsed:.1f}s')

        if batch_texts:
            self._add_to_store('sentences', backend, batch_texts, metadata)

        elapsed = time.perf_counter() - t0
        logger.info(
            f'Indexing summary (sentences): {docs_seen} docs seen, '
            f'{total_units} units generated, duration: {elapsed:.1f}s'
        )

    def index_phrases(self, docs_iter, backend, types=('noun_chunks', 'bigrams'), batch_size=10_000, num_workers=1):
        """Index noun chunks and/or n-grams from documents."""
        # for phrases, build a global vocab first -- TODO: consider parallelizing?
        vocab = set()
        for item in docs_iter:
            if len(item) == 3:
                doc_id, _, text = item
            else:
                doc_id, text = item
            if 'noun_chunks' in types:
                vocab.update(self.nlp_engine.noun_chunks(text))
            if 'bigrams' in types:
                vocab.update(self.nlp_engine.ngrams(text, n=2))
            if 'unigrams' in types:
                vocab.update(self.nlp_engine.tokenize(text))

        # remove empty or very short strings
        vocab = {t.strip() for t in vocab if len(t.strip()) > 1}

        if not vocab:
            logger.warning('No phrases found to index.')
            return

        vocab_list = sorted(list(vocab))

        # embed and store in 'phrases' collection
        vocab_iter = ((f'ph_{i}', term) for i, term in enumerate(vocab_list))
        self.embed_and_store(vocab_iter, backend, 'phrases', batch_size=batch_size, num_workers=num_workers)

    def index_notes(self, docs_iter, backend, batch_size=10_000, num_workers=1):
        """Index full notes."""
        self.embed_and_store(docs_iter, backend, collection='notes', batch_size=batch_size, num_workers=num_workers)

    def index_paragraphs(
            self,
            docs_iter,
            backend,
            mode='sentences',
            window_size=3,
            overlap=1,
            token_window=50,
            token_overlap=10,
            collection='paragraphs',
            batch_size=10000,
            num_workers=1
    ):
        """Index paragraphs."""
        if num_workers > 1:
            para_params = {
                'mode': mode,
                'window_size': window_size,
                'overlap': overlap,
                'token_window': token_window,
                'token_overlap': token_overlap
            }
            worker_kwargs = {
                'nlp_engine_name': self.nlp_engine.name.lower().replace('engine', ''),
                'para_params': para_params
            }
            self._parallel_index(docs_iter, backend, collection, _worker_process_paragraphs,
                                 batch_size, num_workers, worker_kwargs)
            self._write_config(collection, backend.dimension,
                               paths.get_global_vectorstore_path(self.work_dir, self.run_name, collection),
                               backend=backend, extra=para_params)
            return

        batch_texts = []
        metadata = []
        internal_batch_size = 1000
        docs_seen = 0
        total_units = 0
        t0 = time.perf_counter()

        for item in docs_iter:
            docs_seen += 1
            if len(item) == 3:
                doc_id, doc_byte_offset, text = item
            else:
                doc_id, text = item
                doc_byte_offset = None

            sub_docs_iter = [(doc_id, text)]
            if mode == 'sentences':
                para_iter = reader.paragraph_iter_sentences(self.nlp_engine, sub_docs_iter, window_size, overlap)
            else:
                para_iter = reader.paragraph_iter_tokens(self.nlp_engine, sub_docs_iter, token_window, token_overlap)

            for p_item in para_iter:
                total_units += 1
                p_doc_id, para_id, para_text, start_char, end_char = p_item[:5]
                meta = {
                    'doc_id': str(p_doc_id),
                    'para_id': para_id,
                    'start_char': start_char,
                    'end_char': end_char
                }
                if doc_byte_offset is not None:
                    meta['doc_byte_offset'] = doc_byte_offset
                else:
                    meta['text'] = para_text

                if mode == 'sentences':
                    meta['start_sent_id'] = p_item[5]
                    meta['end_sent_id'] = p_item[6]
                else:
                    meta['start_tok_idx'] = p_item[5]
                    meta['end_tok_idx'] = p_item[6]

                batch_texts.append(para_text)
                metadata.append(meta)

                if len(batch_texts) >= internal_batch_size:
                    self._add_to_store(collection, backend, batch_texts, metadata)
                    batch_texts = []
                    metadata = []

            if docs_seen % 1000 == 0:
                elapsed = time.perf_counter() - t0
                logger.info(f'Indexing progress ({collection}): {docs_seen} docs, {total_units} units, {elapsed:.1f}s')

        if batch_texts:
            self._add_to_store(collection, backend, batch_texts, metadata)

        elapsed = time.perf_counter() - t0
        logger.info(
            f'Indexing summary ({collection}): {docs_seen} docs seen, '
            f'{total_units} units generated, duration: {elapsed:.1f}s'
        )

        # write specific paragraph config
        store_path = paths.get_global_vectorstore_path(self.work_dir, self.run_name, collection)
        config = {
            'mode': mode,
            'window_size': window_size,
            'overlap': overlap,
            'token_window': token_window,
            'token_overlap': token_overlap
        }
        self._write_config(collection, backend.dimension, store_path, backend=backend, extra=config)

    def _parallel_index(self, docs_iter, backend, collection, worker_func, batch_size, num_workers, worker_kwargs=None):
        if 'SentenceTransformers' in backend.__class__.__name__:
            backend_name = 'sbert'
        elif 'FastText' in backend.__class__.__name__:
            backend_name = 'fasttext'
        elif 'Word2Vec' in backend.__class__.__name__:
            backend_name = 'word2vec'
        elif 'BM25' in backend.__class__.__name__:
            backend_name = 'bm25'
        else:
            backend_name = 'mock'  # for tests or fallback

        logger.info(
            f'Starting parallel indexing: collection={collection}, batch_size={batch_size}, '
            f'num_workers={num_workers}, backend={backend_name}, dimension={backend.dimension}'
        )
        t0 = time.perf_counter()

        backend_params = {
            'backend_name': backend_name,
            'model_path': backend.model_name
        }

        store_path = paths.get_global_vectorstore_path(self.work_dir, self.run_name, collection)
        store = self._get_or_create_store(collection, backend.dimension, store_path)

        total_vectors = 0
        chunks_scheduled = 0
        chunks_completed = 0

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for chunk in _chunk_it(docs_iter, batch_size):
                kwargs = {'backend_params': backend_params}
                if worker_kwargs:
                    kwargs.update(worker_kwargs)
                futures.append(executor.submit(worker_func, chunk, **kwargs))
                chunks_scheduled += 1

            for future in as_completed(futures):
                result = future.result()
                chunks_completed += 1
                if result:
                    vecs, meta = result
                    total_vectors += len(meta)
                    store.add(vecs, meta)
                    store.persist(store_path)

                if chunks_completed % max(1, (chunks_scheduled // 10)) == 0 or chunks_completed == chunks_scheduled:
                    elapsed = time.perf_counter() - t0
                    logger.info(
                        f'Indexing progress ({collection}): {chunks_completed}/{chunks_scheduled} chunks, '
                        f'{total_vectors} vectors added, {elapsed:.1f}s elapsed'
                    )

        elapsed = time.perf_counter() - t0
        logger.info(
            f'Parallel indexing complete for collection {collection}. '
            f'Total vectors: {total_vectors}, duration: {elapsed:.1f}s'
        )

    def _add_to_store(self, collection, backend, texts, metadata):
        t0 = time.perf_counter()
        vecs = backend.embed(texts)
        embed_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        store_path = paths.get_global_vectorstore_path(self.work_dir, self.run_name, collection)
        store = self._get_or_create_store(collection, backend.dimension, store_path)
        store.add(vecs, metadata)
        store.persist(store_path)
        persist_time = time.perf_counter() - t1

        self._write_config(collection, backend.dimension, store_path, backend=backend)
        logger.debug(
            f'Added to store: collection={collection}, count={len(texts)}, '
            f'dim={backend.dimension}, embed_time={embed_time:.3f}s, persist_time={persist_time:.3f}s'
        )

    def _get_or_create_store(self, collection, dimension, store_path):
        if self.vector_store_type == 'faiss':
            if (store_path / 'index.faiss').exists():
                return FaissStore.load(store_path)
            return FaissStore(dimension)
        elif self.vector_store_type == 'chroma':
            # Chroma handles its own existence check
            return ChromaStore(collection, store_path)
        raise ValueError(f'Unknown store type: {self.vector_store_type}')

    def _write_config(self, collection, dim, store_path, backend=None, extra=None):
        config = {
            'store_type': self.vector_store_type,
            'collection': collection,
            'dim': dim,
            'index_type': 'flatip' if self.vector_store_type == 'faiss' else 'default',
            'persist_dir': str(store_path.relative_to(self.work_dir)),
            'ids_path': str(
                (paths.get_run_artifacts_path(self.work_dir, self.run_name) / 'ids.json').relative_to(self.work_dir)),
            'meta_path': str(
                (paths.get_run_artifacts_path(self.work_dir, self.run_name) / 'store_meta.json').relative_to(
                    self.work_dir)),
            'created_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'version': '0.1.0',
        }
        if backend:
            config['embedding_model'] = backend.model_name

        if self.nlp_engine:
            config['nlp_engine'] = self.nlp_engine.name

        if self.corpus_info:
            config.update(self.corpus_info)

        if extra:
            config.update(extra)

        # minimal manual TOML-like writer for simple key-value pairs
        config_path = store_path / 'config.toml'
        store_path.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf8') as f:
            for k, v in config.items():
                if isinstance(v, str):
                    v_esc = v.replace('\\', '\\\\')
                    f.write(f'{k} = "{v_esc}"\n')
                elif isinstance(v, bool):
                    f.write(f'{k} = {str(v).lower()}\n')
                else:
                    f.write(f'{k} = {v}\n')

    def load_store(self, collection):
        store_path = paths.get_global_vectorstore_path(self.work_dir, self.run_name, collection)
        config_path = store_path / 'config.toml'

        # check for sharded store if global doesn't exist
        shards_dir = store_path / 'shards'
        if not config_path.exists() and shards_dir.exists():
            # load and aggregate shards
            shard_dirs = sorted(list(shards_dir.glob('shard_*')))
            if not shard_dirs:
                raise FileNotFoundError(f'No shards found in {shards_dir}')

            logger.info(f'Loading and aggregating {len(shard_dirs)} shards for {collection}...')

            # load first shard to get type and dimension
            first_shard_path = shard_dirs[0]
            if self.vector_store_type == 'faiss':
                base_store = FaissStore.load(first_shard_path)
                # merge other shards
                for shard_dir in shard_dirs[1:]:
                    shard_store = FaissStore.load(shard_dir)
                    base_store.index.add(shard_store.index.reconstruct_n(0, shard_store.index.ntotal))
                    base_store.metadata.extend(shard_store.metadata)
                return base_store
            elif self.vector_store_type == 'chroma':
                # Chroma doesn't work like this...
                pass

        if not config_path.exists():
            raise FileNotFoundError(f'No config found for collection {collection} at {config_path}')

        # Minimal toml parser
        with open(config_path, 'rb') as f:
            import tomllib
            config = tomllib.load(f)

        if config['store_type'] == 'faiss':
            return FaissStore.load(store_path)
        elif config['store_type'] == 'chroma':
            return ChromaStore(collection, store_path)
        raise ValueError(f"Unknown store type {config['store_type']}")
