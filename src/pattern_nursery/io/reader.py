import json
from pathlib import Path
from typing import Generator, Tuple, Union

import polars as pl


def corpus_iter(
        df_or_path: Union[pl.DataFrame, Path, str],
        id_col: str,
        text_col: str,
        chunk_size: int = 1000,
) -> Generator[Tuple[str, str], None, None]:
    """Stream corpus line-by-line or in chunks. Returns (doc_id, text)."""
    if isinstance(df_or_path, (str, Path)):
        path = Path(df_or_path)
        if path.suffix == '.jsonl':
            with open(path, 'r', encoding='utf8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    yield str(data[id_col]), data[text_col]
        else:
            df = pl.read_csv(path) if path.suffix == '.csv' else pl.read_parquet(path)
            yield from _df_iter(df, id_col, text_col, chunk_size)
    elif isinstance(df_or_path, pl.DataFrame):
        yield from _df_iter(df_or_path, id_col, text_col, chunk_size)


def corpus_iter_jsonl_with_offsets(
        path: Path,
        id_col: str,
        text_col: str,
) -> Generator[Tuple[str, int, str], None, None]:
    """Stream JSONL with byte offsets. Returns (doc_id, byte_offset, text)."""
    with open(path, 'rb') as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                data = json.loads(line.decode('utf8'))
                yield str(data[id_col]), offset, data[text_col]
            except Exception:
                continue


def _df_iter(df: pl.DataFrame, id_col: str, text_col: str, chunk_size: int) -> Generator[Tuple[str, str], None, None]:
    for i in range(0, len(df), chunk_size):
        chunk = df.slice(i, chunk_size)
        for row in chunk.select([id_col, text_col]).iter_rows():
            yield str(row[0]), row[1]


def sentence_iter(nlp_engine, docs_iter: Generator[Tuple[str, str], None, None]):
    """Yield sentences from documents."""
    for doc_id, text in docs_iter:
        for sent_id, (sentence, start, end) in enumerate(nlp_engine.split_sentences_with_spans(text)):
            yield doc_id, sent_id, sentence, start, end


def paragraph_iter_sentences(
        nlp_engine,
        docs_iter: Generator[Tuple[str, str], None, None],
        window_size: int = 3,
        overlap: int = 1,
):
    """Slide a window of N sentences with overlap."""
    for item in docs_iter:
        if len(item) == 3:
            doc_id, _, text = item
        else:
            doc_id, text = item
        sentences = list(nlp_engine.split_sentences_with_spans(text))
        step = window_size - overlap
        if step <= 0:
            step = 1

        para_id = 0
        for i in range(0, len(sentences), step):
            window = sentences[i: i + window_size]
            if not window:
                break

            para_text = ' '.join(s[0] for s in window)
            start_char = window[0][1]
            end_char = window[-1][2]
            start_sent_id = i
            end_sent_id = i + len(window) - 1

            yield doc_id, para_id, para_text, start_char, end_char, start_sent_id, end_sent_id
            para_id += 1

            if i + window_size >= len(sentences):
                break


def paragraph_iter_tokens(
        nlp_engine,
        docs_iter: Generator[Tuple[str, str], None, None],
        token_window: int = 50,
        token_overlap: int = 10,
):
    """Tokenize and slide windows by tokens with overlap."""
    for doc_id, text in docs_iter:
        tokens_with_spans = list(nlp_engine.tokenize_with_spans(text))
        step = token_window - token_overlap
        if step <= 0:
            step = 1

        para_id = 0
        for i in range(0, len(tokens_with_spans), step):
            window = tokens_with_spans[i: i + token_window]
            if not window:
                break

            # reconstruct the text from spans to preserve whitespace if possible
            para_text = text[window[0][1]: window[-1][2]]
            start_char = window[0][1]
            end_char = window[-1][2]
            start_tok_idx = i
            end_tok_idx = i + len(window) - 1

            yield doc_id, para_id, para_text, start_char, end_char, start_tok_idx, end_tok_idx
            para_id += 1

            if i + token_window >= len(tokens_with_spans):
                break
