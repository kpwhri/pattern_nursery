from typing import Dict, Any

from pattern_nursery.io.corpus_backend import CorpusBackend


def resolve_text_for_result(corpus_backend: CorpusBackend, meta: Dict[str, Any], *, context_chars: int = 80) -> str:
    """Resolve full text or snippet for a search result metadata."""
    doc_id = str(meta.get('doc_id'))
    if not doc_id:
        return ''

    try:
        full_text = corpus_backend.get_text(doc_id)
    except Exception:
        return ''

    start_char = meta.get('start_char')
    end_char = meta.get('end_char')

    if start_char is not None and end_char is not None:
        # if span exists, we can return just the span or span + context
        span_text = full_text[start_char:end_char]
        if context_chars > 0:
            # optionally add context ??
            # ctx_start = max(0, start_char - context_chars)
            # ctx_end = min(len(full_text), end_char + context_chars)
            # return full_text[ctx_start:ctx_end]
            return span_text
        return span_text

    # if no span, return a preview
    return full_text[:200] + ('...' if len(full_text) > 200 else '')
