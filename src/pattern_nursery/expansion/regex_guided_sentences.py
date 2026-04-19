from typing import List, Dict, Any

import regex as re


def expand_from_regex(
        corpus_iter,
        regexes: List[str],
        nlp_engine,
        backend=None,
        use_noun_chunks: bool = False,
        top_k_terms: int = 50,
        top_k_sentences: int = 50,
        threshold: float = 0.6,
        include_paragraphs: bool = False,
        **kwargs
) -> List[Dict[str, Any]]:
    """Match regexes in corpus and expand semantically."""
    compiled_regexes = [re.compile(r) for r in regexes]

    matched_sentences = []

    # 1. Match sentences by regex
    for doc_id, text in corpus_iter:
        for sent, start, end in nlp_engine.split_sentences_with_spans(text):
            for r, pattern in zip(regexes, compiled_regexes):
                if pattern.search(sent):
                    matched_sentences.append({
                        'seed_regex': r,
                        'matched_sentence': sent,
                        'doc_id': doc_id,
                        'sent_id': None,  # we don't know it from simple split
                        'start': start,
                        'end': end
                    })

    results = []

    # 2. Term expansion if vocab_store is provided
    if 'vocab_store' in kwargs and kwargs['vocab_store'] is not None and backend:
        vocab_store = kwargs['vocab_store']
        seeds_for_expansion = []
        if use_noun_chunks:
            for match in matched_sentences:
                chunks = nlp_engine.noun_chunks(match['matched_sentence'])
                # normalization of expansion seeds if needed
                seeds_for_expansion.extend(chunks)
        else:
            # use tokens from matched sentences?
            for match in matched_sentences:
                tokens = nlp_engine.tokenize(match['matched_sentence'], lower=True)
                seeds_for_expansion.extend(tokens)

        if seeds_for_expansion:
            # remove duplicates
            seeds_for_expansion = list(set(seeds_for_expansion))
            # if vocab_store was built with lower=True (default), seeds should be lowercased
            # noun_chunks might not be lowercased by the engine...
            seeds_for_expansion = [s.lower() for s in seeds_for_expansion]
            query_vecs = backend.embed(seeds_for_expansion)
            expansion_results = vocab_store.query(query_vecs, top_k=top_k_terms)

            for i, seed in enumerate(seeds_for_expansion):
                for idx, score, meta in expansion_results[i]:
                    if score >= threshold:
                        results.append({
                            'seed_term': seed,
                            'expanded_term': meta.get('text') or meta.get('candidate'),
                            'score': score,
                            'source': 'regex_seeded_term_expansion'
                        })

    # 3. Retrieve similar sentences
    if 'sentence_store' in kwargs and kwargs['sentence_store'] is not None and backend and matched_sentences:
        sentence_store = kwargs['sentence_store']
        sent_texts = [m['matched_sentence'] for m in matched_sentences]
        query_vecs = backend.embed(sent_texts)
        sim_sent_results = sentence_store.query(query_vecs, top_k=top_k_sentences)

        for i, match in enumerate(matched_sentences):
            for idx, score, meta in sim_sent_results[i]:
                if score >= threshold:
                    results.append({
                        'seed_regex': match['seed_regex'],
                        'matched_sentence': match['matched_sentence'],
                        'doc_id': meta['doc_id'],
                        'sent_id': meta.get('sent_id'),
                        'expanded_term': meta.get('text'),
                        'doc_byte_offset': meta.get('doc_byte_offset'),
                        'start_char': meta.get('start_char'),
                        'end_char': meta.get('end_char'),
                        'score': score,
                        'source': 'regex_seeded_sentence_expansion'
                    })

    # 4. Similar paragraphs
    if include_paragraphs and 'paragraph_store' in kwargs and kwargs[
        'paragraph_store'] is not None and backend and matched_sentences:
        paragraph_store = kwargs['paragraph_store']
        sent_texts = [m['matched_sentence'] for m in matched_sentences]
        query_vecs = backend.embed(sent_texts)
        sim_para_results = paragraph_store.query(query_vecs, top_k=top_k_sentences)

        for i, match in enumerate(matched_sentences):
            for idx, score, meta in sim_para_results[i]:
                if score >= threshold:
                    results.append({
                        'seed_regex': match['seed_regex'],
                        'matched_sentence': match['matched_sentence'],
                        'doc_id': meta['doc_id'],
                        'para_id': meta.get('para_id'),
                        'expanded_term': meta.get('text'),
                        'doc_byte_offset': meta.get('doc_byte_offset'),
                        'start_char': meta.get('start_char'),
                        'end_char': meta.get('end_char'),
                        'score': score,
                        'source': 'regex_seeded_paragraph_expansion'
                    })

    return results
