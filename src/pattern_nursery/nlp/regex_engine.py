from typing import List, Tuple

import regex as re

from pattern_nursery.nlp.base import NLPEngine


class RegexEngine(NLPEngine):
    """Basic regex-based NLP engine."""

    def __init__(self):
        # split by . ! ? followed by whitespace, OR by any newline(s)
        self.sent_pattern = re.compile(r'(?<=[.!?])\s+|\n+')
        # simple tokenizer: split by non-word characters
        self.tok_pattern = re.compile(r'\b\w+\b')

    def split_sentences(self, text: str) -> List[str]:
        if not text:
            return []
        return [s.strip() for s in self.sent_pattern.split(text) if s.strip()]

    def split_sentences_with_spans(self, text: str) -> List[Tuple[str, int, int]]:
        if not text:
            return []

        sentences = []
        last_end = 0
        for match in self.sent_pattern.finditer(text):
            sent = text[last_end: match.start()].strip()
            if sent:
                sentences.append((sent, last_end, match.start()))
            last_end = match.end()

        final_sent = text[last_end:].strip()
        if final_sent:
            sentences.append((final_sent, last_end, len(text)))

        return sentences

    def tokenize(self, text: str, lower: bool = False) -> List[str]:
        if not text:
            return []
        tokens = [t.strip() for t in self.tok_pattern.findall(text) if t.strip()]
        if lower:
            return [t.lower() for t in tokens]
        return tokens

    def tokenize_with_spans(self, text: str) -> List[Tuple[str, int, int]]:
        if not text:
            return []
        return [(m.group(), m.start(), m.end()) for m in self.tok_pattern.finditer(text)]

    def noun_chunks(self, text: str) -> List[str]:
        """Extract noun chunks from text.
        
        Regex engine doesn't support true noun chunks, so we return an empty list.
        """
        return []
