from typing import List, Tuple

from loguru import logger

from pattern_nursery.nlp.base import NLPEngine


class SpaCyEngine(NLPEngine):
    """spaCy-based NLP engine."""

    def __init__(self, model_name: str = 'en_core_web_sm'):
        try:
            import spacy
            try:
                self.nlp = spacy.load(model_name)
            except OSError:
                logger.info(f'Downloading spacy model {model_name}')
                spacy.cli.download(model_name)
                self.nlp = spacy.load(model_name)
        except ImportError:
            raise ImportError('spacy is required for SpaCyEngine')

    def split_sentences(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return [sent.text for sent in doc.sents]

    def split_sentences_with_spans(self, text: str) -> List[Tuple[str, int, int]]:
        doc = self.nlp(text)
        return [(sent.text, sent.start_char, sent.end_char) for sent in doc.sents]

    def tokenize(self, text: str, lower: bool = False) -> List[str]:
        doc = self.nlp(text)
        tokens = [token.text.strip() for token in doc if token.text.strip()]
        if lower:
            return [t.lower() for t in tokens]
        return tokens

    def tokenize_with_spans(self, text: str) -> List[Tuple[str, int, int]]:
        doc = self.nlp(text)
        return [(token.text, token.idx, token.idx + len(token.text)) for token in doc]

    def noun_chunks(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return [chunk.text for chunk in doc.noun_chunks]
