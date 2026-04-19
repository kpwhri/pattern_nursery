from typing import List, Tuple

from pattern_nursery.nlp.base import NLPEngine


class NLTKEngine(NLPEngine):
    """NLTK-based NLP engine."""

    def __init__(self):
        try:
            import nltk
            # ensure necessary data is downloaded if possible
            self.nltk = nltk
            try:
                self.nltk.data.find('tokenizers/punkt')
            except LookupError:
                self.nltk.download('punkt', quiet=True)
        except ImportError:
            raise ImportError('nltk is required for NLTKEngine')

    def split_sentences(self, text: str) -> List[str]:
        return self.nltk.sent_tokenize(text)

    def split_sentences_with_spans(self, text: str) -> List[Tuple[str, int, int]]:
        sentences = self.nltk.sent_tokenize(text)
        spans = []
        last_idx = 0
        for sent in sentences:
            start = text.find(sent, last_idx)
            if start != -1:
                end = start + len(sent)
                spans.append((sent, start, end))
                last_idx = end
        return spans

    def tokenize(self, text: str, lower: bool = False) -> List[str]:
        tokens = [t.strip() for t in self.nltk.word_tokenize(text) if t.strip()]
        if lower:
            return [t.lower() for t in tokens]
        return tokens

    def tokenize_with_spans(self, text: str) -> List[Tuple[str, int, int]]:
        # use treebank tokenizer to get spans
        from nltk.tokenize import TreebankWordTokenizer
        tokenizer = TreebankWordTokenizer()
        spans = list(tokenizer.span_tokenize(text))
        return [(text[s:e], s, e) for s, e in spans]

    def noun_chunks(self, text: str) -> List[str]:
        """Extract noun phrases using NLTK's default chunker."""
        try:
            tokens = self.tokenize(text)
            import nltk
            try:
                self.nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                self.nltk.download('averaged_perceptron_tagger', quiet=True)
            try:
                self.nltk.data.find('chunkers/maxent_ne_chunker')
            except LookupError:
                self.nltk.download('maxent_ne_chunker', quiet=True)
            try:
                self.nltk.data.find('corpora/words')
            except LookupError:
                self.nltk.download('words', quiet=True)

            tagged = nltk.pos_tag(tokens)
            # define a simple grammar for noun phrases
            grammar = 'NP: {<DT>?<JJ>*<NN.*>+}'
            cp = nltk.RegexpParser(grammar)
            tree = cp.parse(tagged)

            chunks = []
            for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
                chunks.append(' '.join(word for word, pos in subtree.leaves()))
            return chunks
        except Exception:
            return []
