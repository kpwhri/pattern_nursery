# Example Workflows

This file shows two complete, end-to-end workflows using a simplified local layout.

- Corpus input: `corpus.jsonl`
- Work directory: `.nursery`
- Model path: `models/all-MiniLM-L6-v2`
- Run name: `v1`

The example keeps the seed phrases exactly as written so you can compare results across runs.

- [Build indexes and retrieve results for `time alone`](#tutorial-build-indexes-and-retrieve-results-for-time-alone)
- [Regex-based expansion for `time alone`](#tutorial-regex-based-expansion-for-time-alone)

---

# Tutorial: build indexes and retrieve results for `time alone`

This tutorial walks through the typical workflow for reviewing clinical text with a semantic index.

### What happens in this tutorial

1. The corpus is scanned and a corpus index is built.
2. Sentence, paragraph, phrase, and note indexes are created.
3. Retrieval commands search those indexes using the seed phrases:
    - `time alone`
    - `time alone with teen`
4. The tool writes JSONL artifacts and HTML reports into the run directory for review.

---

## 1) Build the corpus index

Before you can retrieve results, the corpus must be indexed. This step records corpus metadata and prepares the project
for later retrieval.

```bash
pattern-nursery index-corpus \
  --work-dir .nursery \
  --input corpus.jsonl \
  --id-col nlpid \
  --text-col text \
  --total-shards 1
```

### Expected output

This creates corpus indexing data inside the work directory, including:

- a corpus index database
- shard metadata for the corpus
- information needed for later reference reconstruction

### Why this matters

Later commands use the corpus index to:

- validate that the corpus has not changed
- reconstruct note text when needed
- support retrieval and report generation

---

## 2) Build the retrieval indexes

Next, initialize the semantic indexes. The same run name is used for every collection so all outputs stay in one place.

In this workflow, the indexes are built one collection at a time:

- `sentences`
- `paragraphs`
- `phrases`
- `notes`

Each command uses the same corpus, work directory, run name, engine, backend, and model path.

### 2a) Index sentences

```bash
pattern-nursery init-index \
  --work-dir .nursery \
  --run-name v1 \
  --input corpus.jsonl \
  --engine regex \
  --collection sentences \
  --backend sbert \
  --model-path models/all-MiniLM-L6-v2 \
  --id-col nlpid \
  --text-col text
```

#### What happens

- The corpus is read note by note.
- Notes are split into sentences.
- Sentence embeddings are computed with the selected model.
- Sentence vectors and metadata are stored in the `sentences` collection.

#### Expected output

- sentence vectors saved under the run’s vector store
- a run configuration file saved for reuse
- logs written for the run

---

### 2b) Index paragraphs

```bash
pattern-nursery init-index \
  --work-dir .nursery \
  --run-name v1 \
  --input corpus.jsonl \
  --engine regex \
  --collection paragraphs \
  --backend sbert \
  --model-path models/all-MiniLM-L6-v2 \
  --id-col nlpid \
  --text-col text
```

#### What happens

- Notes are split into sentence windows.
- Paragraph-like spans are created from those windows.
- Each paragraph is embedded and stored with provenance metadata.

#### Expected output

- paragraph vectors in the run’s vector store
- paragraph metadata with note and span references
- HTML-ready retrieval output for paragraph review later

---

### 2c) Index phrases

```bash
pattern-nursery init-index \
  --work-dir .nursery \
  --run-name v1 \
  --input corpus.jsonl \
  --engine regex \
  --collection phrases \
  --backend sbert \
  --model-path models/all-MiniLM-L6-v2 \
  --id-col nlpid \
  --text-col text
```

#### What happens

- Phrase-like units are extracted from the corpus.
- Each phrase is embedded and indexed.
- The stored metadata preserves the originating note and character spans.

#### Expected output

- phrase vectors in the run’s vector store
- phrase-level provenance information
- a retrieval-ready index for phrase lookup

---

### 2d) Index notes

```bash
pattern-nursery init-index \
  --work-dir .nursery \
  --run-name v1 \
  --input corpus.jsonl \
  --engine regex \
  --collection notes \
  --backend sbert \
  --model-path models/all-MiniLM-L6-v2 \
  --id-col nlpid \
  --text-col text
```

#### What happens

- Whole notes are embedded.
- Each note becomes a searchable item in the `notes` collection.
- The stored metadata keeps note-level references for later reconstruction and reporting.

#### Expected output

- note vectors saved in the vector store
- note-level JSONL artifacts created during retrieval
- a report-ready collection for full-document review

---

## 3) Retrieve relevant sentences

Now search for the closest matching sentences using the seed phrases.

```bash
pattern-nursery retrieve-sentences \
  --work-dir .nursery \
  --run-name v1 \
  --seeds "time alone" \
  --seeds "time alone with teen" \
  --top-k 100
```

### What happens

- Each seed phrase is embedded.
- The sentence index is queried for the nearest matches.
- The tool reconstructs sentence text when needed.
- Results are written as JSONL and HTML for review.

### Expected output

You should see:

- `results_sentences.jsonl` in the run artifacts folder
- `sentences.html` in the run reports folder

### What the results mean

Each result typically includes:

- the original seed phrase
- similarity score
- note or sentence identifiers
- character span information
- reconstructed text when available

---

## 4) Retrieve relevant phrases

Use the same seeds to find phrase-level matches.

```bash
pattern-nursery retrieve-phrases \
  --work-dir .nursery \
  --run-name v1 \
  --seeds "time alone" \
  --seeds "time alone with teen" \
  --top-k 100
```

### What happens

- The phrase index is queried with each seed.
- The closest phrase candidates are returned.
- Results are saved for later inspection in both machine-readable and human-readable formats.

### Expected output

- `results_phrases.jsonl`
- `phrases.html`

### Why this is useful

Phrase retrieval is often useful when you want:

- tighter lexical matches than sentence search
- compact reviewable snippets
- faster scanning of the most relevant fragments

---

## 5) Retrieve relevant paragraphs

Finally, search at the paragraph level.

```bash
pattern-nursery retrieve-paragraphs \
  --work-dir .nursery \
  --run-name v1 \
  --seeds "time alone" \
  --seeds "time alone with teen" \
  --top-k 100
```

### What happens

- The paragraph index is queried using the seed phrases.
- The best-matching paragraph spans are collected.
- The tool writes a report that can be reviewed by a human reviewer.

### Expected output

- `results_paragraphs.jsonl`
- `paragraphs.html`

### Why this is useful

Paragraph retrieval gives a broader context than sentence or phrase retrieval. It is especially helpful when:

- the relevant evidence spans multiple sentences
- you want to review surrounding context
- you want a balance between precision and context

---

## Review the generated files

After the tutorial completes, the main outputs live under the work directory:

```text
.nursery/
  corpus_index/
  runs/
    v1/
      artifacts/
      logs/
      reports/
  vectorstore/
```

### Common artifacts

- `results_sentences.jsonl`
- `results_phrases.jsonl`
- `results_paragraphs.jsonl`
- `notes.html`
- `sentences.html`
- `phrases.html`
- `paragraphs.html`

### What to look for in the HTML reports

The reports are designed for review. They show:

- the seed phrase that produced the match
- similarity scores
- matched text
- provenance metadata
- a readable layout for manual screening

---

## Summary

This workflow demonstrates a common pattern:

1. **index the corpus**
2. **build one or more collections**
3. **retrieve by seed phrase**
4. **review results in HTML**

For the seed phrases:

- `time alone`
- `time alone with teen`

you can compare sentence, phrase, paragraph, and note retrieval side by side to decide which collection is best for your
task.

---

# Tutorial: regex-based expansion for `time alone`

This workflow starts from a trusted regex anchor, then expands into semantically related sentences, paragraphs, and
terms. It is useful when you want a controlled search pattern but still want broader recall.

### What happens in this tutorial

1. A regex is applied to the corpus to find anchor matches.
2. Matching passages are used as expansion seeds.
3. The tool searches for related terms and related text spans.
4. The results are written to JSONL and HTML reports for review.

### The regex anchor

This example uses the following regex patterns:

- `time\\s+alone`
- `time\\s+alone\\s+with\\s+teen`

---

## 1) Run regex-based expansion

```bash
pattern-nursery expand-from-regex \
  --work-dir .nursery \
  --run-name v1 \
  --input corpus.jsonl \
  --id-col nlpid \
  --text-col text \
  --regex "time\\s+alone" \
  --regex "time\\s+alone\\s+with\\s+teen" \
  --engine regex \
  --backend sbert \
  --model-path models/all-MiniLM-L6-v2 \
  --top-k-terms 50 \
  --top-k-sentences 50 \
  --threshold 0.6 \
  --include-paragraphs
```

### What happens

- The corpus is streamed and each note is checked against the regex patterns.
- Any matching text becomes an anchor for expansion.
- The tool optionally extracts seeds from the matched text.
- It looks up similar vocabulary terms from the term index.
- It looks up similar sentences from the sentence index.
- Because `--include-paragraphs` is set, it also looks up paragraphs.

### Expected output

This command creates:

- `regex_expansion.jsonl` in the run artifacts folder
- `regex_expansion.html` in the run reports folder

### What the results mean

The expansion output usually combines several kinds of evidence:

- the regex match that triggered the expansion
- matched note or sentence context
- semantically related terms
- semantically related sentences
- optionally, related paragraphs

Each row is meant to be reviewable, not just machine-readable.

---

## 2) Review the expansion report

Open the HTML report after the run finishes.

### Expected report contents

The report should show:

- the regex anchor that matched
- the seed text derived from the match
- expansion candidates
- similarity scores
- provenance metadata
- readable review blocks for the matched context and expanded results

### Why this is useful

Regex-based expansion helps when you already trust a literal or pattern-based anchor but want to broaden recall without
switching to a fully manual search strategy.

It is especially helpful for:

- clinical phrases with spelling or phrasing variation
- concepts that are usually expressed in multiple ways
- workflows where reviewability matters as much as recall

---

## 3) Compare with retrieval results

The regex workflow is often easiest to use alongside sentence, phrase, and paragraph retrieval.

A practical pattern is:

1. anchor with regex
2. inspect expansion output
3. compare against direct retrieval results
4. review the HTML report side by side

This gives you both:

- a deterministic anchor
- and a broader semantic search space


