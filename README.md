# Pattern Nursery

`pattern-nursery` is a CPU-first NLP toolkit for **reviewable** clinical text mining. It combines:

- **transparent** keyword/regex anchoring, and
- **semantic retrieval** (embeddings + vector search)

to help you find relevant **notes, sentences, paragraphs, and phrases**, then generate **static HTML reports** for fast
human review.

The design goal is pragmatic: higher recall than pure regex, without the governance and nondeterminism burden of fully
generative workflows.

---

## Key ideas

- **Auditable outputs**: results include `doc_id`, unit IDs (`sent_id`, `para_id`, `phrase_id`) and character spans (
  `start_char`, `end_char`).
- **Reference-first indexing**: vector store metadata stores references (IDs + spans), not full corpus text. Commands
  reconstruct display text from the corpus.
- **Two corpus formats, equal support** (auto-detected by extension):
    - JSONL (`.jsonl`)
    - SQLite (`.sqlite`, `.sqlite3`, `.db`)
- **Sharding is optional**:
    - JSONL sharding uses record-aligned boundaries stored by `index-corpus`.
    - SQLite sharding uses a `shard_id` column in the corpus DB table (created/populated by `index-corpus` only if you
      request `--total-shards`).

---

## Install

This project uses Python and `uv`.

```bash
uv sync
```

Run the CLI:

```bash
pattern-nursery --help
```

---

## Usage

Note: Find some example workflows in [`EXAMPLE.md`](EXAMPLE.md).

```bash
# 1. index your corpus
pattern-nursery index-corpus --input my_notes.jsonl --work-dir ./clinical_search --total-shards 4

# 2. initialize semantic indexes
pattern-nursery init-index --input my_notes.jsonl --work-dir ./clinical_search --run-name study_v1 --backend sbert --model-path ./models/paraphrase-multilingual-MiniLM-L12-v2

# 3. retrieve relevant notes based on seeds
pattern-nursery retrieve-notes --work-dir ./clinical_search --run-name study_v1 --seeds "Väinämöinen sang" --top-k 100

# 4. generate human-reviewable reports
# (Reports are automatically generated in ./clinical_search/runs/study_v1/reports/)
```

---

## Terminology

- **Work dir**: a directory containing:
    - `vectorstore/` (persistent indexes and configs)
    - `runs/` (per-run outputs: artifacts, logs, reports)
    - `corpus_index/` (JSONL-only helper DB; offset index + shard boundaries)
- **Run name**: a label for outputs under `runs/<run-name>/`.
- **Collections**: indexes you can build and query:
    - `notes`, `sentences`, `paragraphs`, `phrases`, `vocab_terms`

---

## Directory layout

```text
<work-dir>/
  corpus_index/                     # JSONL-only: offset index + shard boundaries
  vectorstore/
    notes/
      config.toml
      shards/                       # only if you build sharded indexes
    sentences/
      config.toml
      shards/
    paragraphs/
      config.toml
      shards/
    phrases/
      config.toml
      shards/
    vocab_terms/
      config.toml
  runs/
    <run-name>/
      artifacts/
      logs/
      reports/
```

---

## Corpus formats (auto-detected)

### A) JSONL corpus (`.jsonl`)

Each physical line is a JSON object with at least:

- `id_col` (default: `id`)
- `text_col` (default: `text`)

Example:

```jsonl
{"id":"doc1","text":"..."}
{"id":"doc2","text":"..."}
```

Newlines inside notes are allowed **as long as the file is valid JSONL** (newlines inside the string are encoded as
`\n`).

### B) SQLite corpus (`.sqlite`, `.sqlite3`, `.db`)

A SQLite corpus DB is expected to have a configurable table (default `documents`) with columns:

- `id_col` (default: `id`)
- `text_col` (default: `text`)

Sharded indexing requires a `shard_id` column. `index-corpus` can create/populate it if you pass `--total-shards`.

---

## Embedding models and `MODEL_PATH`

### What is `MODEL_PATH`?

`MODEL_PATH` is passed via `--model-path` and used by the embedding backend to load an embedding model. It can be:

- a local filesystem path to a model directory, or
- a model identifier that your backend understands (depending on backend implementation).

This project focuses on CPU-friendly sentence embedding models.

### Recommended models (good defaults)

These are commonly used and work well for general-purpose semantic similarity:

- `sentence-transformers/all-MiniLM-L6-v2` (fast, 384 dims)
- `sentence-transformers/all-mpnet-base-v2` (stronger, slower, 768 dims)

If you are indexing large corpora, **384-dim models** are a good operational default.

### How do I get the model locally?

If you use the sentence-transformers backend, the easiest approach is typically:

1. Download/copy the model to a local directory on the machine that will run indexing and retrieval.
2. Pass that directory as `--model-path`.

If your environment already uses a local model cache directory, point `--model-path` there.

### Backend choices

You choose the embedding backend with `--backend`:

- `sbert`: sentence-transformers style models (recommended for sentence/paragraph/note retrieval)
- `fasttext` / `word2vec`: embedding expansion and similarity for token/term-level workflows (varies by model)
- `bm25`: sparse baseline (when available in your build)

---

## Quick start (JSONL)

### 1) Prepare corpus (index-corpus)

For JSONL, `index-corpus` builds a helper SQLite DB under `<work-dir>/corpus_index/` and optionally stores
record-aligned shard boundaries.

```bash
pattern-nursery index-corpus \
  --work-dir work \
  --input data/corpus.jsonl \
  --id-col id --text-col text \
  --total-shards 8
```

Notes:

- `--total-shards` is optional if you are not doing sharded indexing.
- For sharded indexing, keep `--total-shards` stable across shard builds.

### 2) Build indices (single-process)

```bash
pattern-nursery init-index \
  --work-dir work \
  --run-name demo \
  --input data/corpus.jsonl \
  --engine regex \
  --collection notes --collection sentences --collection paragraphs --collection phrases \
  --backend sbert \
  --model-path <MODEL_PATH>
```

### 3) Retrieve and review

```bash
pattern-nursery retrieve-sentences \
  --work-dir work \
  --run-name demo \
  --seeds "chest pain" "shortness of breath" \
  --top-k 15
```

Open the report:

```text
<work-dir>/runs/<run-name>/reports/sentences.html
```

---

## Quick start (SQLite corpus)

### 1) Validate corpus and (optionally) ensure shard_id

```bash
pattern-nursery index-corpus \
  --work-dir work \
  --input data/corpus.sqlite3 \
  --table-name documents \
  --id-col id --text-col text \
  --total-shards 8
```

If you omit `--total-shards`, `index-corpus` only validates that the DB/table/columns exist and does not create/populate
`shard_id`.

### 2) Build indices

```bash
pattern-nursery init-index \
  --work-dir work \
  --run-name demo_sqlite \
  --input data/corpus.sqlite3 \
  --table-name documents \
  --engine regex \
  --collection notes --collection sentences --collection paragraphs --collection phrases \
  --backend sbert \
  --model-path <MODEL_PATH>
```

### 3) Retrieve and review

```bash
pattern-nursery retrieve-paragraphs \
  --work-dir work \
  --run-name demo_sqlite \
  --seeds "dyspnea on exertion" \
  --top-k 15
```

---

## Sharded indexing (FAISS) for large corpora

Sharded indexing lets you parallelize index construction by running one process per shard.

### JSONL: shard boundaries come from index-corpus

```bash
pattern-nursery index-corpus \
  --work-dir work \
  --input data/corpus.jsonl \
  --id-col id --text-col text \
  --total-shards 32

pattern-nursery init-index-shard \
  --work-dir work \
  --run-name buildA \
  --input data/corpus.jsonl \
  --total-shards 32 \
  --shard-id 0 \
  --collection sentences --collection paragraphs \
  --backend sbert \
  --model-path <MODEL_PATH>
```

### SQLite: shard_id must exist (only created/populated if requested)

```bash
pattern-nursery index-corpus \
  --work-dir work \
  --input data/corpus.sqlite3 \
  --table-name documents \
  --id-col id --text-col text \
  --total-shards 32

pattern-nursery init-index-shard \
  --work-dir work \
  --run-name buildA \
  --input data/corpus.sqlite3 \
  --table-name documents \
  --total-shards 32 \
  --shard-id 0 \
  --collection sentences --collection paragraphs \
  --backend sbert \
  --model-path <MODEL_PATH>
```

---

## Expansion commands

Two commands look similar but behave differently:

- `expand-terms`: expands **seed terms** against a **term index** (corpus vocabulary).
- `expand-from-regex`: matches your **regex** in the corpus, then expands from those matches (regex-first workflow).

### `expand-terms`

**Purpose:** Given seed terms, find similar terms in the corpus-derived term index (typically `vocab_terms`).

**Prerequisites:**

- You must have built the `vocab_terms` collection (via `init-index` or shards).

**What it does:**

1. Embeds each seed term.
2. Queries the `vocab_terms` vector store for nearest neighbors.
3. Writes JSONL + HTML report.

**Typical usage:**

```bash
pattern-nursery expand-terms \
  --work-dir work \
  --run-name demo \
  --seeds "myocardial infarction" "heart attack" \
  --top-k 50 \
  --threshold 0.7
```

**Outputs:**

- `runs/<run-name>/artifacts/expansion_terms.jsonl`
- `runs/<run-name>/reports/expansion_terms.html`

---

### `expand-from-regex`

**Purpose:** The “middle path” workflow:

- Use a trusted regex to find anchors in the corpus, then
- retrieve semantically similar sentences (and optionally paragraphs), and
- optionally do term expansion from matched sentences.

**Prerequisites:**

- A corpus input (`--input`) is required because this command runs regex matching over the corpus.
- If you want sentence expansion, you must have built the `sentences` store.
- If you want term expansion, you must have built the `vocab_terms` store.
- If you set `--include-paragraphs`, you must have built the `paragraphs` store.

**What it does (high-level):**

1. Stream the corpus and sentence-split each note.
2. Collect sentences matching any `--regex`.
3. Optionally extract expansion seeds from matched sentences:
    - noun chunks (`--use-noun-chunks`) when supported, otherwise tokens
4. Term expansion (optional): query `vocab_terms` with those seeds.
5. Sentence expansion: query `sentences` using the matched sentences as queries.
6. Paragraph expansion (optional): query `paragraphs` using the matched sentences as queries.
7. Write JSONL + HTML report with provenance.

**Typical usage:**

```bash
pattern-nursery expand-from-regex \
  --work-dir work \
  --run-name demo \
  --input data/corpus.jsonl \
  --regex "heart\\s+failure" \
  --regex "CHF" \
  --engine regex \
  --top-k-terms 50 \
  --top-k-sentences 50 \
  --threshold 0.6 \
  --include-paragraphs
```

**Outputs:**

- `runs/<run-name>/artifacts/regex_expansion.jsonl`
- `runs/<run-name>/reports/regex_expansion.html`

---

## Retrieval commands

Once indexes exist, you can run retrieval:

- `retrieve-notes`
- `retrieve-sentences`
- `retrieve-paragraphs`
- `retrieve-phrases`

All retrieval commands write:

- JSONL results under `runs/<run-name>/artifacts/`
- HTML reports under `runs/<run-name>/reports/`

---

## NLP engines

Choose an engine with `--engine`:

- `regex` (default): deterministic, fast, no external models
- `nltk`: general-purpose
- `spacy`: richer linguistic features (noun chunks), requires a spaCy model

---

## Troubleshooting

### JSONL: “Corpus index missing”

Run `index-corpus` first:

```bash
pattern-nursery index-corpus --work-dir work --input data/corpus.jsonl --total-shards 8
```

### SQLite: “missing table/columns”

Confirm your corpus DB includes `--table-name` and the `--id-col`/`--text-col` columns.

### Common mistakes

- **Skipping `index-corpus` before indexing or retrieval**  
  Run `index-corpus` first so the work directory has the corpus metadata, offsets, and shard information needed by later
  commands.

- **Using the wrong corpus format assumptions**  
  JSONL and SQLite are both supported, but they are handled differently. Make sure the command flags match your corpus
  type.

- **Forgetting to keep `--total-shards` consistent**  
  If you shard a corpus, use the same shard count everywhere. Changing it between steps can lead to mismatched shard
  boundaries or missing results.

- **Pointing `--model-path` to something unavailable on the machine**  
  The embedding backend needs a model it can actually load. If the model is local, confirm the path exists; if it is
  remote, confirm the backend supports that identifier.

- **Expecting semantic retrieval without building the right collection**  
  Retrieval and expansion commands depend on specific collections being indexed first. If a collection was not built,
  the command may run but produce incomplete or empty results.

- **Using the wrong engine for the job**  
  `regex` is deterministic and fast, but it will not provide linguistic features like noun chunks. Use a linguistic
  engine when the workflow needs them.

- **Assuming SQLite shard fields exist automatically**  
  For SQLite corpora, sharded workflows require shard metadata to be present. If you did not request sharding during
  corpus indexing, shard-based commands may not work as expected.

- **Overlooking `run-name` and `work-dir` reuse**  
  Reusing the same run name can overwrite or mix outputs from earlier runs. Use a fresh run name when you want clean
  outputs.

- **Expecting good results from very vague seeds**  
  Semantic search works best with specific seed phrases. Extremely broad terms often return noisy or low-value results.

- **Looking only at JSONL output and missing the HTML report**  
  The HTML report is usually the easiest place to review provenance and context, so check it after each run.
