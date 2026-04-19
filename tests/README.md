# Pattern Nursery Test Suite

This directory contains the test suite for `pattern_nursery`.

## Structure

The tests are organized into the following categories:

- **unit/**: Isolated tests for individual components (NLP engines, IO utilities, etc.).
- **integration/**: Tests that exercise multiple components together, including CLI workflows and full indexing
  pipelines.
- **storage/**: Tests specifically targeting corpus storage backends (SQLite, CSV) and indexing mechanisms.
- **regression/**: Tests that reproduce past issues to ensure they don't reappear.
- **helpers/**: Shared test utilities and corpus builders.
- **data/**: Static data files used by tests.

## Adding New Tests

- Use **unit tests** for new logic in small functions or classes.
- Use **integration tests** for new CLI commands or complex workflows.
- Place reproduction tests for bugs in **regression/**.
- Follow the existing style: use Finnish mythological names (Kalevala) for test data where applicable.
- Avoid adding `__init__.py` files to test directories.

## Optional Dependencies

Some tests require optional dependencies (e.g., `spacy`, `sentence-transformers`).

- Tests for `ImportError` paths should always run.
- Tests requiring the actual dependency should be marked to skip if the dependency is not installed.

### Tests for `sentence-transformers`

- Copy `.pytest.toml` to `pytest.toml`
- Fill in LOCAL (i.e., downloaded) location of `sentence_transformers` model files
- Now run `pytest` to execute the tests with `pytest` command.
