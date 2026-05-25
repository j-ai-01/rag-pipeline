# RAG Pipeline

A fully local Retrieval-Augmented Generation (RAG) pipeline. Organise documents into named indexes, then query one, several, or all of them at once. No cloud services required — everything runs via [Ollama](https://ollama.com).

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running

Pull the required models:

```bash
ollama pull nomic-embed-text   # embeddings
ollama pull llama3             # answers
ollama pull llava              # image descriptions (only needed for image files)
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start (single index)

```bash
# 1. Create the index data folder and add your files
mkdir -p indexes/default/data
cp your-documents/* indexes/default/data/

# 2. Ingest
python ingest.py

# 3. Query
python query.py "What is the refund policy?"
```

## Working with Multiple Indexes

Create as many named indexes as you need:

```bash
# Finance index
mkdir -p indexes/finance/data
cp finance-reports/* indexes/finance/data/
python ingest.py --index finance

# HR index
mkdir -p indexes/hr/data
cp hr-policies/* indexes/hr/data/
python ingest.py --index hr

# Query one index
python query.py --index finance "What was Q3 revenue?"

# Query two indexes
python query.py --index finance,hr "Who approved the budget?"

# Query all indexes (omit --index)
python query.py "What is the leave policy?"
```

## CLI Reference

### `ingest.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--index <name>` | `default` | Name of the index to ingest into |
| `--reindex` | off | Wipe and re-ingest all files in this index only |

### `query.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--index <names>` | all indexed | Comma-separated index names to query |
| `question` | required | The question to ask (positional, multi-word OK) |

## Index Directory Structure

Each index lives under `indexes/<name>/`:

```
indexes/
  finance/
    data/                   ← put your documents here
    chroma_db/              ← vector embeddings (auto-generated)
    docstore.json           ← node store for auto-merging (auto-generated)
    bm25_index.pkl/         ← keyword index (auto-generated)
    leaf_nodes.pkl          ← leaf node cache (auto-generated)
    .ingested_files.json    ← tracks indexed files (auto-generated)
```

Only the `data/` folder needs to be managed by you. Everything else is created and updated automatically by `ingest.py`.

To delete an index, remove its folder: `rm -rf indexes/<name>/`

## Supported File Types

| Type | Extensions |
|------|-----------|
| Text | `.txt`, `.md` |
| PDF | `.pdf` |
| Word | `.docx` |
| Image | `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp` |

## Migration for Existing Users

If you used a previous version with a flat `data/` folder at the project root:

```bash
mkdir -p indexes/default/data
mv data/* indexes/default/data/
python ingest.py --index default --reindex
```

After verifying the new index works, remove the old root-level artifacts:

```bash
rm -rf chroma_db docstore.json bm25_index.pkl leaf_nodes.pkl .ingested_files.json
```

## Run Tests

```bash
source venv/bin/activate
pytest -v
```
