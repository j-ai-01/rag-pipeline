# RAG Pipeline POC

Chat with any document — PDFs, images, text, markdown, Word docs. Fully local. Zero cost.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running

## Setup

```bash
# 1. Pull required models
ollama pull nomic-embed-text
ollama pull llama3
ollama pull llava

# 2. Install Python dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Add your documents

Drop any files into the `data/` folder:
- PDFs (`.pdf`)
- Images (`.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`)
- Word docs (`.docx`)
- Text / Markdown (`.txt`, `.md`)

### 2. Ingest

```bash
python ingest.py
```

This parses, chunks, embeds, and stores all documents in ChromaDB. Re-run anytime you add new files — existing files are skipped automatically.

### 3. Query

```bash
python query.py "What does the report say about Q3 revenue?"
python query.py "Summarise the key points from the meeting notes"
python query.py "What is shown in the diagram?"
```

Example output:
```
Searching for: What does the report say about Q3 revenue?

Answer:
Q3 revenue increased by 23% year-over-year, driven primarily by...

Sources:
  - annual_report.pdf (page 4)
  - q3_summary.md
```

## Run Tests

```bash
pytest -v
```
