# RAG Pipeline

A fully local Retrieval-Augmented Generation (RAG) pipeline with an agentic query engine, streaming web UI, and MCP server. Organise documents into named indexes, query them via browser or API, and connect to Claude Desktop / Claude Code via MCP — no cloud services required.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     mcp_server.py                       │
│                                                         │
│  GET  /          →  Browser UI (ui.html)                │
│  GET  /indexes   →  List available indexes              │
│  POST /query     →  Single-shot answer (JSON)           │
│  POST /query/stream → Streaming answer (SSE)            │
│  GET  /sse       →  MCP SSE transport (Claude Desktop)  │
│  POST /messages  →  MCP tool calls                      │
└─────────────────────────────────────────────────────────┘
         │                          │
    ReActAgent                  MCP Tools
   (gemma4 local)           (list_indexes, query_rag)
         │                          │
   RetrieverTool              build_retriever()
   (KB lookup)               (raw chunks → Claude)
         │
   Hybrid Retrieval
   (BM25 + vector + AutoMerging)
```

**Agent flow:** The model (`gemma4`) sees your question, decides whether the knowledge base is needed, calls it as a tool if so, then synthesises a comprehensive answer. Greetings and general questions are handled directly without touching the KB.

**Session memory:** Each browser tab keeps its own `session_id`. The agent accumulates full conversation history for the duration of the server process — follow-up questions, pronoun references, and context all work.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running

```bash
ollama pull nomic-embed-text   # embeddings
ollama pull gemma4             # agent LLM (recommended — strong instruction following)
ollama pull llava              # image descriptions (optional)
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Add documents to an index
mkdir -p indexes/my-docs/data
cp your-documents/* indexes/my-docs/data/

# 2. Ingest
python ingest.py --index my-docs

# 3. Start the server
python mcp_server.py

# 4. Open the UI
open http://localhost:8765
```

## Browser UI

Open `http://localhost:8765` after starting `mcp_server.py`.

Features:
- **Streaming answers** — tokens appear as the model generates them
- **Agent-driven retrieval** — model decides when to search the KB; greetings and off-topic questions are answered directly
- **Session memory** — the agent remembers the full conversation within a browser tab
- **Source snippets** — retrieved chunks shown with preview and "show more" toggle
- **Chat history** — previous Q&As stack above, dimmed
- **Markdown rendering** — headers, bullet points, code blocks (syntax-highlighted), Mermaid diagrams
- **Light / dark mode** — toggle in the top bar, saved to `localStorage`
- **Index filter chips** — click to query specific indexes; none selected = search all

## Connect to Claude Desktop (MCP)

Add the following to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "rag-pipeline": {
      "url": "http://localhost:8765/sse"
    }
  }
}
```

Make sure `python mcp_server.py` is running first. Claude Desktop will then have access to two tools:

| Tool | Description |
|------|-------------|
| `list_indexes` | List all ingested indexes |
| `query_rag` | Search the KB and return raw chunks (Claude synthesises the answer) |

## Connect to Claude Code (MCP)

Add to your project's `.mcp.json` or run:

```bash
claude mcp add rag-pipeline --transport sse http://localhost:8765/sse
```

## Working with Multiple Indexes

```bash
# Create indexes
mkdir -p indexes/finance/data && cp finance-reports/* indexes/finance/data/
python ingest.py --index finance

mkdir -p indexes/hr/data && cp hr-policies/* indexes/hr/data/
python ingest.py --index hr

# CLI queries
python query.py --index finance "What was Q3 revenue?"
python query.py --index finance,hr "Who approved the budget?"
python query.py "What is the leave policy?"   # searches all indexes
```

## API Reference

### `POST /query`

Single-shot Q&A. Uses the ReActAgent — model decides whether to call the KB.

```bash
curl -X POST http://localhost:8765/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Jai career?", "indexes": ["docs"], "session_id": "abc123"}'
```

Response:
```json
{
  "answer": "...",
  "sources": "  - doc.pdf",
  "snippets": [{"filename": "doc.pdf", "page": 1, "index": "docs", "text": "..."}],
  "session_id": "abc123"
}
```

### `POST /query/stream`

Same as `/query` but returns `text/event-stream`. Event order:

```
data: {"type": "token", "text": "..."}   ← one per token during generation
data: {"type": "sources", "snippets": [...]}  ← after generation (if KB was used)
data: {"type": "done"}
data: {"type": "error", "message": "..."}   ← instead of above on failure
```

## CLI Reference

### `ingest.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--index <name>` | `default` | Name of the index to ingest into |
| `--reindex` | off | Wipe and re-ingest all files in this index |

### `query.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--index <names>` | all indexed | Comma-separated index names to query |
| `question` | required | The question (positional, multi-word OK) |

## Configuration (`config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MODEL` | `gemma4` | Ollama model for the agent |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `TOP_K` | `5` | Chunks retrieved per query |
| `PARENT_CHUNK_SIZE` | `1024` | Parent chunk size (tokens) |
| `CHILD_CHUNK_SIZE` | `256` | Child chunk size for auto-merging |
| `HYBRID_ALPHA` | `0.5` | BM25 vs vector blend (0 = BM25 only, 1 = vector only) |

## Index Directory Structure

```
indexes/
  <name>/
    data/                   ← put your documents here
    chroma_db/              ← vector embeddings (auto-generated)
    docstore.json           ← node store for auto-merging (auto-generated)
    bm25_index.pkl/         ← keyword index (auto-generated)
    leaf_nodes.pkl          ← leaf node cache (auto-generated)
    .ingested_files.json    ← tracks indexed files (auto-generated)
```

Only `data/` needs to be managed by you. Delete an index with `rm -rf indexes/<name>/`.

## Supported File Types

| Type | Extensions |
|------|-----------|
| Text | `.txt`, `.md` |
| PDF | `.pdf` |
| Word | `.docx` |
| Image | `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp` |

## Run Tests

```bash
source venv/bin/activate
pytest -v
```

## Migration from Earlier Versions

If you used a previous version with a flat `data/` folder at the project root:

```bash
mkdir -p indexes/default/data
mv data/* indexes/default/data/
python ingest.py --index default --reindex
rm -rf chroma_db docstore.json bm25_index.pkl leaf_nodes.pkl .ingested_files.json
```
