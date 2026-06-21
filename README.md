# Recall

A fully local, privacy-first Retrieval-Augmented Generation (RAG) engine. Drop documents into named indexes, ask questions in a browser or via API, and connect any AI assistant via MCP — no cloud, no SaaS, no data leaving your machine.

Designed to scale: one index or a hundred, one user or a team, personal notes or company-wide knowledge bases.

---

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                     mcp_server.py                       │
│                                                         │
│  GET  /          →  Browser UI                          │
│  GET  /indexes   →  List available indexes              │
│  POST /query     →  Single-shot answer (JSON)           │
│  POST /query/stream → Streaming answer (SSE)            │
│  GET  /sse       →  MCP SSE transport                   │
│  POST /messages  →  MCP tool calls                      │
└─────────────────────────────────────────────────────────┘
         │                          │
    ReActAgent                  MCP Tools
  (any Ollama LLM)          (list_indexes, query_rag)
         │                          │
   RetrieverTool              build_retriever()
   (KB lookup)               (raw chunks → your LLM)
         │
   Hybrid Retrieval
   (BM25 + vector + AutoMerging)
```

**Agent flow:** The model sees your question, decides whether the knowledge base needs to be searched, calls it as a tool if so, then synthesises a comprehensive answer. Greetings and general questions are handled directly without touching documents.

**Session memory:** Each browser tab gets its own `session_id`. The agent carries full conversation history for the lifetime of the server — follow-up questions, pronoun references, and multi-turn context all work.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running

Pull the models you need:

```bash
ollama pull nomic-embed-text   # embeddings (required)
ollama pull gemma4             # LLM for the agent (recommended)
ollama pull llava              # image descriptions (optional)
```

You can swap any Ollama-compatible LLM — see [Configuration](#configuration).

---

## Installation

```bash
git clone <this-repo>
cd recall

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Quick Start

```bash
# 1. Create an index and add documents
mkdir -p indexes/my-docs/data
cp /path/to/your/documents/* indexes/my-docs/data/

# 2. Ingest
python ingest.py --index my-docs

# 3. Start the server
python mcp_server.py

# 4. Open the browser UI
open http://localhost:8765
```

That's it. Ask questions in the browser.

---

## Working with Indexes

Indexes are independent knowledge bases. Create as many as you need — by topic, team, project, or client.

```bash
# Create and ingest multiple indexes
mkdir -p indexes/finance/data   && cp finance-reports/*  indexes/finance/data/
mkdir -p indexes/legal/data     && cp contracts/*         indexes/legal/data/
mkdir -p indexes/engineering/data && cp tech-docs/*       indexes/engineering/data/

python ingest.py --index finance
python ingest.py --index legal
python ingest.py --index engineering

# Re-ingest after adding or changing files
python ingest.py --index finance --reindex

# Delete an index entirely
rm -rf indexes/finance
```

### Index directory layout

```
indexes/
  <name>/
    data/                   ← your documents go here (you manage this)
    chroma_db/              ← vector store        (auto-generated)
    docstore.json           ← node store          (auto-generated)
    bm25_index.pkl/         ← keyword index       (auto-generated)
    leaf_nodes.pkl          ← retrieval cache     (auto-generated)
    .ingested_files.json    ← change tracking     (auto-generated)
```

Only `data/` needs to be managed. Everything else is generated on ingest.

### Supported file types

| Type  | Extensions |
|-------|-----------|
| Text  | `.txt`, `.md` |
| PDF   | `.pdf` |
| Word  | `.docx` |
| Image | `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp` |

---

## Browser UI

Open `http://localhost:8765` after starting the server.

- **Streaming answers** — tokens appear as the model generates them
- **Agent-driven retrieval** — model decides when to search; small talk skips the KB
- **Session memory** — full conversation context within a browser tab
- **Source snippets** — retrieved chunks shown with preview and "show more" toggle
- **Index filter chips** — click to target specific indexes; none selected = search all
- **Chat history** — previous Q&As stack above, dimmed
- **Markdown rendering** — headers, bullet points, code blocks (syntax-highlighted), Mermaid diagrams
- **Light / dark mode** — toggle in the top bar, saved to `localStorage`

---

## CLI Queries

```bash
# Query a single index
python query.py --index finance "What was Q3 revenue?"

# Query multiple indexes at once
python query.py --index finance,legal "Who signed the budget approval?"

# Query everything (no --index flag = searches all indexes)
python query.py "What is the leave policy?"
```

---

## Connect via MCP

Recall exposes an MCP server over SSE. Any MCP-compatible AI client can connect to it and search your knowledge base as a tool.

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "recall": {
      "url": "http://localhost:8765/sse"
    }
  }
}
```

Restart Claude Desktop. It will have access to:

| Tool | Description |
|------|-------------|
| `list_indexes` | List all ingested indexes |
| `query_rag` | Search the knowledge base and return relevant passages |

### Claude Code

```bash
claude mcp add recall --transport sse http://localhost:8765/sse
```

### Any other MCP client

Point your client at `http://localhost:8765/sse` (SSE transport) or `http://localhost:8765/messages` (POST endpoint for tool calls).

---

## API Reference

### `GET /indexes`

Returns a list of all ingested index names.

```bash
curl http://localhost:8765/indexes
# ["finance", "legal", "engineering"]
```

### `POST /query`

Single-shot Q&A. The agent decides whether to call the KB.

```bash
curl -X POST http://localhost:8765/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What were the key risks identified in the Q3 audit?",
    "indexes": ["finance", "legal"],
    "session_id": "user-abc-tab-1"
  }'
```

```json
{
  "answer": "...",
  "sources": "  - audit-q3.pdf (page 4) [finance]",
  "snippets": [
    {
      "filename": "audit-q3.pdf",
      "page": 4,
      "index": "finance",
      "text": "..."
    }
  ],
  "session_id": "user-abc-tab-1"
}
```

- `indexes` — optional; omit to search all indexes
- `session_id` — optional; omit to start a fresh session (one is returned in the response)

### `POST /query/stream`

Same as `/query` but streams tokens as they are generated (`text/event-stream`).

Event sequence:

```
data: {"type": "token",   "text": "..."}      ← one per token
data: {"type": "sources", "snippets": [...]}  ← after generation (if KB was used)
data: {"type": "done"}
data: {"type": "error",   "message": "..."}   ← on failure (replaces above)
```

---

## Configuration

Edit `config.py` to customise the engine:

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MODEL` | `gemma4` | Ollama model used by the agent |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model for vector search |
| `TOP_K` | `5` | Number of chunks retrieved per query |
| `PARENT_CHUNK_SIZE` | `1024` | Parent chunk size (tokens) |
| `CHILD_CHUNK_SIZE` | `256` | Child chunk size for auto-merging retrieval |
| `HYBRID_ALPHA` | `0.5` | BM25 vs vector blend (`0` = BM25 only, `1` = vector only) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |

**Swap the LLM** — any model available in Ollama works:

```python
# config.py
LLM_MODEL = "llama3.2"   # lighter
LLM_MODEL = "mistral"    # fast
LLM_MODEL = "gemma4"     # best instruction-following (recommended)
```

---

## Running Tests

```bash
source venv/bin/activate
pytest -v
```

---

## Deployment Notes

- **Single user / local:** run `python mcp_server.py` as-is
- **Team / shared server:** run behind a reverse proxy (nginx, Caddy) and bind to `0.0.0.0` (already the default); add auth at the proxy layer if needed
- **Multiple environments:** each environment maintains its own `indexes/` directory; indexes are fully portable — copy the folder to move them
- **Port:** default is `8765`; change by passing `--port` to uvicorn or editing the `__main__` block in `mcp_server.py`

---

## Migration from Earlier Versions

If you used a previous version with a flat `data/` folder at the project root:

```bash
mkdir -p indexes/default/data
mv data/* indexes/default/data/
python ingest.py --index default --reindex
rm -rf chroma_db docstore.json bm25_index.pkl leaf_nodes.pkl .ingested_files.json
```
