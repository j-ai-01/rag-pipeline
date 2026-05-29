# MCP Server for RAG Pipeline

**Date:** 2026-05-30  
**Status:** Approved

## Overview

Add an HTTP-based MCP server to the RAG pipeline so any MCP-compatible LLM client (Claude Code, Codex, Cursor, Gemini CLI, etc.) can query local knowledge base indexes as a tool. The server exposes two tools: `list_indexes` and `query_rag`.

## Goals

- Any MCP-compatible client can discover and search indexes via a local URL
- Claude (or any other LLM) does its own synthesis from raw retrieved chunks — Ollama is used only for embeddings, not generation
- Manual start — no daemon or launchd required
- Minimal new code: one new file, three new dependencies

## Non-Goals

- Ingest/write operations via MCP (out of scope for this phase)
- Authentication (local personal use only)
- Auto-start on boot

## Architecture

```
MCP Client (Claude / Codex / Cursor)
        │
        │  HTTP/SSE   http://localhost:8765/sse
        ▼
  mcp_server.py  (FastAPI + mcp[cli] SSE transport)
        │
        ├── list_indexes()     → query.list_indexed()
        └── query_rag()        → build_retriever() × N indexes
                                  → retriever.retrieve(question)
                                  → raw NodeWithScore chunks
```

The server reuses existing functions from `query.py` directly — `list_indexed()` and `build_retriever()`. No duplication of retrieval logic.

## Files Changed

| File | Change |
|------|--------|
| `mcp_server.py` | New — FastAPI app with MCP SSE transport and two tools |
| `requirements.txt` | Add `mcp[cli]`, `fastapi`, `uvicorn` |

## Tools

### `list_indexes`

- **Arguments:** none
- **Returns:** JSON array of strings — names of all ingested indexes
- **Example response:** `["default", "persona", "work"]`
- **Error:** Returns empty array if no indexes exist

### `query_rag`

- **Arguments:**
  - `question` (string, required) — the query to run
  - `indexes` (string, optional) — comma-separated index names; defaults to all indexed
- **Returns:** JSON array of chunk objects, sorted by relevance score descending
- **Chunk shape:**
  ```json
  {
    "text": "...",
    "source": "bio.md",
    "index": "persona",
    "score": 0.91
  }
  ```
- **Error:** Returns error string if Ollama is not running or no indexes found

## Data Flow (query_rag)

1. Client sends MCP tool call with `question` and optional `indexes`
2. Server resolves index list (explicit or all via `list_indexed()`)
3. Initialises Ollama embedding model (no LLM — retrieval only)
4. Calls `build_retriever(name)` for each index
5. Calls `retriever.retrieve(question)` to get `NodeWithScore` objects
6. Formats and returns raw chunks as JSON — no generation step
7. Client LLM synthesises the answer from the chunks

## Start Command

```bash
cd /Users/jaibhambri/workplace/rag-pipeline
source venv/bin/activate
uvicorn mcp_server:app --port 8765
```

## Client Configuration

### Claude Code (global)

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "rag-pipeline": {
      "url": "http://localhost:8765/sse"
    }
  }
}
```

### Other clients (Codex, Cursor, Gemini CLI, etc.)

Point their MCP config to the same URL: `http://localhost:8765/sse`

## Dependencies

| Package | Purpose |
|---------|---------|
| `mcp[cli]` | MCP server + SSE transport |
| `fastapi` | HTTP framework hosting the SSE endpoint |
| `uvicorn` | ASGI server to run FastAPI |

All other retrieval dependencies (`llama-index`, `chromadb`, `ollama`) are already in `requirements.txt`.

## Error Handling

- **Ollama not running:** `query_rag` returns an error string; server stays up
- **Index not found:** that index is skipped with a warning; others proceed
- **No indexes at all:** `query_rag` returns an error string
- **Retrieval exception:** caught per-index, error included in response
