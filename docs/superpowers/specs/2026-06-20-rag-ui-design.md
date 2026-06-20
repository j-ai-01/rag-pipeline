# RAG Search UI — Design Spec
Date: 2026-06-20

## Overview

Add a minimal browser-based UI to the existing RAG pipeline so the user can query their local knowledge base without using the CLI. The UI is served from the existing FastAPI server (`mcp_server.py`) at `http://localhost:8765`.

## Goals

- Type a question, get a full LLM-generated answer and sources in the browser
- Optionally filter which index(es) to query; no selection = search all
- No new runtime dependencies; reuse existing FastAPI + query pipeline

## Architecture

Single server (`mcp_server.py`) handles both the MCP protocol and the new UI endpoints. A new `ui.html` file is served at the root route so there are no CORS issues.

```
Browser → GET /          → serves ui.html
Browser → GET /indexes   → returns list of index names (JSON)
Browser → POST /query    → runs RAG query, returns answer + sources (JSON)
```

The MCP SSE endpoints (`/sse`, `/messages`) are unchanged.

## New Endpoints

### `GET /`
Returns `ui.html` as an HTML response.

### `GET /indexes`
Returns a JSON array of available index names by calling `list_indexed()`.

```json
["docs", "notes", "reports"]
```

### `POST /query`
Request body:
```json
{
  "question": "What is the refund policy?",
  "indexes": ["reports"]   // or null to query all
}
```

Response body:
```json
{
  "answer": "Refunds are issued within 30 days...",
  "sources": "  - policy.pdf (page 2) [reports]\n  - handbook.docx [reports]"
}
```

On error (e.g., Ollama not running):
```json
{
  "error": "Ollama is not running. Start it with: ollama serve"
}
```

Reuses `build_query_engine()` and `format_sources()` from `query.py`.

## `ui.html` Behaviour

**On page load:**
- `GET /indexes` to populate index chips
- If no indexes exist, show: *"No indexes found — run `python ingest.py --index <name>` first."*

**Index chips:**
- Each chip is a toggle button (highlighted when selected)
- Status line below chips: *"Searching: all indexes"* or *"Searching: docs, reports"*
- No chips selected = `indexes: null` sent to backend = all queried

**Search:**
- Ask button disabled when input is empty
- On submit: Ask button disabled + spinner shown
- `POST /query` with `{question, indexes: selectedList.length ? selectedList : null}`

**Results:**
- Answer text displayed in a result card
- Sources listed below the answer
- On error: error message shown in the result area

## Files Changed

| File | Change |
|---|---|
| `mcp_server.py` | Add `GET /`, `GET /indexes`, `POST /query` endpoints |
| `ui.html` | New file — the browser UI |

## Out of Scope

- Authentication
- Conversation/chat history (single Q&A only)
- Streaming responses
- Ingesting new documents from the UI
