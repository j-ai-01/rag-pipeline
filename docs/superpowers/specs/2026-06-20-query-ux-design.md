# Query UX Enhancements — Design Spec
Date: 2026-06-20

## Overview

Upgrade the RAG Search UI with streaming responses, source text snippets, chat history, light/dark mode, and rich markdown/code/diagram rendering — all while keeping the existing `POST /query` endpoint and its tests intact.

## Goals

- Answers stream token-by-token so the UI feels fast instead of freezing for 30s
- Mermaid diagrams, syntax-highlighted code blocks, and markdown render correctly in answers
- Chat history stacks above so previous Q&As stay visible
- Light/dark mode toggle saved to `localStorage`
- Source cards show a 2-3 sentence text snippet with "show more" toggle, not just filenames

## Architecture

Three files change:

| File | Change |
|---|---|
| `mcp_server.py` | Add `POST /query/stream` SSE endpoint; add `snippets` array to existing `POST /query` response |
| `ui.html` | Full rewrite — chat history layout, streaming, CDN libraries, light/dark |
| `tests/test_ui_endpoints.py` | New tests for snippet shape and stream event sequence |

The existing `POST /query` remains fully intact. All 10 current tests continue to pass.

## Backend Changes

### `POST /query` — add snippets

Extend the existing response to include a `snippets` array alongside `answer` and `sources`:

```json
{
  "answer": "...",
  "sources": "  - policy.pdf (page 2) [docs]",
  "snippets": [
    {
      "filename": "policy.pdf",
      "page": 2,
      "index": "docs",
      "text": "Refunds are issued within 30 days of purchase. The customer must..."
    }
  ]
}
```

`text` is the first 300 characters of `node.node.get_content()` for each source node (deduplicated by filename+page).

### `POST /query/stream` — new SSE endpoint

Accepts the same `QueryRequest` body as `POST /query`. Returns `text/event-stream`.

Event sequence:
1. `{"type": "sources", "snippets": [...]}` — sent immediately after retrieval, before LLM starts
2. `{"type": "token", "text": "..."}` — one event per LLM token
3. `{"type": "done"}` — signals completion
4. `{"type": "error", "message": "..."}` — sent instead of the above if something fails

The streaming query engine is built with `RetrieverQueryEngine.from_args(..., streaming=True)`. Token iteration uses `response.response_gen`.

CORS is not needed — the UI is served from the same FastAPI origin.

## Frontend — `ui.html` rewrite

### Layout

```
┌─────────────────────────────────────────────┐
│ topbar: "RAG Search"    model info  ☀️/🌙   │
├─────────────────────────────────────────────┤
│                                             │
│  [past Q&A — dimmed]                        │
│    Q: ...                                   │
│    ┌─ Answer ──────┐ ┌─ Sources ──────┐    │
│    │ markdown text │ │ snippet cards  │    │
│    └───────────────┘ └────────────────┘    │
│                                             │
│  [current Q&A]                              │
│    Q: ...                                   │
│    ┌─ Answer ──────┐ ┌─ Sources ──────┐    │
│    │ streaming...▌ │ │ shown already  │    │
│    └───────────────┘ └────────────────┘    │
│                                             │
├─────────────────────────────────────────────┤
│ index chips  ·  search bar  [Ask]           │
└─────────────────────────────────────────────┘
```

- Chat history scrolls; search bar and index chips are fixed at the bottom
- Past Q&As are dimmed (opacity 0.6) once a new question is asked
- Auto-scroll to bottom when new content arrives

### Rendering libraries (CDN, no build step)

| Library | Purpose | CDN |
|---|---|---|
| `marked.js` | Markdown → HTML | jsDelivr |
| `highlight.js` | Syntax highlighting for code fences | jsDelivr |
| `mermaid.js` | Render ` ```mermaid ``` ` blocks as SVG | jsDelivr |

Rendering pipeline: accumulate streamed tokens → on `done`, pass full answer text to `marked.js` → replace placeholder with rendered HTML → `highlight.js` colours code blocks → `mermaid.js` renders diagrams.

During streaming, raw text is shown in a `<pre>` so it's readable. Final render happens once on `done`.

### Source snippets

Each source card shows:
- Filename + page/index tag
- First 2-3 sentences of the chunk text
- "show more ▾" toggle to expand the full chunk

### Light / dark mode

CSS custom properties on `:root` and `[data-theme="light"]`. A toggle button in the topbar switches the `data-theme` attribute on `<html>` and saves to `localStorage`. Default is dark.

### Streaming implementation

Uses `fetch()` with `ReadableStream` (not `EventSource`, which doesn't support POST):

```js
const res = await fetch('/query/stream', { method: 'POST', ... });
const reader = res.body.getReader();
// read chunks, split on \n\n, parse JSON from "data: {...}" lines
```

## Error Handling

| Scenario | Behaviour |
|---|---|
| Ollama not running | `{"type": "error", "message": "Ollama is not running..."}` — shown in the answer area |
| No indexes | `{"type": "error", "message": "No indexes found..."}` — shown inline |
| Stream drops mid-answer | UI shows received text + "⚠ Connection lost" notice |
| Malformed SSE event | Skipped silently; stream continues |

## Testing

New tests in `tests/test_ui_endpoints.py`:

1. `test_post_query_returns_snippets` — asserts `snippets` key in response, each item has `filename`, `text`
2. `test_post_query_stream_event_sequence` — consumes SSE response, asserts first event type is `sources`, subsequent events include `token` type, final event type is `done`
3. `test_post_query_stream_error_when_ollama_not_running` — asserts error event when Ollama check fails

## Out of Scope

- Persisting chat history across page reloads (session memory)
- Cancelling an in-flight stream
- Copy-to-clipboard button on code blocks (nice-to-have, deferred)
- Claude session auto-indexing (separate spec)
- Ingest from UI (separate spec — Group B)
