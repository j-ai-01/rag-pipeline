import json
import sys
import io
import os
import warnings
import logging

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("chromadb").setLevel(logging.ERROR)


class _StderrFilter(io.TextIOWrapper):
    _blocked = ("Failed to send telemetry event",)

    def write(self, msg):
        if not any(s in msg for s in self._blocked):
            sys.__stderr__.write(msg)
        return len(msg)

    def flush(self):
        sys.__stderr__.flush()


sys.stderr = _StderrFilter(io.BytesIO())

from pathlib import Path
from typing import Any, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp import types

from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding

from pydantic import BaseModel, field_validator

from config import EMBED_MODEL, OLLAMA_BASE_URL
from query import list_indexed, build_retriever, build_query_engine, format_sources
from utils.ollama_check import check_ollama_running

app = FastAPI(title="RAG Pipeline MCP Server")


@app.get("/")
async def serve_ui():
    return FileResponse(str(Path(__file__).parent / "ui.html"), media_type="text/html")


@app.get("/indexes")
async def get_indexes() -> List[str]:
    return list_indexed()


class QueryRequest(BaseModel):
    question: str
    indexes: Optional[List[str]] = None

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty")
        return v.strip()


def extract_snippets(source_nodes) -> List[dict]:
    snippets = []
    seen = set()
    for node in source_nodes:
        meta = node.metadata
        filename = meta.get("filename", "unknown")
        page = meta.get("page_number")
        index_name = meta.get("index_name", "")
        key = f"{filename}:{page}:{index_name}"
        if key not in seen:
            seen.add(key)
            snippets.append({
                "filename": filename,
                "page": page,
                "index": index_name,
                "text": node.node.get_content()[:300],
            })
    return snippets


@app.post("/query")
async def query_endpoint(req: QueryRequest):
    if not check_ollama_running():
        return JSONResponse(
            status_code=503,
            content={"error": "Ollama is not running. Start it with: ollama serve"},
        )

    index_names = req.indexes if req.indexes is not None else list_indexed()
    if not index_names:
        return JSONResponse(
            status_code=404,
            content={"error": "No indexes found. Run `python ingest.py --index <name>` first."},
        )

    try:
        engine = build_query_engine(index_names)
        response = engine.query(req.question)
        answer = response.response or "Could not generate a summary. See sources below."
        sources = format_sources(response.source_nodes)
        snippets = extract_snippets(response.source_nodes)
        return {"answer": answer, "sources": sources, "snippets": snippets}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


server = Server("rag-pipeline")
sse = SseServerTransport("/messages")


def retrieve_chunks(question: str, index_names: Optional[List[str]] = None) -> List[dict]:
    if index_names is None:
        index_names = list_indexed()
    if not index_names:
        return []

    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    chunks = []
    for name in index_names:
        try:
            retriever = build_retriever(name)
            nodes = retriever.retrieve(question)
            for node in nodes:
                chunks.append({
                    "text": node.node.get_content(),
                    "source": node.node.metadata.get("filename", "unknown"),
                    "index": name,
                    "score": round(float(node.score or 0), 4),
                })
        except FileNotFoundError as e:
            chunks.append({"error": str(e), "index": name})
        except Exception as e:
            chunks.append({"error": f"Retrieval failed for index '{name}': {e}", "index": name})

    return sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="list_indexes",
            description="List all available RAG knowledge base indexes that have been ingested.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="query_rag",
            description=(
                "Search the local knowledge base and return relevant text passages. "
                "Returns raw chunks — you synthesise the answer. "
                "Requires Ollama to be running locally."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question or search query.",
                    },
                    "indexes": {
                        "type": "string",
                        "description": "Comma-separated index names to search (default: all indexes).",
                    },
                },
                "required": ["question"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    if name == "list_indexes":
        result = list_indexed()
        return [types.TextContent(type="text", text=json.dumps(result))]

    if name == "query_rag":
        question = arguments["question"]
        indexes_str = arguments.get("indexes")
        index_names = [n.strip() for n in indexes_str.split(",")] if indexes_str else None
        try:
            chunks = retrieve_chunks(question, index_names)
        except Exception as e:
            return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]
        return [types.TextContent(type="text", text=json.dumps(chunks))]

    return [types.TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


@app.get("/sse")
async def handle_sse(request: Request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await server.run(
            streams[0],
            streams[1],
            server.create_initialization_options(),
        )


@app.post("/messages")
async def handle_messages(request: Request):
    await sse.handle_post_message(request.scope, request.receive, request._send)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
