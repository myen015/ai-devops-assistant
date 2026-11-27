# AI DevOps & Incident Response Assistant

![AI Service CI](https://github.com/myen015/ai-devops-assistant/actions/workflows/ci.yml/badge.svg)

A production-ready, local-first Applied AI microservice combining **Structured Log Analysis**, **Hybrid RAG with Cross-Encoder Reranking**, **Automated AI Evaluations**, **Stateful & Multi-Agent Orchestration**, and an **MCP 2.x Server** — powered by FastAPI, Qdrant, LiteLLM, and Ollama (`qwen2.5:3b`).

## Architecture Overview

```text
Client / Swagger UI / MCP Client (Cursor, Claude Desktop)
        │
        ├── REST API (FastAPI + Observability & X-API-Key Middleware)
        │     ├── POST /analyze       -> Structured Log Analyzer (Pydantic + LiteLLM)
        │     ├── POST /ingest(-pdf)  -> Page-Aware PDF/Text Chunker -> Qdrant Vector Store
        │     ├── POST /query         -> Hybrid Search (Dense + BM25 RRF) -> Cross-Encoder Reranker -> Grounded QA
        │     ├── POST /agent         -> Stateful Tool-Calling Agent (Session Memory + Loop Guardrails)
        │     └── POST /incident      -> Multi-Agent Pipeline (Triage -> Researcher -> Responder)
        │
        └── MCP Server (mcp 2.x MCPServer)
              ├── search_docs_tool
              ├── analyze_logs_tool
              └── resolve_incident_tool
```

## Key Capabilities

1. **Structured Log Analysis (`analyzer.py`)**
   * Parses raw server and application logs into strictly validated Pydantic schemas (`LogReport`) containing categorized errors, warnings, and plain-text summaries.
   * Enforces zero-hallucination behavior by returning empty issue lists on clean `INFO` logs.

2. **Hybrid RAG & Cross-Encoder Reranking (`rag_store.py`, `reranker.py`, `pdf_loader.py`, `rag_qa.py`)**
   * **Ingestion:** Splits multi-page PDFs (`pypdf`) and technical docs into chunks with exact `source` and `page` metadata stored in local Qdrant.
   * **Two-Stage Retrieval:** Combines Dense Vector embeddings (`FastEmbed`) and BM25 lexical matching (`rank-bm25`) via Reciprocal Rank Fusion (RRF), followed by Cross-Encoder reranking.
   * **Grounding & Citations:** Answers exclusively from retrieved context with exact file/page citations; returns `"Not found in provided documents."` when evidence is absent.

3. **Automated AI Evaluations (`eval_rag.py`)**
   * Separates retrieval quality (**Retrieval Recall@3: 100%**) from end-to-end generation accuracy (**LLM-as-a-Judge 1–5 scoring: 5/5**).

4. **Stateful & Multi-Agent Workflows (`agent.py`, `multi_agent.py`)**
   * **Stateful Agent:** Autonomous tool selection with `session_id` conversation memory and repeated-call loop guardrails.
   * **Multi-Agent Incident Pipeline:** 3-stage orchestration (**Triage → Researcher → Responder**) producing prioritized `P1–P3` reports with root causes, action steps, and citations.

5. **MCP 2.x Server (`mcp_server.py`)**
   * Exposes core engineering tools via the official `mcp` v2 (`MCPServer`) standard for seamless integration with Cursor, Claude Desktop, and other MCP clients.

6. **Production Readiness (`observability.py`, `Dockerfile`, `.github/workflows/ci.yml`)**
   * Structured JSON request logging, `X-Request-ID` tracing, `X-Process-Time-Ms` headers, `/metrics` latency tracking, optional `X-API-Key` authentication, non-root Docker setup, and GitHub Actions CI.

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health check (always public) |
| `GET` | `/metrics` | Request counts and average latency per route |
| `POST` | `/analyze` | Parse raw logs into structured errors/warnings |
| `POST` | `/ingest` | Index raw text into Qdrant with source/page metadata |
| `POST` | `/ingest-pdf` | Upload and index multi-page PDF documents |
| `GET` | `/sources` | List all indexed documents and page coverage |
| `DELETE` | `/sources/{source_name}` | Remove all chunks belonging to a source |
| `POST` | `/query` | Hybrid RAG question answering with file/page citations |
| `POST` | `/agent` | Stateful tool-calling agent with `session_id` memory |
| `POST` | `/incident` | Multi-agent `P1–P3` incident response pipeline |

## Quickstart (Local)

### Prerequisites
* Python 3.11+
* [Ollama](https://ollama.com/) running locally with `qwen2.5:3b`:
  ```bash
  ollama pull qwen2.5:3b
  ollama serve
  ```

### Setup & Run
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: enable API key protection
export API_KEY="your-secret-key"

uvicorn api:app --reload
```
Open Swagger UI at `http://127.0.0.1:8000/docs`.

### Docker Compose
Runs the FastAPI container connected directly to the host's Ollama instance (preserves Apple Silicon / host GPU acceleration):
```bash
docker compose up --build
```

## Testing & AI Evaluations

```bash
# Fast unit, API, security, observability & MCP tests (CI mode, no Ollama needed)
pytest -v -m "not live_llm"

# Full test suite (requires running Ollama)
pytest -v

# RAG Retrieval Recall@3 + LLM-as-a-Judge evaluation
python eval_rag.py
```

## Running the MCP Server

```bash
python mcp_server.py
```

