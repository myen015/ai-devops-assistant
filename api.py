from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from analyzer import LogReport, analyze_logs
from pdf_loader import ingest_pdf_bytes
from rag_qa import RagAnswer, ask_docs
from rag_store import ingest_document
from rag_store import delete_source, ingest_document, list_sources
from agent import AgentResponse, run_agent
from multi_agent import IncidentReport, run_incident_pipeline
from observability import get_metrics_snapshot, observability_and_auth_middleware

app = FastAPI(title="Applied AI Engineering Service")
app.middleware("http")(observability_and_auth_middleware)


class LogRequest(BaseModel):
    log_text: str = Field(..., min_length=5)
    model: str = Field(default="ollama/qwen2.5:3b")


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=5)
    source_name: str = Field(..., min_length=1)
    page: int = Field(default=1, ge=1)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    model: str = Field(default="ollama/qwen2.5:3b")


class AgentRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=2000)
    model: str = Field(default="ollama_chat/qwen2.5:3b")
    session_id: str | None = Field(default=None)


# Pydantic модельдеріне қос:
class IncidentRequest(BaseModel):
    description: str = Field(..., min_length=5, max_length=3000)
    model: str = Field(default="ollama/qwen2.5:3b")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/analyze", response_model=LogReport)
def analyze_endpoint(payload: LogRequest) -> LogReport:
    return analyze_logs(log_text=payload.log_text, model=payload.model)


@app.post("/ingest")
def ingest_endpoint(payload: IngestRequest):
    chunks_added = ingest_document(
        text=payload.text,
        source_name=payload.source_name,
        page=payload.page,
    )
    return {"status": "indexed", "chunks_added": chunks_added, "source": payload.source_name}


@app.post("/ingest-pdf")
async def ingest_pdf_endpoint(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported")

    content = await file.read()
    result = ingest_pdf_bytes(file_bytes=content, source_name=file.filename)
    return {"status": "indexed", **result}


@app.post("/query", response_model=RagAnswer)
def query_endpoint(payload: QueryRequest) -> RagAnswer:
    return ask_docs(question=payload.question, model=payload.model)


@app.get("/sources")
def get_sources_endpoint():
    return {"sources": list_sources()}


@app.delete("/sources/{source_name}")
def delete_source_endpoint(source_name: str):
    deleted = delete_source(source_name)
    return {"status": "deleted" if deleted else "not_found", "source": source_name}


@app.post("/agent", response_model=AgentResponse)
def agent_endpoint(payload: AgentRequest) -> AgentResponse:
    return run_agent(
        user_prompt=payload.prompt,
        model=payload.model,
        session_id=payload.session_id,
    )


@app.post("/incident", response_model=IncidentReport)
def incident_endpoint(payload: IncidentRequest) -> IncidentReport:
    return run_incident_pipeline(user_input=payload.description, model=payload.model)


@app.get("/metrics")
def metrics_endpoint():
    return get_metrics_snapshot()