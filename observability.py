import json
import logging
import os
import time
import uuid
from collections import defaultdict
from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("ai_service")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

# Жадтағы жеңіл метрикалар қоймасы
METRICS = {
    "total_requests": 0,
    "error_requests": 0,
    "by_path": defaultdict(lambda: {"count": 0, "total_ms": 0.0}),
}


def get_metrics_snapshot() -> dict:
    """Барлық эндпоинттар бойынша сұраныс саны мен орташа кідірісті (latency) қайтарады."""
    by_path = {}
    for path, stats in METRICS["by_path"].items():
        avg_ms = round(stats["total_ms"] / stats["count"], 2) if stats["count"] else 0.0
        by_path[path] = {"count": stats["count"], "avg_latency_ms": avg_ms}

    return {
        "total_requests": METRICS["total_requests"],
        "error_requests": METRICS["error_requests"],
        "by_path": by_path,
    }


async def observability_and_auth_middleware(request: Request, call_next):
    """Әр HTTP сұранысты тіркейтін, уақытын өлшейтін және X-API-Key тексеретін middleware."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
    start = time.perf_counter()
    path = request.url.path

    # Security: егер API_KEY env айнымалысы берілсе, қорғалған жолдарды тексереміз
    expected_key = os.getenv("API_KEY")
    if expected_key and path not in PUBLIC_PATHS:
        provided_key = request.headers.get("X-API-Key")
        if provided_key != expected_key:
            response = JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing X-API-Key"},
            )
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            _record_and_log(request_id, request.method, path, 401, duration_ms)
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time-Ms"] = str(duration_ms)
            return response

    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)

    _record_and_log(request_id, request.method, path, response.status_code, duration_ms)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = str(duration_ms)
    return response


def _record_and_log(request_id: str, method: str, path: str, status_code: int, duration_ms: float) -> None:
    METRICS["total_requests"] += 1
    if status_code >= 400:
        METRICS["error_requests"] += 1
    METRICS["by_path"][path]["count"] += 1
    METRICS["by_path"][path]["total_ms"] += duration_ms

    log_entry = {
        "event": "http_request",
        "request_id": request_id,
        "method": method,
        "path": path,
        "status": status_code,
        "duration_ms": duration_ms,
    }
    logger.info(json.dumps(log_entry))