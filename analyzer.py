from typing import Literal
import litellm
from pydantic import BaseModel, Field


class LogIssue(BaseModel):
    service: str = Field(..., description="Service name where issue occurred")
    severity: Literal["low", "medium", "high"]
    reason: str = Field(..., description="Short description of the issue")


class LogReport(BaseModel):
    summary: str = Field(
        ...,
        description="One plain-text sentence summarizing the overall log status. No markdown or bullet points.",
    )
    issues: list[LogIssue] = Field(default_factory=list)


def analyze_logs(log_text: str, model: str = "ollama/qwen2.5:3b") -> LogReport:
    response = litellm.completion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict log analyzer. Extract every ERROR and WARN line into the 'issues' list. "
                    "Ignore INFO lines. Keep 'summary' as one plain sentence without markdown.\n"
                    'Example input: "2026-01-01 01:00:00 ERROR auth: Invalid password"\n'
                    'Example output: {"summary": "One authentication error detected.", '
                    '"issues": [{"service": "auth", "severity": "high", "reason": "Invalid password"}]}'
                ),
            },
            {
                "role": "user",
                "content": f"Analyze these logs:\n{log_text}",
            },
        ],
        response_format=LogReport,
        temperature=0.0,
    )
    return LogReport.model_validate_json(response.choices[0].message.content)