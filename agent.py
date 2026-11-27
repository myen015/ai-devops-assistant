import os
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import json
import litellm
from pydantic import BaseModel, Field
from analyzer import analyze_logs
from rag_qa import ask_docs
from rag_store import list_sources

# Әр session_id бойынша сөйлесу тарихын сақтайтын жад (Memory)
SESSION_MEMORY: dict[str, list[dict]] = {}


class ToolCallRecord(BaseModel):
    tool: str
    args: dict


class AgentResponse(BaseModel):
    answer: str
    tools_used: list[ToolCallRecord] = Field(default_factory=list)
    steps_taken: int
    session_id: str | None = None


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documentation",
            "description": "Search internal technical documentation (API specs, error codes, infra guides) to answer questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The technical question to look up in docs"}
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_logs",
            "description": "Analyze raw application or server logs to extract errors, warnings, and root causes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "log_text": {"type": "string", "description": "Raw log lines to analyze"}
                },
                "required": ["log_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_indexed_documents",
            "description": "List all files currently stored in the knowledge base. Call this ONLY ONCE when asked about available files.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _execute_tool(name: str, args: dict) -> str:
    if name == "search_documentation":
        res = ask_docs(args["question"])
        return res.model_dump_json()
    if name == "inspect_logs":
        res = analyze_logs(args["log_text"])
        return res.model_dump_json()
    if name == "get_indexed_documents":
        return json.dumps(list_sources())
    return json.dumps({"error": f"Unknown tool: {name}"})


def run_agent(
    user_prompt: str,
    model: str = "ollama_chat/qwen2.5:3b",
    max_steps: int = 4,
    session_id: str | None = None,
) -> AgentResponse:
    system_msg = {
        "role": "system",
        "content": (
            "You are an autonomous DevOps & AI Engineering Assistant. "
            "Use the provided tools when needed. Do not call the same tool twice with the same arguments. "
            "Once you have the information, provide a clear final answer immediately."
        ),
    }

    # Егер session_id берілсе, алдыңғы сөйлесу тарихын жүктейміз (соңғы 10 хабарлама)
    history = SESSION_MEMORY.get(session_id, [])[-10:] if session_id else []
    messages = [system_msg, *history, {"role": "user", "content": user_prompt}]

    tools_used: list[ToolCallRecord] = []
    seen_calls: set[str] = set()

    for step in range(1, max_steps + 1):
        response = litellm.completion(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.0,
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not getattr(msg, "tool_calls", None):
            final_answer = msg.content or ""
            if session_id:
                SESSION_MEMORY.setdefault(session_id, []).extend(
                    [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": final_answer},
                    ]
                )
            return AgentResponse(
                answer=final_answer,
                tools_used=tools_used,
                steps_taken=step,
                session_id=session_id,
            )

        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments or "{}")
            call_signature = f"{fn_name}:{json.dumps(fn_args, sort_keys=True)}"

            if call_signature in seen_calls:
                tool_output = json.dumps({"note": "Already called this tool with identical arguments. Use previous result."})
            else:
                seen_calls.add(call_signature)
                tools_used.append(ToolCallRecord(tool=fn_name, args=fn_args))
                tool_output = _execute_tool(fn_name, fn_args)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": fn_name,
                    "content": tool_output,
                }
            )

    return AgentResponse(
        answer="Max reasoning steps reached without a final answer.",
        tools_used=tools_used,
        steps_taken=max_steps,
        session_id=session_id,
    )