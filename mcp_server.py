from mcp.server.mcpserver import MCPServer
from analyzer import analyze_logs
from multi_agent import run_incident_pipeline
from rag_qa import ask_docs

mcp = MCPServer("Applied-AI-Engineering-Server")


@mcp.tool()
def search_docs_tool(question: str) -> dict:
    """Search internal technical documentation and return an answer with exact file/page citations."""
    return ask_docs(question).model_dump()


@mcp.tool()
def analyze_logs_tool(log_text: str) -> dict:
    """Analyze raw application or server logs to extract errors, warnings, and root causes."""
    return analyze_logs(log_text).model_dump()


@mcp.tool()
def resolve_incident_tool(incident_description: str) -> dict:
    """Run the 3-agent incident response pipeline (Triage -> Researcher -> Responder) to produce a P1-P3 report."""
    return run_incident_pipeline(incident_description).model_dump()


if __name__ == "__main__":
    mcp.run()