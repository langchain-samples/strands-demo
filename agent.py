"""Strands research assistant agent with tools."""

import json
import math

from strands import Agent, tool

from setup.config import create_model


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = {
    "langsmith": {
        "description": "LangSmith is a platform for building production-grade LLM applications. It provides tracing, evaluation, dataset management, and prompt versioning.",
        "features": ["Tracing & observability", "Evaluation & testing", "Prompt management", "Dataset curation", "Online evaluators"],
        "url": "https://smith.langchain.com",
    },
    "strands": {
        "description": "Strands Agents is an open-source SDK for building AI agents. It supports multiple model providers including Amazon Bedrock, OpenAI, and Anthropic.",
        "features": ["Multi-provider support", "Tool use", "Streaming", "OpenTelemetry tracing"],
        "url": "https://github.com/strands-agents/sdk-python",
    },
    "bedrock": {
        "description": "Amazon Bedrock is a fully managed service that offers foundation models from leading AI companies through a unified API.",
        "features": ["Multiple model providers", "Serverless", "Fine-tuning", "Guardrails", "Knowledge bases"],
        "url": "https://aws.amazon.com/bedrock/",
    },
    "opentelemetry": {
        "description": "OpenTelemetry is an open-source observability framework for generating, collecting, and exporting telemetry data (traces, metrics, logs).",
        "features": ["Distributed tracing", "Metrics collection", "Log correlation", "Vendor-neutral"],
        "url": "https://opentelemetry.io",
    },
}


@tool
def lookup_knowledge_base(topic: str) -> str:
    """Look up information about a topic in the internal knowledge base.

    Args:
        topic: The topic to search for (e.g. 'langsmith', 'strands', 'bedrock').
    """
    key = topic.lower().strip()
    # Try exact match first, then substring match
    entry = KNOWLEDGE_BASE.get(key)
    if not entry:
        for k, v in KNOWLEDGE_BASE.items():
            if key in k or k in key:
                entry = v
                break
    if not entry:
        return f"No information found for '{topic}'. Available topics: {', '.join(KNOWLEDGE_BASE.keys())}"
    return json.dumps(entry, indent=2)


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Supports basic arithmetic, exponents, sqrt, log, etc.

    Args:
        expression: A mathematical expression to evaluate (e.g. '2**10', 'sqrt(144)', 'log(1000, 10)').
    """
    allowed = {
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
        "log2": math.log2, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "pi": math.pi, "e": math.e,
        "abs": abs, "round": round, "pow": pow,
        "min": min, "max": max,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool
def web_search(query: str) -> str:
    """Search the web for current information about a topic.

    Args:
        query: The search query string.
    """
    # Stub — in production, wire this to a real search API
    return (
        f"[Web search results for '{query}']\n"
        f"Note: This is a simulated search result. In production, "
        f"connect this tool to a real search API (e.g. Tavily, Brave, SerpAPI)."
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a research assistant that helps answer technical questions about \
AI/ML tools and platforms. You have access to:

- **lookup_knowledge_base**: Query the internal knowledge base for information \
about specific tools and platforms (LangSmith, Strands, Bedrock, OpenTelemetry).
- **calculator**: Evaluate mathematical expressions.
- **web_search**: Search the web for current information.

When answering questions:
1. Always check the knowledge base first for relevant topics.
2. Use the calculator for any numerical computations.
3. Fall back to web search for topics not in the knowledge base.
4. Synthesize information from multiple tools when needed.
5. Be concise but thorough — cite which tool provided each piece of information.\
"""

TOOLS = [lookup_knowledge_base, calculator, web_search]


def create_agent() -> Agent:
    """Create a Strands research assistant agent with tools."""
    return Agent(model=create_model(), system_prompt=SYSTEM_PROMPT, tools=TOOLS)


def ask(agent: Agent, question: str) -> str:
    """Ask the agent a question. Strands auto-emits OTEL spans for the agent call,
    each LLM invocation, and each tool execution — no manual tracing needed."""
    response = agent(question)
    return getattr(response, "output", str(response))
