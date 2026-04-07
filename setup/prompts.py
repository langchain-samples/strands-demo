"""Push and pull prompts to/from LangSmith Hub."""

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langsmith.utils import LangSmithConflictError

from setup.config import client


def _push_prompt(name: str, obj) -> Optional[str]:
    """Push a prompt object via the LangSmith SDK. Returns the URL or None if unchanged."""
    try:
        url = client.push_prompt(name, object=obj)
        print(f"  Pushed prompt: {url}")
        return url
    except LangSmithConflictError:
        print(f"  Prompt '{name}' unchanged.")
        return None


# ---------------------------------------------------------------------------
# Agent prompt
# ---------------------------------------------------------------------------

AGENT_PROMPT_NAME = "strands-research-assistant"

AGENT_SYSTEM_PROMPT = """\
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


def push_agent_prompt() -> Optional[str]:
    """Push the agent's system prompt to LangSmith Hub."""
    prompt = ChatPromptTemplate([("system", AGENT_SYSTEM_PROMPT)])
    return _push_prompt(AGENT_PROMPT_NAME, prompt)


def pull_agent_prompt() -> str:
    """Pull the agent's system prompt from LangSmith Hub.

    Returns the system message text as a plain string for use with Strands.
    """
    prompt = client.pull_prompt(f"{AGENT_PROMPT_NAME}:latest")
    # Render with no variables and grab the system message
    messages = prompt.invoke({}).to_messages()
    for msg in messages:
        if isinstance(msg, SystemMessage):
            return msg.content
    raise ValueError(f"No system message found in Hub prompt '{AGENT_PROMPT_NAME}'")
