"""Push agent and evaluation prompts to LangSmith Hub."""

import requests
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.structured import StructuredPrompt
from langsmith.utils import LangSmithConflictError

from setup.config import LANGSMITH_API_URL, client, auth_headers, get_owner


# ---------------------------------------------------------------------------
# REST API push (for simple ChatPromptTemplates)
# ---------------------------------------------------------------------------

def _push_prompt_rest(name: str, manifest: dict) -> Optional[str]:
    """Push a prompt manifest to Hub via REST. Creates the repo if needed."""
    owner = get_owner()
    url = f"{LANGSMITH_API_URL}/api/v1/commits/{owner}/{name}"
    body = {"manifest": manifest}

    resp = requests.post(url, headers=auth_headers(), json=body, timeout=30)
    if resp.status_code == 404:
        requests.post(
            f"{LANGSMITH_API_URL}/api/v1/repos",
            headers=auth_headers(),
            json={"repo_handle": name, "owner_handle": owner, "is_public": False},
            timeout=30,
        )
        resp = requests.post(url, headers=auth_headers(), json=body, timeout=30)
    if resp.status_code == 409:
        print(f"  Prompt '{name}' unchanged.")
        return None
    if resp.status_code >= 300:
        print(f"  Warning: Failed to push prompt '{name}': {resp.status_code} {resp.text}")
        return None
    prompt_url = f"{LANGSMITH_API_URL}/hub/{owner}/{name}"
    print(f"  Pushed prompt: {prompt_url}")
    return prompt_url


def _make_prompt_template(template: str, input_variables: list) -> dict:
    return {
        "lc": 1, "type": "constructor",
        "id": ["langchain", "prompts", "prompt", "PromptTemplate"],
        "kwargs": {"input_variables": input_variables, "template": template, "template_format": "f-string"},
    }


def _make_message(role_id: str, template: str, input_variables: list) -> dict:
    return {
        "lc": 1, "type": "constructor",
        "id": ["langchain", "prompts", "chat", role_id],
        "kwargs": {"prompt": _make_prompt_template(template, input_variables)},
    }


# ---------------------------------------------------------------------------
# SDK push (for StructuredPrompts — the REST API doesn't support them)
# ---------------------------------------------------------------------------

def _push_prompt_sdk(name: str, obj) -> Optional[str]:
    """Push a prompt object via the LangSmith SDK (handles StructuredPrompt serialization)."""
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
    prompt = ChatPromptTemplate([
        ("system", AGENT_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    return _push_prompt_sdk("strands-research-assistant", prompt)


# ---------------------------------------------------------------------------
# Evaluation prompt (StructuredPrompt with output schema)
# ---------------------------------------------------------------------------

EVAL_SYSTEM_PROMPT = (
    "You are an expert evaluator. Given a question, the AI's answer, and a "
    "reference answer, judge whether the AI answer is correct.\n\n"
    "Grade as correct (true) if the AI answer captures the key facts from "
    "the reference, even if worded differently. Grade as incorrect (false) "
    "if the AI answer is missing critical information or is factually wrong."
)
EVAL_HUMAN_PROMPT = (
    "Question: {input}\n\n"
    "AI Answer: {output}\n\n"
    "Reference Answer: {reference}\n\n"
    "Is the AI answer correct?"
)

EVAL_SCHEMA = {
    "title": "extract",
    "description": "Extract information from the user's response.",
    "type": "object",
    "properties": {
        "correctness": {
            "type": "boolean",
            "description": "Is the AI answer correct based on the reference?",
        },
        "comment": {
            "type": "string",
            "description": "Reasoning for the correctness score",
        },
    },
    "required": ["correctness", "comment"],
}


def push_eval_prompt() -> Optional[str]:
    """Push the correctness evaluation prompt to LangSmith Hub.

    Uses a StructuredPrompt so the output schema (correctness + comment)
    is bundled with the prompt itself, matching the starter-kit pattern.
    Pushed via the SDK since the REST API doesn't support StructuredPrompt.
    """
    prompt = StructuredPrompt(
        messages=[
            ("system", EVAL_SYSTEM_PROMPT),
            ("human", EVAL_HUMAN_PROMPT),
        ],
        schema_=EVAL_SCHEMA,
    )
    return _push_prompt_sdk("strands-answer-correctness-eval", prompt)
