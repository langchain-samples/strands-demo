"""Shared configuration: LangSmith client, auth headers, model factory."""

import os
from typing import Dict

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

client = Client()
LANGSMITH_API_URL = os.getenv("LANGSMITH_API_URL", "https://api.smith.langchain.com")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

PROJECT_NAME = "strands-repo-demo"
DATASET_NAME = "Strands Research QA"


def auth_headers() -> Dict[str, str]:
    if not LANGSMITH_API_KEY:
        raise RuntimeError("LANGSMITH_API_KEY is required.")
    return {"x-api-key": LANGSMITH_API_KEY, "Content-Type": "application/json"}


def get_owner() -> str:
    import requests
    resp = requests.get(
        f"{LANGSMITH_API_URL}/api/v1/settings", headers=auth_headers(), timeout=30
    )
    if resp.status_code >= 300:
        return "-"
    return resp.json().get("tenant_handle") or "-"


# ---------------------------------------------------------------------------
# Model — swap between OpenAI and Bedrock here
# ---------------------------------------------------------------------------

def create_model():
    """Create the Strands model instance.

    Swap providers by commenting/uncommenting the blocks below.
    """

    # --- OpenAI (default for testing) ---
    from strands.models.openai import OpenAIModel
    return OpenAIModel(
        model_id="gpt-4o",
        client_args={"api_key": os.getenv("OPENAI_API_KEY")},
    )

    # --- Bedrock (uncomment for production / sharing) ---
    # from strands.models import BedrockModel
    # return BedrockModel(
    #     model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    # )
