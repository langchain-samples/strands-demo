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
