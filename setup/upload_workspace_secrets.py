"""Upload AWS credentials to LangSmith workspace secrets for Bedrock playground access.

Two authentication methods (for self-hosted LangSmith):
  1. Direct credentials: AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY (+ optional AWS_SESSION_TOKEN)
  2. Bearer token: AWS_BEARER_TOKEN_BEDROCK

Can be used standalone (CLI) or called from a notebook via upload_bedrock_secrets().
"""

import os

import httpx
from dotenv import load_dotenv

load_dotenv()

# ── Secret groups by method ──────────────────────────────────────────────────

METHODS = {
    "1": {
        "name": "Direct credentials (AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY)",
        "required": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
        "optional": ["AWS_SESSION_TOKEN"],
    },
    "2": {
        "name": "Bearer token (AWS_BEARER_TOKEN_BEDROCK)",
        "required": ["AWS_BEARER_TOKEN_BEDROCK"],
        "optional": [],
    },
}


def _collect_secrets(method: str) -> list[dict]:
    """Read secret values from env vars for the chosen method."""
    spec = METHODS[method]
    secrets = []
    missing = []

    for key in spec["required"]:
        val = os.getenv(key)
        if not val:
            missing.append(key)
        else:
            secrets.append({"key": key, "value": val})

    if missing:
        raise ValueError(
            f"Missing required env vars for this method: {', '.join(missing)}\n"
            "Set them in your .env file and re-run."
        )

    for key in spec["optional"]:
        val = os.getenv(key)
        if val:
            secrets.append({"key": key, "value": val})

    return secrets


def _upload(secrets: list[dict], api_url: str, api_key: str) -> None:
    url = f"{api_url}/api/v1/workspaces/current/secrets"
    resp = httpx.post(
        url,
        json=secrets,
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()


def upload_bedrock_secrets(
    api_url: str | None = None,
    api_key: str | None = None,
) -> None:
    """Interactive prompt to choose a method and upload secrets.

    Args:
        api_url: LangSmith API URL. Falls back to LANGSMITH_ENDPOINT / LANGSMITH_API_URL env vars.
        api_key: LangSmith API key. Falls back to LANGSMITH_API_KEY env var.
    """
    api_url = api_url or os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGSMITH_API_URL", "https://api.smith.langchain.com")
    api_key = api_key or os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        raise RuntimeError("LANGSMITH_API_KEY is required (set env var or pass api_key=).")

    print("Which AWS credentials from your .env should be uploaded?\n")
    for key, spec in METHODS.items():
        print(f"  {key}) {spec['name']}")
    print()

    choice = input("Choice [1/2]: ").strip()
    if choice not in METHODS:
        raise ValueError(f"Invalid choice: {choice}")

    secrets = _collect_secrets(choice)

    keys = [s["key"] for s in secrets]
    print(f"\nUploading to {api_url}: {', '.join(keys)}")
    _upload(secrets, api_url, api_key)
    print("Done — workspace secrets updated.")


# ── CLI entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    upload_bedrock_secrets()
