"""Create LLM-judge and code evaluators on the dataset."""

import requests

from setup.config import client, LANGSMITH_API_URL, DATASET_NAME, auth_headers


def create_llm_judge_evaluator() -> None:
    """Create an LLM-judge evaluator on the dataset referencing the Hub eval prompt."""
    dataset = next(client.list_datasets(dataset_name=DATASET_NAME), None)
    if not dataset:
        print("  Dataset not found, skipping evaluator creation.")
        return

    dataset_id = str(dataset.id)
    eval_name = "correctness"

    # Check if evaluator already exists
    resp = requests.get(
        f"{LANGSMITH_API_URL}/api/v1/runs/rules",
        headers=auth_headers(),
        params={"dataset_id": dataset_id, "name_contains": eval_name},
        timeout=30,
    )
    if resp.status_code == 200:
        for e in resp.json():
            if (
                e.get("display_name") == eval_name
                and e.get("dataset_id") == dataset_id
                and (e.get("evaluators") or e.get("code_evaluators"))
            ):
                print(f"  Evaluator '{eval_name}' already exists. Skipping.")
                return

    body = {
        "display_name": eval_name,
        "evaluator_version": 3,
        "dataset_id": dataset_id,
        "sampling_rate": 1.0,
        "is_enabled": True,
        "filter": "eq(is_root, true)",
        "evaluators": [
            {
                "structured": {
                    "prompt": [
                        ["system", (
                            "You are an expert evaluator. Given a question, the AI's answer, "
                            "and a reference answer, judge whether the AI answer is correct.\n\n"
                            "Grade as correct (true) if the AI answer captures the key facts "
                            "from the reference, even if worded differently. Grade as incorrect "
                            "(false) if the AI answer is missing critical information or is "
                            "factually wrong."
                        )],
                        ["human", (
                            "Question: {{input}}\n\n"
                            "AI Answer: {{output}}\n\n"
                            "Reference Answer: {{reference}}\n\n"
                            "Is the AI answer correct?"
                        )],
                    ],
                    "model": {
                        "lc": 1, "type": "constructor",
                        "id": ["langchain_aws", "chat_models", "ChatBedrockConverse"],
                        "kwargs": {
                            "temperature": 1,
                            "model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                            "region_name": "us-east-1",
                            "aws_access_key_id": {"id": ["AWS_ACCESS_KEY_ID"], "lc": 1, "type": "secret"},
                            "aws_secret_access_key": {"id": ["AWS_SECRET_ACCESS_KEY"], "lc": 1, "type": "secret"},
                            "aws_session_token": {"id": ["AWS_SESSION_TOKEN"], "lc": 1, "type": "secret"},
                        },
                    },
                    "schema": {
                        "title": "extract",
                        "description": "Extract information from the user's response.",
                        "type": "object",
                        "properties": {
                            "correctness": {"type": "boolean", "description": "Is the AI answer correct?"},
                            "comment": {"type": "string", "description": "Reasoning for the score"},
                        },
                        "required": ["correctness", "comment"],
                        "strict": True,
                    },
                    "template_format": "mustache",
                    "variable_mapping": {"input": "input", "output": "output", "reference": "referenceOutput"},
                }
            }
        ],
    }

    resp = requests.post(f"{LANGSMITH_API_URL}/runs/rules", headers=auth_headers(), json=body, timeout=30)
    if resp.status_code < 300:
        print(f"  Created LLM-judge evaluator '{eval_name}' on dataset.")
    else:
        print(f"  Warning: Failed to create evaluator: {resp.status_code} {resp.text}")
