"""Create the evaluation dataset in LangSmith."""

from setup.config import client, DATASET_NAME

EVAL_EXAMPLES = [
    {
        "input": {"question": "What is LangSmith and what are its main features?"},
        "output": {"answer": "LangSmith is a platform for building production-grade LLM applications. Key features include tracing & observability, evaluation & testing, prompt management, dataset curation, and online evaluators."},
    },
    {
        "input": {"question": "What model providers does Strands Agents support?"},
        "output": {"answer": "Strands Agents supports multiple model providers including Amazon Bedrock, OpenAI, and Anthropic."},
    },
    {
        "input": {"question": "What is 2^16 and what is the square root of that result?"},
        "output": {"answer": "2^16 = 65536, and the square root of 65536 is 256."},
    },
    {
        "input": {"question": "Compare Amazon Bedrock and OpenTelemetry — what does each do?"},
        "output": {"answer": "Amazon Bedrock is a managed service offering foundation models from multiple AI providers through a unified API. OpenTelemetry is an observability framework for generating and exporting telemetry data like traces, metrics, and logs. Bedrock is for AI model access, while OpenTelemetry is for application monitoring."},
    },
    {
        "input": {"question": "If I have 3 Bedrock models each handling 1000 requests per hour, how many total requests per day is that?"},
        "output": {"answer": "3 models * 1000 requests/hour * 24 hours = 72,000 requests per day."},
    },
    {
        "input": {"question": "What is the URL for the Strands Agents SDK repository?"},
        "output": {"answer": "https://github.com/strands-agents/sdk-python"},
    },
]


def create_dataset() -> None:
    """Create the evaluation dataset (idempotent)."""
    if client.has_dataset(dataset_name=DATASET_NAME):
        print(f"  Dataset '{DATASET_NAME}' already exists. Skipping.")
        return
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="QA pairs for evaluating the Strands research assistant (knowledge base + calculator + web search)",
    )
    client.create_examples(
        inputs=[ex["input"] for ex in EVAL_EXAMPLES],
        outputs=[ex["output"] for ex in EVAL_EXAMPLES],
        dataset_id=dataset.id,
    )
    print(f"  Created dataset '{DATASET_NAME}' with {len(EVAL_EXAMPLES)} examples.")
