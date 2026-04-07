# Strands + LangSmith

A Strands-native agent example with LangSmith integration: OTEL tracing, PromptHub, datasets, evaluators, and experiments.

## Quickstart

```bash
cp .env.example .env   # fill in your keys
uv sync
uv run jupyter notebook demo.ipynb
```

## What the notebook does

The notebook (`demo.ipynb`) walks through the full lifecycle step by step:

1. **Setup** — Load env vars and wire up OTEL tracing so Strands spans export to LangSmith.
2. **Push prompt to Hub** — Push the agent's system prompt to LangSmith PromptHub (versioned).
3. **Create agent** — Pull the system prompt back from Hub and create a Strands agent with three tools: `lookup_knowledge_base`, `calculator`, and `web_search`.
4. **Create dataset** — Upload a QA evaluation dataset to LangSmith (6 examples covering knowledge lookups, math, and multi-tool questions).
5. **Create evaluator** — Attach an LLM-judge evaluator (correctness) to the dataset with an inline prompt.
6. **Run experiment** — Run the agent against every dataset example via `client.evaluate()`, scored by a PII-detection code evaluator.
7. **Demo** — Run a single multi-tool agent call and inspect the trace in LangSmith.
8. **Cleanup** — Optionally delete all created resources.

The agent is defined in `agent.py` and reused across experiments and the demo. The system prompt is pulled from PromptHub at runtime, so you can update it in Hub without changing code.

## Project structure

```
agent.py                 # Agent definition: tools + create_agent() + ask()
demo.ipynb               # Walkthrough notebook (run cells top-to-bottom)
langsmith_exporter.py    # OTEL exporter: transforms Strands spans for LangSmith
setup/
  config.py              # LangSmith client, constants, create_model()
  prompts.py             # Push/pull agent prompt to/from Hub
  datasets.py            # Create evaluation dataset
  evaluators.py          # Create LLM-judge evaluator on dataset
  cleanup.py             # Delete all created LangSmith resources
```

## Switching models

All model configuration lives in `setup/config.py` in the `create_model()` function. Comment/uncomment the provider you want:

### OpenAI (default)

```python
def create_model():
    from strands.models.openai import OpenAIModel
    return OpenAIModel(
        model_id="gpt-4o",
        client_args={"api_key": os.getenv("OPENAI_API_KEY")},
    )
```

Requires `OPENAI_API_KEY` in your `.env`.

### Amazon Bedrock

```python
def create_model():
    from strands.models import BedrockModel
    return BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
```

Uses the default boto3 credential chain (env vars, `~/.aws/credentials`, IAM role, etc).

### Anthropic (direct API)

```python
def create_model():
    from strands.models.anthropic import AnthropicModel
    return AnthropicModel(
        model_id="claude-sonnet-4-20250514",
        client_args={"api_key": os.getenv("ANTHROPIC_API_KEY")},
    )
```

Requires `pip install 'strands-agents[anthropic]'` and `ANTHROPIC_API_KEY` in your `.env`.

### Ollama (local)

```python
def create_model():
    from strands.models.ollama import OllamaModel
    return OllamaModel(model_id="llama3.1:8b")
```

Requires a running Ollama server (`ollama serve`).

## Environment variables

See `.env.example` for the full list. At minimum you need:

| Variable | Required for |
|---|---|
| `LANGSMITH_API_KEY` | All LangSmith features (tracing, Hub, datasets, evaluators) |
| `OPENAI_API_KEY` | OpenAI model provider + LLM-judge evaluator |
| AWS credentials | Bedrock model provider |

## Cleanup

To delete all LangSmith resources created by the demo:

```bash
uv run python -m setup.cleanup
```

Or uncomment the cleanup cell at the bottom of the notebook.
