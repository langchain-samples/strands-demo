"""Delete all LangSmith resources created by this demo."""

import requests

from setup.config import client, LANGSMITH_API_URL, DATASET_NAME, PROJECT_NAME, auth_headers


PROMPT_NAMES = ["strands-research-assistant"]


def delete_dataset() -> None:
    """Delete the evaluation dataset."""
    ds = next(client.list_datasets(dataset_name=DATASET_NAME), None)
    if ds:
        client.delete_dataset(dataset_id=ds.id)
        print(f"  Deleted dataset: {DATASET_NAME}")
    else:
        print(f"  Dataset '{DATASET_NAME}' not found, skipping.")


def delete_evaluators() -> None:
    """Delete evaluators attached to the dataset."""
    ds = next(client.list_datasets(dataset_name=DATASET_NAME), None)
    if not ds:
        print("  No dataset found, skipping evaluator cleanup.")
        return
    dataset_id = str(ds.id)
    resp = requests.get(
        f"{LANGSMITH_API_URL}/api/v1/runs/rules",
        headers=auth_headers(),
        params={"dataset_id": dataset_id},
        timeout=30,
    )
    if resp.status_code >= 300:
        print(f"  Warning: Failed to list evaluators: {resp.status_code}")
        return
    for rule in resp.json():
        if rule.get("dataset_id") == dataset_id:
            rule_id = rule.get("id")
            name = rule.get("display_name", rule_id)
            del_resp = requests.delete(
                f"{LANGSMITH_API_URL}/api/v1/runs/rules/{rule_id}",
                headers=auth_headers(),
                timeout=30,
            )
            if del_resp.status_code < 300:
                print(f"  Deleted evaluator: {name}")
            else:
                print(f"  Warning: Failed to delete evaluator '{name}': {del_resp.status_code}")


def delete_prompts() -> None:
    """Delete prompts from LangSmith Hub."""
    for name in PROMPT_NAMES:
        try:
            client.delete_prompt(name)
            print(f"  Deleted prompt: {name}")
        except Exception:
            print(f"  Prompt '{name}' not found, skipping.")


def delete_project() -> None:
    """Delete the LangSmith project (and all its traces)."""
    project = next(client.list_projects(name=PROJECT_NAME), None)
    if project:
        client.delete_project(project_name=PROJECT_NAME)
        print(f"  Deleted project: {PROJECT_NAME}")
    else:
        print(f"  Project '{PROJECT_NAME}' not found, skipping.")


def cleanup_all() -> None:
    """Delete all resources created by this demo."""
    print("Cleaning up LangSmith resources...")
    print("\n[1/4] Evaluators...")
    delete_evaluators()
    print("\n[2/4] Dataset...")
    delete_dataset()
    print("\n[3/4] Prompts...")
    delete_prompts()
    print("\n[4/4] Project...")
    delete_project()
    print("\nDone.")


if __name__ == "__main__":
    cleanup_all()
