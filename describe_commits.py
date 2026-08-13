"""
Describe changes between two commits in an Azure DevOps repository using LLM.
Fetches the diff via Azure DevOps REST API and generates an AI description,
saving the result as a local markdown file.

Prerequisites:
  1. AWS SSO login:  aws sso login --profile ssop-all-all-allx-accadm-001-029099142207
  2. Environment variables (set before running):
       SET AWS_PROFILE=ssop-all-all-allx-accadm-001-029099142207
       SET ADO_PAT=your-azure-devops-personal-access-token
  3. The LLM calls go through AWS Bedrock (Claude). Make sure your SSO session is active.

Usage:
  python describe_commits.py <base_commit> <target_commit>
  python describe_commits.py abc1234 def5678

Output:
  Markdown file saved to ./commit_descriptions/<repo_name>/<base>..<target>.md
"""

import asyncio
import base64
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote

os.environ.setdefault("AWS_PROFILE", "ssop-all-all-allx-accadm-001-029099142207")
os.environ.setdefault("AWS_USE_IMDS", "true")
os.environ.setdefault("AWS_REGION_NAME", "us-east-1")

import dotenv
dotenv.load_dotenv()
import requests

# Azure DevOps configuration
ADO_ORG = "https://dev.azure.com/jblprd"
ADO_PROJECT = "JGP Common Data Platform"
ADO_REPO = "GenAI-PL-Invoice"

# PR-Agent needs PYTHONPATH=. to resolve imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler
from pr_agent.config_loader import get_settings
from pr_agent.log import get_logger, setup_logger

setup_logger("INFO")
logger = get_logger()

MODEL = "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"

SYSTEM_PROMPT = """\
You are a code change analyst. You will be given a diff between two commits in a repository.
Analyze the changes and produce a structured markdown description including:

1. **Overview**: A concise summary (2-3 sentences) of what changed overall.
2. **Change Type**: Categorize (e.g., Bug fix, Feature, Refactor, Config change, Documentation, etc.)
3. **Key Changes**: A bullet list of the most important changes, grouped by file or component.
4. **Impact Assessment**: What areas of the codebase are affected and potential risks.

Keep the description clear and useful for someone reviewing the commit history.
Do NOT include the raw diff in your output.
"""

USER_PROMPT_TEMPLATE = """\
Here is the diff between commit `{base_commit}` and commit `{target_commit}` in repository `{repo}`:

```diff
{diff_content}
```

Please provide a structured description of these changes.
"""


def get_ado_pat() -> str:
    """Get an Azure DevOps PAT from the ADO_PAT environment variable."""
    pat = os.environ.get("ADO_PAT")
    if not pat:
        print("ERROR: ADO_PAT environment variable is not set.")
        print("  Generate a PAT at: https://dev.azure.com/jblprd/_usersSettings/tokens")
        print("  Then set it:  SET ADO_PAT=your-token-here")
        sys.exit(1)
    return pat


def get_ado_headers(access_token: str) -> dict:
    """Build authorization headers for Azure DevOps REST API."""
    b64_pat = base64.b64encode(f":{access_token}".encode()).decode()
    return {"Authorization": f"Basic {b64_pat}"}


def get_commit_diff(access_token: str, base_commit: str, target_commit: str) -> str:
    """Fetch the diff between two commits using Azure DevOps REST API."""
    project_encoded = quote(ADO_PROJECT)
    repo_encoded = quote(ADO_REPO)
    headers = get_ado_headers(access_token)

    # Get the list of changes between commits
    url = (
        f"{ADO_ORG}/{project_encoded}/_apis/git/repositories/{repo_encoded}"
        f"/diffs/commits?baseVersion={base_commit}&targetVersion={target_commit}"
        f"&baseVersionType=commit&targetVersionType=commit&api-version=7.1"
    )
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"ERROR: Failed to get commit diff. Status: {response.status_code}")
        print(response.text)
        sys.exit(1)

    diff_data = response.json()
    changes = diff_data.get("changes", [])

    if not changes:
        print("No changes found between the two commits.")
        sys.exit(0)

    # Build a unified diff by fetching each changed file's content
    diff_parts = []
    for change in changes:
        item = change.get("item", {})
        change_type = change.get("changeType", "unknown")
        file_path = item.get("path", "unknown")

        # Skip folders
        if item.get("isFolder", False):
            continue

        diff_parts.append(f"--- {change_type}: {file_path}")

        # For add/edit, try to get the actual diff via the items API
        if change_type in ("add", "edit", "rename"):
            item_url = (
                f"{ADO_ORG}/{project_encoded}/_apis/git/repositories/{repo_encoded}"
                f"/commits/{target_commit}/changes?api-version=7.1"
            )
            # We'll use a simpler approach: get the diff for the specific commit range
            # using the /diffs endpoint with item-level detail
            diff_parts.append(f"  (change type: {change_type})")
        elif change_type == "delete":
            diff_parts.append(f"  (file deleted)")

    # Use the commits comparison endpoint for actual text diffs
    # Azure DevOps doesn't have a single "unified diff" endpoint like GitHub,
    # so we fetch changes per item
    full_diff = fetch_text_diffs(access_token, base_commit, target_commit, changes)
    return full_diff


def fetch_text_diffs(access_token: str, base_commit: str, target_commit: str, changes: list) -> str:
    """Fetch text diffs for changed files by comparing file contents between commits."""
    project_encoded = quote(ADO_PROJECT)
    repo_encoded = quote(ADO_REPO)
    headers = get_ado_headers(access_token)

    diff_output = []
    for change in changes:
        item = change.get("item", {})
        change_type = change.get("changeType", "unknown")
        file_path = item.get("path", "")

        if item.get("isFolder", False):
            continue

        diff_output.append(f"=== {change_type.upper()}: {file_path} ===")

        if change_type == "delete":
            # Fetch the old version
            old_url = (
                f"{ADO_ORG}/{project_encoded}/_apis/git/repositories/{repo_encoded}"
                f"/items?path={quote(file_path)}&versionDescriptor.version={base_commit}"
                f"&versionDescriptor.versionType=commit&api-version=7.1"
            )
            resp = requests.get(old_url, headers=headers)
            if resp.status_code == 200:
                for line in resp.text.splitlines()[:50]:  # Limit for context
                    diff_output.append(f"- {line}")
                if len(resp.text.splitlines()) > 50:
                    diff_output.append(f"  ... ({len(resp.text.splitlines()) - 50} more lines)")
            diff_output.append("")
            continue

        if change_type == "add":
            # Fetch the new version
            new_url = (
                f"{ADO_ORG}/{project_encoded}/_apis/git/repositories/{repo_encoded}"
                f"/items?path={quote(file_path)}&versionDescriptor.version={target_commit}"
                f"&versionDescriptor.versionType=commit&api-version=7.1"
            )
            resp = requests.get(new_url, headers=headers)
            if resp.status_code == 200:
                for line in resp.text.splitlines()[:100]:
                    diff_output.append(f"+ {line}")
                if len(resp.text.splitlines()) > 100:
                    diff_output.append(f"  ... ({len(resp.text.splitlines()) - 100} more lines)")
            diff_output.append("")
            continue

        if change_type in ("edit", "rename"):
            # Fetch both versions and show a simple diff
            old_url = (
                f"{ADO_ORG}/{project_encoded}/_apis/git/repositories/{repo_encoded}"
                f"/items?path={quote(file_path)}&versionDescriptor.version={base_commit}"
                f"&versionDescriptor.versionType=commit&api-version=7.1"
            )
            new_url = (
                f"{ADO_ORG}/{project_encoded}/_apis/git/repositories/{repo_encoded}"
                f"/items?path={quote(file_path)}&versionDescriptor.version={target_commit}"
                f"&versionDescriptor.versionType=commit&api-version=7.1"
            )
            old_resp = requests.get(old_url, headers=headers)
            new_resp = requests.get(new_url, headers=headers)

            if old_resp.status_code == 200 and new_resp.status_code == 200:
                import difflib
                old_lines = old_resp.text.splitlines(keepends=True)
                new_lines = new_resp.text.splitlines(keepends=True)
                unified = list(difflib.unified_diff(
                    old_lines, new_lines,
                    fromfile=f"a{file_path}",
                    tofile=f"b{file_path}",
                    n=3
                ))
                # Limit diff size per file
                if len(unified) > 200:
                    diff_output.extend(line.rstrip() for line in unified[:200])
                    diff_output.append(f"  ... (diff truncated, {len(unified) - 200} more lines)")
                else:
                    diff_output.extend(line.rstrip() for line in unified)
            elif old_resp.status_code != 200:
                diff_output.append(f"  (could not fetch base version: {old_resp.status_code})")
            elif new_resp.status_code != 200:
                diff_output.append(f"  (could not fetch target version: {new_resp.status_code})")
            diff_output.append("")
            continue

        diff_output.append(f"  (unhandled change type: {change_type})")
        diff_output.append("")

    return "\n".join(diff_output)


def get_commit_info(access_token: str, commit_id: str) -> dict:
    """Fetch commit metadata from Azure DevOps."""
    project_encoded = quote(ADO_PROJECT)
    repo_encoded = quote(ADO_REPO)
    headers = get_ado_headers(access_token)

    url = (
        f"{ADO_ORG}/{project_encoded}/_apis/git/repositories/{repo_encoded}"
        f"/commits/{commit_id}?api-version=7.1"
    )
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        return resp.json()
    return {}


async def describe_diff(diff_content: str, base_commit: str, target_commit: str) -> str:
    """Call the LLM to generate a description of the diff."""
    # Configure settings for the AI handler
    get_settings().set("config.model", MODEL)
    get_settings().set("config.fallback_models", [MODEL])
    get_settings().set("config.ai_timeout", 120)

    ai_handler = LiteLLMAIHandler()
    user_prompt = USER_PROMPT_TEMPLATE.format(
        base_commit=base_commit[:8],
        target_commit=target_commit[:8],
        repo=ADO_REPO,
        diff_content=diff_content[:50000]  # Truncate if massive
    )

    response, finish_reason = await ai_handler.chat_completion(
        model=MODEL,
        system=SYSTEM_PROMPT,
        user=user_prompt,
        temperature=0.2,
    )
    return response


async def main():
    if len(sys.argv) < 3:
        print("Usage: python describe_commits.py <base_commit> <target_commit>")
        print("")
        print("  base_commit   - The older commit SHA (or short SHA)")
        print("  target_commit - The newer commit SHA (or short SHA)")
        print("")
        print("Example:")
        print("  python describe_commits.py abc1234 def5678")
        sys.exit(1)

    base_commit = sys.argv[1]
    target_commit = sys.argv[2]

    # Output directory
    output_dir = Path("commit_descriptions") / ADO_REPO
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already described
    output_file = output_dir / f"{base_commit[:8]}..{target_commit[:8]}.md"
    if output_file.exists():
        print(f"Already described: {output_file}")
        print("Delete the file to regenerate.")
        return

    print("Reading ADO_PAT from environment...")
    access_token = get_ado_pat()
    print("  ✓ PAT loaded.\n")

    print(f"Fetching diff between {base_commit[:8]} and {target_commit[:8]}...")
    diff_content = get_commit_diff(access_token, base_commit, target_commit)

    if not diff_content.strip():
        print("No meaningful diff found between the commits.")
        return

    print(f"  ✓ Diff fetched ({len(diff_content)} chars)\n")

    # Get commit metadata
    base_info = get_commit_info(access_token, base_commit)
    target_info = get_commit_info(access_token, target_commit)

    print("Generating AI description...")
    description = await describe_diff(diff_content, base_commit, target_commit)
    print("  ✓ Description generated\n")

    # Build the markdown output
    md_content = f"# Commit Comparison: `{base_commit[:8]}` → `{target_commit[:8]}`\n\n"
    md_content += f"**Repository:** {ADO_REPO}\n\n"

    if base_info:
        author = base_info.get("author", {}).get("name", "Unknown")
        date = base_info.get("author", {}).get("date", "Unknown")
        comment = base_info.get("comment", "")
        md_content += f"**Base Commit:** `{base_commit}` by {author} on {date}\n"
        md_content += f"> {comment}\n\n"

    if target_info:
        author = target_info.get("author", {}).get("name", "Unknown")
        date = target_info.get("author", {}).get("date", "Unknown")
        comment = target_info.get("comment", "")
        md_content += f"**Target Commit:** `{target_commit}` by {author} on {date}\n"
        md_content += f"> {comment}\n\n"

    md_content += "---\n\n"
    md_content += description

    output_file.write_text(md_content, encoding="utf-8")
    print(f"  ✓ Saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
