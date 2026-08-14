"""
Describe PRs or compare commits for an Azure DevOps repository using AI.

Subcommands:
  prs      - Describe all completed PRs (uses PR-Agent /describe)
  commits  - Describe changes between two commits (direct LLM call)

Prerequisites:
  1. AWS SSO login:  aws sso login --profile ssop-all-all-allx-accadm-001-029099142207
  2. Environment variables (set before running):
       SET AWS_PROFILE=ssop-all-all-allx-accadm-001-029099142207
       SET ADO_PAT=your-azure-devops-personal-access-token
  3. The LLM calls go through AWS Bedrock (Claude). Make sure your SSO session is active.

Usage:
  python scripts/describe_ado.py prs
  python scripts/describe_ado.py commits <base_commit> <target_commit>

Output:
  prs     -> ./pr_descriptions/<repo_name>/<pr_id>_<title>.md
  commits -> ./commit_descriptions/<repo_name>/<base>..<target>.md
"""

import asyncio
import base64
import difflib
import os
import re
import sys
from pathlib import Path
from urllib.parse import quote

# Ensure AWS Bedrock credentials are picked up via SSO profile
os.environ.setdefault("AWS_PROFILE", "ssop-all-all-allx-accadm-001-029099142207")
os.environ.setdefault("AWS_USE_IMDS", "true")
os.environ.setdefault("AWS_REGION_NAME", "us-east-1")

import dotenv
dotenv.load_dotenv()
import requests

# ---------------------------------------------------------------------------
# Azure DevOps configuration
# ---------------------------------------------------------------------------
ADO_ORG = "https://dev.azure.com/jblprd"
ADO_PROJECT = "JGP Common Data Platform"
REPOS = ["GenAI-Quoting-and-Scheduling","GenAI-Engineering-Verification-Test-Planning","GenAI-Intelligence-GenRFQ"]
ADO_REPO = REPOS[1]

MODEL = "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"

# Resolve project root (parent of scripts/) so output paths and imports work correctly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pr_agent.agent.pr_agent import PRAgent
from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler
from pr_agent.config_loader import get_settings
from pr_agent.log import get_logger, setup_logger

setup_logger("INFO")
logger = get_logger()

# ---------------------------------------------------------------------------
# LLM prompts for commit comparison
# ---------------------------------------------------------------------------
COMMIT_SYSTEM_PROMPT = """\
You are a code change analyst. You will be given a diff between two commits in a repository.
Analyze the changes and produce a structured markdown description including:

1. **Overview**: A concise summary (2-3 sentences) of what changed overall.
2. **Change Type**: Categorize (e.g., Bug fix, Feature, Refactor, Config change, Documentation, etc.)
3. **Key Changes**: A bullet list of the most important changes, grouped by file or component.
4. **Impact Assessment**: What areas of the codebase are affected and potential risks.

Keep the description clear and useful for someone reviewing the commit history.
Do NOT include the raw diff in your output.
"""

COMMIT_USER_PROMPT_TEMPLATE = """\
Here is the diff between commit `{base_commit}` and commit `{target_commit}` in repository `{repo}`:

```diff
{diff_content}
```

Please provide a structured description of these changes.
"""


# ===========================================================================
# Shared utilities
# ===========================================================================

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


def check_aws_bedrock_connection():
    """Verify AWS Bedrock is reachable with current credentials."""
    import boto3
    import botocore.exceptions

    print("Checking AWS Bedrock connection...")
    try:
        session = boto3.Session()
        creds = session.get_credentials()
        if not creds:
            print("  ✗ No AWS credentials found. Run: aws sso login --profile ssop-all-all-allx-accadm-001-029099142207")
            sys.exit(1)

        frozen = creds.get_frozen_credentials()
        if not frozen.access_key:
            print("  ✗ AWS credentials are empty. SSO session may have expired.")
            print("    Run: aws sso login --profile ssop-all-all-allx-accadm-001-029099142207")
            sys.exit(1)

        bedrock_client = session.client("bedrock", region_name=os.environ.get("AWS_REGION_NAME", "us-east-1"))
        bedrock_client.list_foundation_models(byOutputModality="TEXT")
        print("  ✓ AWS Bedrock connection OK.\n")
    except botocore.exceptions.NoCredentialsError:
        print("  ✗ No AWS credentials found. Run: aws sso login --profile ssop-all-all-allx-accadm-001-029099142207")
        sys.exit(1)
    except botocore.exceptions.TokenRetrievalError as e:
        print(f"  ✗ AWS SSO token expired or invalid: {e}")
        print("    Run: aws sso login --profile ssop-all-all-allx-accadm-001-029099142207")
        sys.exit(1)
    except botocore.exceptions.ClientError as e:
        print(f"  ✗ AWS Bedrock access error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"  ✗ Unexpected AWS error: {e}")
        sys.exit(1)


def check_ado_connection(access_token: str):
    """Verify Azure DevOps PAT is valid by calling a lightweight API."""
    print("Checking Azure DevOps connection...")
    b64_pat = base64.b64encode(f":{access_token}".encode()).decode()
    headers = {"Authorization": f"Basic {b64_pat}"}
    url = f"{ADO_ORG}/_apis/projects?api-version=7.1&$top=1"

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            print("  ✓ Azure DevOps PAT is valid.\n")
        elif response.status_code == 401:
            print("  ✗ Azure DevOps PAT is invalid or expired.")
            print("    Generate a new one at: https://dev.azure.com/jblprd/_usersSettings/tokens")
            sys.exit(1)
        elif response.status_code == 403:
            print("  ✗ Azure DevOps PAT lacks permissions. Ensure Code (Read) scope.")
            sys.exit(1)
        else:
            print(f"  ✗ Unexpected ADO response: {response.status_code} {response.text[:200]}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("  ✗ Cannot reach dev.azure.com. Check network/proxy.")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("  ✗ Connection to dev.azure.com timed out.")
        sys.exit(1)


# ===========================================================================
# PR description (uses PR-Agent /describe)
# ===========================================================================

def get_all_pull_requests(access_token: str):
    """Fetch all completed (merged) pull requests from Azure DevOps REST API, with pagination."""
    project_encoded = quote(ADO_PROJECT)
    repo_encoded = quote(ADO_REPO)
    url = f"{ADO_ORG}/{project_encoded}/_apis/git/repositories/{repo_encoded}/pullrequests"
    headers = get_ado_headers(access_token)

    all_prs = []
    skip = 0
    top = 1000

    while True:
        params = {
            "searchCriteria.status": "completed",
            "api-version": "7.1",
            "$top": top,
            "$skip": skip,
        }
        response = requests.get(url, params=params, headers=headers)
        if response.status_code != 200:
            print(f"ERROR: Failed to list PRs. Status: {response.status_code}")
            print(response.text)
            sys.exit(1)

        data = response.json()
        batch = data.get("value", [])
        if not batch:
            break

        all_prs.extend(batch)
        print(f"  Fetched {len(all_prs)} PRs so far...")

        if len(batch) < top:
            break
        skip += top

    return all_prs


def build_pr_url(pr_id: int) -> str:
    """Build the full Azure DevOps PR URL that PR-Agent expects."""
    project_encoded = quote(ADO_PROJECT)
    repo_encoded = quote(ADO_REPO)
    return f"{ADO_ORG}/{project_encoded}/_git/{repo_encoded}/pullrequest/{pr_id}"


async def describe_single_pr(pr_url: str):
    """Run /describe on a single PR."""
    agent = PRAgent()
    return await agent.handle_request(pr_url, ["describe"])


async def cmd_prs():
    """Describe all completed PRs and save to local markdown."""
    # Configure PR-Agent
    get_settings().set("azure_devops.org", ADO_ORG)
    get_settings().set("config.git_provider", "azure")
    get_settings().set("config.model", MODEL)
    get_settings().set("config.fallback_models", [MODEL])
    get_settings().set("config.publish_output", False)

    # Output directory
    output_dir = PROJECT_ROOT / "pr_descriptions" / ADO_REPO
    output_dir.mkdir(parents=True, exist_ok=True)

    # Migrate any old flat files into the repo subfolder
    base_dir = PROJECT_ROOT / "pr_descriptions"
    for old_file in base_dir.glob("*.md"):
        dest = output_dir / old_file.name
        if not dest.exists():
            old_file.rename(dest)
            print(f"  Migrated: {old_file.name} -> {ADO_REPO}/")
        else:
            print(f"  Skipped (already exists): {old_file.name}")

    print("Reading ADO_PAT from environment...")
    access_token = get_ado_pat()
    print("  ✓ PAT loaded.\n")

    check_ado_connection(access_token)
    check_aws_bedrock_connection()

    get_settings().set("azure_devops.pat", access_token)

    print(f"Fetching all completed PRs from {ADO_ORG}/{ADO_PROJECT}/_git/{ADO_REPO} ...")
    pull_requests = get_all_pull_requests(access_token)

    if not pull_requests:
        print("No completed pull requests found.")
        return

    print(f"\nFound {len(pull_requests)} completed PR(s). Starting /describe...\n")

    for i, pr in enumerate(pull_requests, 1):
        pr_id = pr["pullRequestId"]
        title = pr.get("title", "(no title)")
        pr_url = build_pr_url(pr_id)

        # Skip if already exists
        existing = list(output_dir.glob(f"{pr_id}_*.md"))
        if existing:
            print(f"[{i}/{len(pull_requests)}] Skipping PR #{pr_id} (already exists: {existing[0].name})")
            continue

        print(f"[{i}/{len(pull_requests)}] Describing PR #{pr_id}: {title}")
        print(f"  URL: {pr_url}")

        try:
            await describe_single_pr(pr_url)
            artifact = getattr(get_settings(), "data", None)
            if artifact and isinstance(artifact, dict):
                body = artifact.get("artifact", "")
            elif artifact:
                body = str(artifact)
            else:
                body = ""

            if body:
                safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)[:80]
                filename = f"{pr_id}_{safe_title}.md"
                filepath = output_dir / filename

                md_content = f"# PR #{pr_id}: {title}\n\n"
                md_content += f"**URL:** {pr_url}\n\n"
                md_content += f"**Author:** {pr.get('createdBy', {}).get('displayName', 'Unknown')}\n\n"
                md_content += f"**Created:** {pr.get('creationDate', 'Unknown')}\n\n"
                md_content += f"**Closed:** {pr.get('closedDate', 'Unknown')}\n\n"
                md_content += "---\n\n"
                md_content += body

                filepath.write_text(md_content, encoding="utf-8")
                print(f"  ✓ Saved to {filepath}")
            else:
                print(f"  ✗ No description generated")
        except Exception as e:
            print(f"  ✗ Error: {e}")

        print()

    print(f"All PRs processed. Descriptions saved to: {output_dir.resolve()}")


# ===========================================================================
# Commit comparison (direct LLM call via LiteLLMAIHandler)
# ===========================================================================

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
            old_url = (
                f"{ADO_ORG}/{project_encoded}/_apis/git/repositories/{repo_encoded}"
                f"/items?path={quote(file_path)}&versionDescriptor.version={base_commit}"
                f"&versionDescriptor.versionType=commit&api-version=7.1"
            )
            resp = requests.get(old_url, headers=headers)
            if resp.status_code == 200:
                lines = resp.text.splitlines()
                for line in lines[:50]:
                    diff_output.append(f"- {line}")
                if len(lines) > 50:
                    diff_output.append(f"  ... ({len(lines) - 50} more lines)")
            diff_output.append("")
            continue

        if change_type == "add":
            new_url = (
                f"{ADO_ORG}/{project_encoded}/_apis/git/repositories/{repo_encoded}"
                f"/items?path={quote(file_path)}&versionDescriptor.version={target_commit}"
                f"&versionDescriptor.versionType=commit&api-version=7.1"
            )
            resp = requests.get(new_url, headers=headers)
            if resp.status_code == 200:
                lines = resp.text.splitlines()
                for line in lines[:100]:
                    diff_output.append(f"+ {line}")
                if len(lines) > 100:
                    diff_output.append(f"  ... ({len(lines) - 100} more lines)")
            diff_output.append("")
            continue

        if change_type in ("edit", "rename"):
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
                old_lines = old_resp.text.splitlines(keepends=True)
                new_lines = new_resp.text.splitlines(keepends=True)
                unified = list(difflib.unified_diff(
                    old_lines, new_lines,
                    fromfile=f"a{file_path}",
                    tofile=f"b{file_path}",
                    n=3
                ))
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


def get_commit_diff(access_token: str, base_commit: str, target_commit: str) -> str:
    """Fetch the diff between two commits using Azure DevOps REST API."""
    project_encoded = quote(ADO_PROJECT)
    repo_encoded = quote(ADO_REPO)
    headers = get_ado_headers(access_token)

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

    return fetch_text_diffs(access_token, base_commit, target_commit, changes)


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


async def describe_diff_with_llm(diff_content: str, base_commit: str, target_commit: str) -> str:
    """Call the LLM to generate a description of the diff."""
    get_settings().set("config.model", MODEL)
    get_settings().set("config.fallback_models", [MODEL])
    get_settings().set("config.ai_timeout", 120)

    ai_handler = LiteLLMAIHandler()
    user_prompt = COMMIT_USER_PROMPT_TEMPLATE.format(
        base_commit=base_commit[:8],
        target_commit=target_commit[:8],
        repo=ADO_REPO,
        diff_content=diff_content[:50000]
    )

    response, finish_reason = await ai_handler.chat_completion(
        model=MODEL,
        system=COMMIT_SYSTEM_PROMPT,
        user=user_prompt,
        temperature=0.2,
    )
    return response


async def cmd_commits(base_commit: str = None, target_commit: str = None):
    """Compare two commits, or describe all commits (each vs parent) if none specified."""
    output_dir = PROJECT_ROOT / "commit_descriptions" / ADO_REPO
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Reading ADO_PAT from environment...")
    access_token = get_ado_pat()
    print("  ✓ PAT loaded.\n")

    check_ado_connection(access_token)
    check_aws_bedrock_connection()

    if base_commit and target_commit:
        # Describe a specific pair
        await _describe_commit_pair(access_token, base_commit, target_commit, output_dir)
    else:
        # Describe all commits (each commit vs its parent)
        await _describe_all_commits(access_token, output_dir)


def get_all_commits(access_token: str):
    """Fetch all commits from Azure DevOps REST API, with pagination."""
    project_encoded = quote(ADO_PROJECT)
    repo_encoded = quote(ADO_REPO)
    url = f"{ADO_ORG}/{project_encoded}/_apis/git/repositories/{repo_encoded}/commits"
    headers = get_ado_headers(access_token)

    all_commits = []
    skip = 0
    top = 1000

    while True:
        params = {
            "api-version": "7.1",
            "$top": top,
            "$skip": skip,
        }
        response = requests.get(url, params=params, headers=headers)
        if response.status_code != 200:
            print(f"ERROR: Failed to list commits. Status: {response.status_code}")
            print(response.text)
            sys.exit(1)

        data = response.json()
        batch = data.get("value", [])
        if not batch:
            break

        all_commits.extend(batch)
        print(f"  Fetched {len(all_commits)} commits so far...")

        if len(batch) < top:
            break
        skip += top

    return all_commits


async def _describe_all_commits(access_token: str, output_dir: Path):
    """Describe each commit by comparing it to its parent."""
    print(f"Fetching all commits from {ADO_ORG}/{ADO_PROJECT}/_git/{ADO_REPO} ...")
    commits = get_all_commits(access_token)

    if not commits:
        print("No commits found.")
        return

    print(f"\nFound {len(commits)} commit(s). Starting describe...\n")

    for i, commit in enumerate(commits, 1):
        commit_id = commit["commitId"]
        comment = commit.get("comment", "(no message)")
        parents = commit.get("parents", [])

        if not parents:
            print(f"[{i}/{len(commits)}] Skipping {commit_id[:8]} (no parent — initial commit)")
            continue

        parent_id = parents[0]

        # Skip if already described
        output_file = output_dir / f"{parent_id[:8]}..{commit_id[:8]}.md"
        if output_file.exists():
            print(f"[{i}/{len(commits)}] Skipping {commit_id[:8]} (already exists)")
            continue

        print(f"[{i}/{len(commits)}] Describing {commit_id[:8]}: {comment[:60]}")

        try:
            await _describe_commit_pair(access_token, parent_id, commit_id, output_dir)
        except Exception as e:
            print(f"  ✗ Error: {e}")

        print()

    print(f"All commits processed. Descriptions saved to: {output_dir.resolve()}")


async def _describe_commit_pair(access_token: str, base_commit: str, target_commit: str, output_dir: Path):
    """Describe a single commit pair and save to markdown."""
    # Skip if already described
    output_file = output_dir / f"{base_commit[:8]}..{target_commit[:8]}.md"
    if output_file.exists():
        print(f"  Already described: {output_file.name}")
        return

    diff_content = get_commit_diff(access_token, base_commit, target_commit)

    if not diff_content.strip():
        print(f"  ✗ No diff found between {base_commit[:8]} and {target_commit[:8]}")
        return

    print(f"  Diff fetched ({len(diff_content)} chars), generating description...")

    # Get commit metadata
    base_info = get_commit_info(access_token, base_commit)
    target_info = get_commit_info(access_token, target_commit)

    description = await describe_diff_with_llm(diff_content, base_commit, target_commit)

    # Build markdown output
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
    print(f"  ✓ Saved to {output_file.name}")


# ===========================================================================
# CLI entry point
# ===========================================================================

def print_usage():
    print("Usage:")
    print("  python scripts/describe_ado.py              (run both prs and commits)")
    print("  python scripts/describe_ado.py prs")
    print("  python scripts/describe_ado.py commits")
    print("  python scripts/describe_ado.py commits <base_commit> <target_commit>")
    print("")
    print("Subcommands:")
    print("  (none)   Run both prs and commits in sequence")
    print("  prs      Describe all completed PRs and save to ./pr_descriptions/<repo>/")
    print("  commits  Describe all commits (each vs parent) or two specific commits")
    print("")
    print("Examples:")
    print("  python scripts/describe_ado.py")
    print("  python scripts/describe_ado.py prs")
    print("  python scripts/describe_ado.py commits")
    print("  python scripts/describe_ado.py commits abc1234 def5678")


async def main():
    if len(sys.argv) < 2:
        # No subcommand — run both prs and commits
        print("=" * 60)
        print("  Running: describe PRs")
        print("=" * 60)
        await cmd_prs()
        print("\n")
        print("=" * 60)
        print("  Running: describe commits")
        print("=" * 60)
        await cmd_commits()
        return

    command = sys.argv[1].lower()

    if command == "prs":
        await cmd_prs()
    elif command == "commits":
        if len(sys.argv) >= 4:
            base_commit = sys.argv[2]
            target_commit = sys.argv[3]
            await cmd_commits(base_commit, target_commit)
        else:
            await cmd_commits()
    elif command in ("--help", "-h", "help"):
        print_usage()
    else:
        print(f"Unknown command: {command}")
        print("")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
