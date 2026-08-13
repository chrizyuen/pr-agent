"""
Describe all completed PRs for an Azure DevOps repository using PR-Agent.
Saves descriptions as local markdown files instead of updating PRs remotely.

Prerequisites:
  1. AWS SSO login:  aws sso login --profile ssop-all-all-allx-accadm-001-029099142207
  2. Environment variables (set before running):
       SET AWS_PROFILE=ssop-all-all-allx-accadm-001-029099142207
       SET ADO_PAT=your-azure-devops-personal-access-token
  3. The LLM calls go through AWS Bedrock (Claude). Make sure your SSO session is active.

Usage:
  python describe_all_prs.py

Output:
  Markdown files saved to ./pr_descriptions/<pr_id>.md
"""

import asyncio
import base64
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

# Azure DevOps configuration
ADO_ORG = "https://dev.azure.com/jblprd"
ADO_PROJECT = "JGP Common Data Platform"
ADO_REPO = "GenAI-PL-Invoice"

# PR-Agent needs PYTHONPATH=. to resolve imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pr_agent.agent.pr_agent import PRAgent
from pr_agent.config_loader import get_settings
from pr_agent.log import get_logger, setup_logger

setup_logger("INFO")
logger = get_logger()


def get_ado_pat() -> str:
    """Get an Azure DevOps PAT from the ADO_PAT environment variable."""
    pat = os.environ.get("ADO_PAT")
    if not pat:
        print("ERROR: ADO_PAT environment variable is not set.")
        print("  Generate a PAT at: https://dev.azure.com/jblprd/_usersSettings/tokens")
        print("  Then set it:  SET ADO_PAT=your-token-here")
        sys.exit(1)
    return pat


def get_all_pull_requests(access_token: str):
    """Fetch all completed (merged) pull requests from Azure DevOps REST API, with pagination."""
    project_encoded = quote(ADO_PROJECT)
    repo_encoded = quote(ADO_REPO)
    url = f"{ADO_ORG}/{project_encoded}/_apis/git/repositories/{repo_encoded}/pullrequests"
    b64_pat = base64.b64encode(f":{access_token}".encode()).decode()
    headers = {
        "Authorization": f"Basic {b64_pat}"
    }

    all_prs = []
    skip = 0
    top = 1000  # max page size

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


async def describe_pr(pr_url: str):
    """Run /describe on a single PR."""
    agent = PRAgent()
    result = await agent.handle_request(pr_url, ["describe"])
    return result


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

        # Resolve credentials to verify they're not expired
        frozen = creds.get_frozen_credentials()
        if not frozen.access_key:
            print("  ✗ AWS credentials are empty. SSO session may have expired.")
            print("    Run: aws sso login --profile ssop-all-all-allx-accadm-001-029099142207")
            sys.exit(1)

        # Try a lightweight Bedrock call to confirm access
        client = session.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION_NAME", "us-east-1"))
        # list_foundation_models is on the bedrock client, not bedrock-runtime
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
            print("  ✗ Azure DevOps PAT lacks permissions. Ensure Code (Read & Write) scope.")
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


async def main():
    # Configure PR-Agent for Azure DevOps
    get_settings().set("azure_devops.org", ADO_ORG)
    get_settings().set("config.git_provider", "azure")
    get_settings().set("config.model", "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0")
    get_settings().set("config.fallback_models", ["bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"])

    # Disable publishing to Azure DevOps — save locally instead
    get_settings().set("config.publish_output", False)

    # Output directory: pr_descriptions/<repo_name>/
    output_dir = Path("pr_descriptions") / ADO_REPO
    output_dir.mkdir(parents=True, exist_ok=True)

    # Migrate any existing .md files from the old flat pr_descriptions/ into the repo subfolder
    base_dir = Path("pr_descriptions")
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

    # Pre-flight checks
    check_ado_connection(access_token)
    check_aws_bedrock_connection()

    # Also set the PAT in settings so PR-Agent's internal provider can use it
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

        # Skip if a markdown file for this PR already exists
        existing = list(output_dir.glob(f"{pr_id}_*.md"))
        if existing:
            print(f"[{i}/{len(pull_requests)}] Skipping PR #{pr_id} (already exists: {existing[0].name})")
            continue

        print(f"[{i}/{len(pull_requests)}] Describing PR #{pr_id}: {title}")
        print(f"  URL: {pr_url}")

        try:
            result = await describe_pr(pr_url)
            # When publish_output=False, PR-Agent stores the body in get_settings().data
            artifact = getattr(get_settings(), "data", None)
            if artifact and isinstance(artifact, dict):
                body = artifact.get("artifact", "")
            elif artifact:
                body = str(artifact)
            else:
                body = ""

            if body:
                # Sanitize title for use in filename
                safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)[:80]
                filename = f"{pr_id}_{safe_title}.md"
                filepath = output_dir / filename

                # Write markdown with PR metadata header
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


if __name__ == "__main__":
    asyncio.run(main())
