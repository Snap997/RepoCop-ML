import csv
import time
import requests as req
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
DESTINATION_FOLDER = os.getenv("DATA_PATH")
STATE = "closed"  # We are fetching closed issues
PER_PAGE = 100    # Maximum allowed per page by GitHub API
MAX_PAGES = 10000    # Limit to avoid rate limiting; adjust as needed
WAIT_TIME = 0  # Time to wait between pages


# Headers for authentication
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

REPOS = [
         "tensorflow/tensorflow", 
         "flutter/flutter",
         "angular/angular",
         "opencv/opencv",
         "kubernetes/kubernetes",
         "rust-lang/rust",
         "microsoft/TypeScript",
         "nodejs/node"
         ]


print(f"Retriving {len(REPOS)} repos")


def get_issues(repoUrl, state, per_page, max_pages):
    """Retrieve issues from the GitHub API."""
    print(f"RETRIVING {state} issues from {repoUrl}")
    issues = []
    base_url = f"https://api.github.com/repos/{repoUrl}/issues"

    for page in range(1, max_pages + 1):
        if page%10 == 0:
            print(f"Fetching page {page}...")
        params = {
            "state": state,
            "per_page": per_page,
            "page": page
        }

        response = req.get(base_url, headers=HEADERS, params=params, timeout=10)

        if response.status_code == 200:
            page_issues = response.json()

            # Stop if no more issues are returned
            if "rel=\"next\"" not in str(response.headers["Link"]):
                print("No more issues found. Stopping early.")
                break

            # Collect relevant data from each issue
            for issue in page_issues:
                issues.append({
                    "id": issue.get("id"),
                    "title": issue.get("title"),
                    "body": issue.get("body"),
                    "assignees": [assignee.get("id") for assignee in issue.get("assignees")],
                    "created_at": issue.get("created_at"),
                    "closed_at": issue.get("closed_at"),
                    "labels": [label.get("name") for label in issue.get("labels", [])]   
                })

            # To avoid hitting rate limits
            if WAIT_TIME != 0:
                time.sleep(WAIT_TIME/1000)
        else:
            print(f"Failed to fetch page {page}: {response.status_code}")
            break

    return issues

for repo in REPOS:
    issues_data = get_issues(repo, STATE, PER_PAGE, MAX_PAGES)

    # Save the data to a CSV file
    filename = DESTINATION_FOLDER + "raw/" + repo.split("/")[-1] + "_closed_issues.csv"
    if issues_data:
        df = pd.DataFrame(issues_data)
        df.to_csv(
            filename,
            index=False,
            #escapechar='\\',       # Use a backslash to escape problematic characters
            #quoting=csv.QUOTE_MINIMAL  # Minimize quoting, only quote fields with special characters
        )

        print(f"Saved {len(issues_data)} closed issues to '{filename}'.")
    else:
        print("No issues were fetched.")
