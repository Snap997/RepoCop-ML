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

def get_issues(repo_url, state, per_page, max_pages, filename):
    """Retrieve issues from the GitHub API and save in batches."""
    print(f"Retrieving {state} issues from {repo_url}")
    base_url = f"https://api.github.com/repos/{repo_url}/issues"

    batch_issues = []  # Store issues for the current batch
    total_issues = 0   # Keep track of total issues fetched

    # Open the CSV file for writing
    with open(filename, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["id", "title", "body", "assignees", "created_at", "closed_at", "labels"])
        writer.writeheader()  # Write CSV header

        for page in range(1, max_pages + 1):
            if page % 50 == 0:
                print(f"Fetching page {page}...")
            params = {
                "state": state,
                "per_page": per_page,
                "page": page
            }

            try:
                response = req.get(base_url, headers=HEADERS, params=params, timeout=10)
                response.raise_for_status()  # Raise an error for HTTP status codes >= 400
                page_issues = response.json()

                # Stop if no more issues are returned
                if not page_issues or "rel=\"next\"" not in str(response.headers.get("Link", "")):
                    print("No more issues found. Stopping early.")
                    break

                # Process issues and add to the batch
                for issue in page_issues:
                    batch_issues.append({
                        "id": issue.get("id", None),
                        "title": (issue.get("title") or "").replace("\n", " ").strip(),
                        "body": (issue.get("body") or "").replace("\n", " ").strip(),
                        "assignees": ", ".join([str(assignee.get("id", "")) for assignee in issue.get("assignees", [])]),
                        "created_at": issue.get("created_at", None),
                        "closed_at": issue.get("closed_at", None),
                        "labels": ", ".join([label.get("name", "") for label in issue.get("labels", [])])
                    })

                # Write to CSV after every batch
                if page % BATCH_SIZE == 0:
                    print(f"Saving batch of {len(batch_issues)} issues to file...")
                    writer.writerows(batch_issues)
                    total_issues += len(batch_issues)
                    batch_issues.clear()  # Clear batch after saving

                # Avoid hitting rate limits
                if WAIT_TIME > 0:
                    time.sleep(WAIT_TIME / 1000)

            except req.exceptions.RequestException as e:
                print(f"Error fetching page {page} for {repo_url}: {e}")
                break

        # Write remaining issues in the last batch
        if batch_issues:
            print(f"Saving final batch of {len(batch_issues)} issues to file...")
            writer.writerows(batch_issues)
            total_issues += len(batch_issues)

    print(f"Saved a total of {total_issues} issues to '{filename}'.")


for repo in REPOS:
    filename = os.path.join(DESTINATION_FOLDER, "raw", f"{repo.split('/')[-1]}_closed_issues.csv")
    get_issues(repo, STATE, PER_PAGE, MAX_PAGES, filename)
    print(f"Finished processing repository: {repo}")
