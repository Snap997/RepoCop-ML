
# GitHub Issue Assigner using Machine Learning

## Introduction

This project aims to solve the problem of unassigned open issues in large GitHub repositories by developing a machine learning model that automatically assigns new issues to the best-suited contributor based on their past activity. The model utilizes data from GitHub repositories, including issues, assignees, and user activities. 

---

## Installation

To get started with the project, follow these steps:

### 1. Install Dependencies

First, install the required Python dependencies by using `pip`. Make sure you have `pip` installed on your machine.

```bash
pip install -r requirements.txt
```

### 2. Create .env File

Create a `.env` file in the root directory of the project and add the following variable:

```
GITHUB_TOKEN=gh_YOUR_GITHUB_TOKEN
DATA_PATH=path/of/the/project/
```

You can update the `DATA` path as needed based on your environment or data location.

### 3. Modify and Run the Code

Next, modify and run the `tempCodeRunner.py` file to get started with the data processing and machine learning workflow.

```bash
python tempCodeRunner.py
```

---

## Data Fetching from Google BigQuery

If you'd like to fetch data directly from Google BigQuery, follow the steps below. The following SQL queries will help you extract and structure the data needed for the project.

### Query 1: Extract Issue Data from GitHub Archive

This query extracts relevant issue data from the GitHub archive for the year 2021.

```sql
SELECT
    JSON_EXTRACT(payload, "$.issue.id") AS issue_id,
    JSON_EXTRACT(payload, "$.issue.title") AS issue_title,
    JSON_EXTRACT(payload, "$.issue.body") AS issue_body,
    JSON_EXTRACT(payload, "$.issue.state") AS issue_state,
    JSON_EXTRACT(payload, "$.issue.user.id") AS author_id,
    ARRAY(
        SELECT JSON_EXTRACT_SCALAR(label, "$.name")
        FROM UNNEST(JSON_EXTRACT_ARRAY(payload, "$.issue.labels")) AS label
    ) AS issue_labels,
    ARRAY(
        SELECT JSON_EXTRACT_SCALAR(assignee, "$.id")
        FROM UNNEST(JSON_EXTRACT_ARRAY(payload, "$.issue.assignees")) AS assignee
    ) AS issue_assignees,
    JSON_EXTRACT(payload, "$.issue.created_at") AS issue_created_at,
    JSON_EXTRACT(payload, "$.issue.closed_at") AS issue_closed_at,
    repo.id AS repo_id,
    repo.name AS repo_name,
    created_at AS event_created_at
FROM
    `githubarchive.year.2021`
WHERE
    type = 'IssuesEvent'
    AND JSON_EXTRACT(payload, "$.issue") IS NOT NULL
```

### Query 2: Count Issues per Assignee

This query counts the number of issues per assignee, filtering for assignees with at least 50 assigned issues.

```sql
WITH AssigneeIssueCounts AS (
    SELECT
        CAST(issue_assignees[OFFSET(0)] AS STRING) AS assignee,
        COUNT(*) AS issue_count,
        repo_id
    FROM
        "The repo from above"
    WHERE
        ARRAY_LENGTH(issue_assignees) > 0
    GROUP BY
        assignee, repo_id
    HAVING
        COUNT(*) >= 50
),
...
-- Continue with the rest of the query logic
```

Make sure to adjust the queries according to your specific dataset and project needs.

---

## Contributing

Feel free to contribute to this project by opening issues, suggesting features, or submitting pull requests. Please ensure to follow the repository's guidelines and contribute positively!

---

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
