import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

directory_path = os.getenv("DATA_PATH")


# Change most of these making them project based
def get_average_time_to_close():
    """Calculates the average time to close issues."""
    pass

def get_median_time_to_close():
    """Calculates the median time to close issues."""
    return df['time_to_close'].median()

def get_top_assignees(n=10):
    """Returns the top N assignees by number of issues assigned."""
    assignee_counts = (
        df['assignees']
        .explode()  # Expand lists into separate rows
        .value_counts()
    )
    return assignee_counts.head(n)

def get_top_labels(n=10):
    """Returns the top N labels used across issues."""
    label_counts = (
        df['labels']
        .explode()  # Expand lists into separate rows
        .value_counts()
    )
    return label_counts.head(n)

def get_unassigned_issues():
    """Returns a DataFrame of unassigned issues."""
    return df[df['assignees'].apply(len) == 0]

def get_reassigned_issues():
    """Returns a DataFrame of issues that were reassigned."""
    # Assuming you track reassignments in a separate field (e.g., 'reassignment_count')
    if 'reassignment_count' in df.columns:
        return df[df['reassignment_count'] > 0]
    return pd.DataFrame(columns=df.columns)  # Return empty if not available

def get_issues_with_comments():
    """Returns a DataFrame of issues that have comments."""
    return df[df['comments'] > 0]

def get_issues_by_state(state='closed'):
    """Filters issues by state (e.g., 'open', 'closed')."""
    return df[df['state'] == state]

def get_issues_by_time_range(start_date, end_date):
    """Filters issues by a time range based on their creation date."""
    mask = (df['created_at'] >= pd.to_datetime(start_date)) & (df['created_at'] <= pd.to_datetime(end_date))
    return df[mask]


dataframes = []
# Scan the directory for files with a .csv extension
for filename in os.listdir(directory_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory_path, filename)
        # Read the CSV file into a DataFrame and add it to the list
        df = pd.read_csv(file_path)
        dataframes.append((filename, df))
        print(f"File read: {filename}")


combined_df = pd.concat([df for _, df in dataframes], ignore_index=True)


#DB SIZE
print(f"Dataset has {len(combined_df)} rows")

# Number of issues per developer
print("Issue per developer")
issues_per_user = combined_df['assignees'].value_counts()
print(issues_per_user.head(10))



# Drop rows with negative or missing 'time_to_close' values
combined_df = combined_df[combined_df['time_to_close'] >= 0]

# Calculate the average time to close an issue
average_time_to_close = combined_df['time_to_close'].mean()

# Output the result as a double
print(f"Average Time to Close an Issue: {average_time_to_close:.2f} days")
