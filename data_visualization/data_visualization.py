import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

directory_path = os.getenv("DATA_PATH")

# List to store DataFrames from each CSV file
dataframes = []

# Scan the directory for files with a .csv extension
for filename in os.listdir(directory_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory_path, filename)
        # Read the CSV file into a DataFrame and add it to the list
        df = pd.read_csv(file_path)
        dataframes.append((filename, df))
        print(f"File read: {filename}")

# Optionally, combine all DataFrames into a single DataFrame
combined_df = pd.concat([df for _, df in dataframes], ignore_index=True)

# Print the combined DataFrame or perform further operations
print(combined_df.head())

# Number of issues per developer
issues_per_user = combined_df['assignees'].value_counts()
print(issues_per_user.head(10))

# Summary statistics for numerical columns
print(combined_df.describe())

# Ensure 'created_at' and 'closed_at' are in datetime format
# Calculate the time taken to close issues (in days)
combined_df['created_at'] = pd.to_datetime(combined_df['created_at'], errors='coerce')
combined_df['closed_at'] = pd.to_datetime(combined_df['closed_at'], errors='coerce')
combined_df['time_to_close'] = (combined_df['closed_at'] - combined_df['created_at']).dt.days

# Drop rows with negative or missing 'time_to_close' values
combined_df = combined_df[combined_df['time_to_close'] >= 0]

# Calculate the average time to close an issue
average_time_to_close = combined_df['time_to_close'].mean()

# Output the result as a double
print(f"Average Time to Close an Issue: {average_time_to_close:.2f} days")
