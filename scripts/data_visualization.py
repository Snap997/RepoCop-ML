import pandas as pd
import matplotlib.pyplot as plt
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

def plot_distribution():
    plt.figure(figsize=(6, 6))
    labels = list(dataset_distribution.keys())
    values = list(dataset_distribution.values())
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title("Issue Distribution")
    plt.show()

dataset_distribution = {}
dataframes = []
# Scan the directory for files with a .csv extension
for filename in os.listdir(str(directory_path + "processed/")):
    if filename.endswith(".csv"):
        file_path = os.path.join(str(directory_path + "processed/"), filename)
        # Read the CSV file into a DataFrame and add it to the list
        df = pd.read_csv(file_path)
        dataframes.append((filename, df))
        dataset_distribution[filename] = df['id'].count()
        print(f"File read: {filename} with {df['id'].count()} issues")

combined_df = pd.concat([df for _, df in dataframes], ignore_index=True)


#DB SIZE
print(f"Dataset has {combined_df['id'].count()} issues")

plot_distribution()

# Number of issues per developer
print("Issue per developer")
issues_per_user = combined_df['assignees'].value_counts()
print(f"Total developers {issues_per_user.count()}")
print(issues_per_user.head(50))


# Calculate the average time to close an issue
average_time_to_close = combined_df['time_to_close'].mean()

# Output the result as a double
print(f"Average Time to Close an Issue: {average_time_to_close:.2f} days")
