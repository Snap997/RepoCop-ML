import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import os

load_dotenv()

# Carica il percorso dei dati dal file .env
directory_path = os.getenv("DATA_PATH") 

data = pd.read_csv(
    f"{directory_path}data/processed/combined_issues.csv",
    low_memory=False
)

# Identify embedding feature columns
embedding_columns = [col for col in data.columns if "_embedding_dim" in col]

# Select features (X) and target variable (y)
X = data[embedding_columns]
y = data['assignee']  # Adjust this column name if necessary

# Identify embedding feature columns
embedding_columns = [col for col in data.columns if "embedding" in col]
embedding_columns.append("repo_id")  # Include `repo_id` for grouping

# Select features (X) and target variable (y)
X = data[embedding_columns]
y = data['assignee'].apply(str)  # Ensure `assignee` is treated as string

print("Class distribution in `y` before filtering:")
print(y.value_counts())

# Filter classes with more than one instance
class_counts = y.value_counts()
valid_classes = class_counts[class_counts > 1].index
data_filtered = data[data['assignee'].apply(str).isin(valid_classes)]

# Log filtering results
print("Original dataset size:", data.shape)
print("Filtered dataset size:", data_filtered.shape)
print("Class distribution in `y` after filtering:")
print(data_filtered['assignee'].value_counts())

# Reassign X and y after filtering
X = data_filtered[embedding_columns]
y = data_filtered['assignee'].apply(str)

# Check if dataset is empty after filtering
if X.empty or y.empty:
    raise ValueError("Filtered dataset is empty. Adjust filtering criteria or dataset preparation.")

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Define models to evaluate
model_files = ["logistic_regression.pkl", "random_forest.pkl", "xgboost.pkl"]
model_files = ["logistic_regression.pkl", "random_forest.pkl"]
metrics = {}

# Evaluate each model
for model_file in model_files:
    print(f"Evaluating {model_file}...")
    model = joblib.load(f"models/{model_file}")
    y_pred = model.predict(X_test)
    
    # Collect metrics
    metrics[model_file] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1 Score": f1_score(y_test, y_pred, average='weighted')
    }

# Display metrics
for model_name, metric_values in metrics.items():
    print(f"\nModel: {model_name}")
    for metric, value in metric_values.items():
        print(f"{metric}: {value:.4f}")
