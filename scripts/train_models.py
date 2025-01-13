import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import joblib
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Load the directory path from the .env file
directory_path = os.getenv("DATA_PATH")

# Load the dataset with low_memory=False to prevent DtypeWarning
data = pd.read_csv(
    f"{directory_path}data/processed/combined_issues.csv",
    low_memory=False
)

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

# Log the shapes of the splits
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)

# Log samples of the datasets
print("\nSample of X_train:")
print(X_train.head())

print("\nSample of y_train:")
print(y_train.head())

print("\nSample of X_test:")
print(X_test.head())

print("\nSample of y_test:")
print(y_test.head())

# Define models 
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=500, max_depth=30, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Train and save models
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    model_path = f"{directory_path}models/{name.replace(' ', '_').lower()}.pkl"
    joblib.dump(model, model_path)
    print(f"Saved {name} model to {model_path}")
