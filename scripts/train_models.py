import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import joblib
from dotenv import load_dotenv
import os
load_dotenv()

directory_path = os.getenv("DATA_PATH") 
# Load the processed dataset
data = pd.read_csv(f"{directory_path}data/processed/combined_issues.csv")

# Extract features and target
X = data[['title_embeddings', 'body_embeddings', 'encoded_labels']]  # Adjust as needed
y = data['assignees']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Train each model and save
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    joblib.dump(model, f"{directory_path}models/{name.replace(' ', '_').lower()}.pkl")
    print(f"Saved {name} model to ../models/{name.replace(' ', '_').lower()}.pkl")

