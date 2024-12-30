import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load testing data
X_test = pd.read_csv("data/processed/testing_data.csv")
y_test = pd.read_csv("data/processed/testing_labels.csv")

# Define models to evaluate
model_files = ["logistic_regression.pkl", "random_forest.pkl", "xgboost.pkl"]
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
