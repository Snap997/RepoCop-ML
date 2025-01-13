import os
import subprocess

if __name__ == "__main__":
    directory_path = os.getenv("DATA_PATH") + "scripts/"
    if directory_path:
        subprocess.run(["python", "scripts/data_retrival.py"])
        subprocess.run(["python", "scripts/data_cleaning.py"])
        subprocess.run(["python", "scripts/train_models.py"])
        subprocess.run(["python", "scripts/evaluate_models.py"])
    else:
        print("DATA_PATH environment variable is not set")
