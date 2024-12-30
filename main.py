import os

def main():

    print("Step 1: Training models...")
    os.system("python scripts/train_models.py")

    print("Step 2: Evaluating models...")
    os.system("python scripts/evaluate_models.py")

if __name__ == "__main__":
    main()
