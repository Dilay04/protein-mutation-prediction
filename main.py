import pandas as pd
from src.utils import get_stratified_folds, evaluate_predictions, SEED

DATA_PATH = "data/raw/YARISMA_TRAIN_MASTER.csv"
TARGET_COL = "Label"

def main():
    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())

    if TARGET_COL in df.columns:
        print("Label distribution:")
        print(df[TARGET_COL].value_counts())

        folds = get_stratified_folds(df, target_col=TARGET_COL, n_splits=5)
        print(f"Number of folds: {len(folds)}")

        for fold_id, (train_idx, valid_idx) in enumerate(folds, start=1):
            train_df = df.iloc[train_idx]
            valid_df = df.iloc[valid_idx]

            print(f"\nFold {fold_id}")
            print("Train label distribution:")
            print(train_df[TARGET_COL].value_counts(normalize=True))
            print("Validation label distribution:")
            print(valid_df[TARGET_COL].value_counts(normalize=True))

    else:
        print(f"Target column '{TARGET_COL}' not found.")


if __name__ == "__main__":
    main()
