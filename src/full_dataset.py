from pathlib import Path

import pandas as pd

from data import load_wikimia


def prepare_full_wikimia(length: int = 64) -> pd.DataFrame:
    """
    Load, inspect, validate, and save the full WikiMIA dataset split.
    """

    Path("data").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    print(f"Loading WikiMIA_length{length}...")

    df = load_wikimia(length=length)

    print("\nDATASET LOADED SUCCESSFULLY")
    print(f"Rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    # Check missing values
    missing_text = df["text"].isna().sum()
    missing_label = df["label"].isna().sum()

    # Check duplicates
    duplicate_text = df["text"].duplicated().sum()

    # Check label distribution
    label_counts = df["label"].value_counts().sort_index()

    print("\nLABEL DISTRIBUTION")
    print(label_counts)

    print("\nDATA QUALITY CHECK")
    print(f"Missing text values: {missing_text}")
    print(f"Missing label values: {missing_label}")
    print(f"Duplicate texts: {duplicate_text}")

    # Save processed full dataset
    processed_path = f"data/wikimia_length{length}_processed.csv"
    df.to_csv(processed_path, index=False, encoding="utf-8")

    print(f"\nSaved processed dataset to: {processed_path}")

    # Save dataset check summary
    summary_path = "outputs/dataset_check_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write(f"WikiMIA_length{length} Dataset Check Summary\n")
        file.write("=" * 45 + "\n\n")

        file.write(f"Total rows: {len(df)}\n")
        file.write(f"Columns: {df.columns.tolist()}\n\n")

        file.write("Label meaning:\n")
        file.write("0 = non-member / unseen\n")
        file.write("1 = member / seen\n\n")

        file.write("Label distribution:\n")
        file.write(label_counts.to_string())
        file.write("\n\n")

        file.write("Data quality check:\n")
        file.write(f"Missing text values: {missing_text}\n")
        file.write(f"Missing label values: {missing_label}\n")
        file.write(f"Duplicate texts: {duplicate_text}\n\n")

        file.write("Sample rows:\n")
        file.write(df.head(10).to_string())

    print(f"Saved dataset summary to: {summary_path}")

    return df


if __name__ == "__main__":
    prepare_full_wikimia(length=64)