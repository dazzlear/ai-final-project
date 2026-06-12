from pathlib import Path

import pandas as pd
from datasets import load_dataset


def load_wikimia(length: int = 64) -> pd.DataFrame:
   
    """
    Load the WikiMIA dataset from Hugging Face.

    Label meaning:
    0 = non-member / unseen
    1 = member / seen
    """

    split_name = f"WikiMIA_length{length}"

    dataset = load_dataset(
        "swj0419/WikiMIA",
        split=split_name
    )

    df = dataset.to_pandas()

    # Add text_id so each row can be tracked during scoring
    if "text_id" not in df.columns:
        df.insert(0, "text_id", range(len(df)))

    # Make sure the text column is named "text"
    possible_text_columns = ["text", "input", "sentence", "content"]

    text_column = None
    for col in possible_text_columns:
        if col in df.columns:
            text_column = col
            break

    if text_column is None:
        raise ValueError(f"No text column found. Available columns: {df.columns.tolist()}")

    if text_column != "text":
        df = df.rename(columns={text_column: "text"})

    if "label" not in df.columns:
        raise ValueError(f"No label column found. Available columns: {df.columns.tolist()}")

    return df[["text_id", "text", "label"]]


def load_wikimia_all(lengths=(32, 64, 128, 256)) -> pd.DataFrame:
    """Load and concatenate all WikiMIA length splits into one DataFrame.
    
    Args:
        lengths: Tuple of length values to load (default: all available).
    
    Returns:
        Concatenated DataFrame with 'length' column tracking split origin.
    """
    frames = []
    offset = 0
    for length in lengths:
        df = load_wikimia(length=length)
        df.insert(1, "length", length)           # track which split each row came from
        df["text_id"] = range(offset, offset + len(df))   # avoid duplicate text_ids
        offset += len(df)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def inspect_wikimia(df: pd.DataFrame) -> None:
    """
    Print basic dataset information.
    """

    print("\nDATASET PREVIEW")
    print(df.head())

    print("\nDATASET SHAPE")
    print(df.shape)

    print("\nCOLUMNS")
    print(df.columns.tolist())

    print("\nLABEL DISTRIBUTION")
    print(df["label"].value_counts())

    print("\nLABEL MEANING")
    print("0 = non-member / unseen")
    print("1 = member / seen")


def save_wikimia_sample(length: int = 64, sample_size: int = 10) -> pd.DataFrame:
    """
    Save a small balanced sample for smoke testing.
    """

    Path("data").mkdir(exist_ok=True)

    df = load_wikimia(length=length)

    half = sample_size // 2

    members = df[df["label"] == 1].head(half)
    non_members = df[df["label"] == 0].head(half)

    sample_df = pd.concat([members, non_members], ignore_index=True)

    output_path = f"data/wikimia_length{length}_sample.csv"
    sample_df.to_csv(output_path, index=False)

    print(f"\nSaved sample file: {output_path}")
    print(sample_df.head())

    return sample_df


def save_wikimia_full(length: int = 64) -> pd.DataFrame:
    """
    Save the full WikiMIA length split as a processed CSV.
    """

    Path("data").mkdir(exist_ok=True)

    df = load_wikimia(length=length)

    output_path = f"data/wikimia_length{length}_processed.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved full dataset file: {output_path}")
    print(f"Total rows: {len(df)}")

    return df


if __name__ == "__main__":
    df = load_wikimia(length=64)
    inspect_wikimia(df)
    save_wikimia_sample(length=64, sample_size=10)