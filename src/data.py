"""Dataset loading and preparation for Min-K% replication."""

from pathlib import Path

import pandas as pd
from datasets import load_dataset

VALID_LENGTHS = {32, 64, 128, 256}


def load_wikimia(length: int = 64) -> pd.DataFrame:
    """Load WikiMIA dataset from Hugging Face.
    
    Labels: 0=non-member, 1=member. Returns columns: text_id, text, label.
    """
    if length not in VALID_LENGTHS:
        raise ValueError(
            f"Unsupported length {length!r}. "
            f"Must be one of {sorted(VALID_LENGTHS)}."
        )

    # Split names are 'WikiMIA_length32', 'WikiMIA_length64', etc.
    split_name = f"WikiMIA_length{length}"
    dataset = load_dataset("swj0419/WikiMIA", split=split_name)
    df = dataset.to_pandas()

    # Track each row with text_id
    if "text_id" not in df.columns:
        df.insert(0, "text_id", range(len(df)))

    # Normalize text column (HuggingFace may use "input", "sentence", etc.)
    possible_text_columns = ["text", "input", "sentence", "content"]
    text_column = next((col for col in possible_text_columns if col in df.columns), None)

    if text_column is None:
        raise ValueError(
            f"No text column found. Available columns: {df.columns.tolist()}"
        )

    if text_column != "text":
        df = df.rename(columns={text_column: "text"})

    if "label" not in df.columns:
        raise ValueError(
            f"No label column found. Available columns: {df.columns.tolist()}"
        )

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
    """Print dataset statistics and label distribution."""
    print("\nDATASET PREVIEW")
    print(df.head())
    print("\nDATASET SHAPE")
    print(df.shape)
    print("\nCOLUMNS")
    print(df.columns.tolist())
    print("\nLABEL DISTRIBUTION")
    print(df["label"].value_counts())
    print("\nLABEL MEANING: 0=non-member, 1=member")


def save_wikimia_sample(length: int = 64, sample_size: int = 10) -> pd.DataFrame:
    """Save balanced sample for smoke testing."""
    Path("data").mkdir(exist_ok=True)
    df = load_wikimia(length=length)

    half = sample_size // 2

    # Use sample() not head() to avoid row-order selection bias
    members = df[df["label"] == 1].sample(half, random_state=42)
    non_members = df[df["label"] == 0].sample(half, random_state=42)

    # Shuffle so members and non-members are interleaved
    sample_df = (
        pd.concat([members, non_members], ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    output_path = f"data/wikimia_length{length}_sample.csv"
    sample_df.to_csv(output_path, index=False)

    print(f"\nSaved sample file: {output_path}")
    print(sample_df.head())

    return sample_df


def save_wikimia_full(length: int = 64) -> pd.DataFrame:
    """Save full WikiMIA split as processed CSV."""
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