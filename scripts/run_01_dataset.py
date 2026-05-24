"""Dataset loading, validation, and preparation for all configured lengths.

Entry point for Module 01. Calls src/data.py and writes to outputs/.
Usage: python scripts/run_01_dataset.py --lengths 32 64 128 256
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Ensure src/ is importable from repo root or Colab
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from data import load_wikimia, _VALID_LENGTHS  # noqa: E402


def _validate(df: pd.DataFrame, length: int) -> dict:
    """Run quality checks and return summary dict."""
    return {
        "length":          length,
        "total_rows":      len(df),
        "columns":         df.columns.tolist(),
        "missing_text":    int(df["text"].isna().sum()),
        "missing_label":   int(df["label"].isna().sum()),
        "duplicate_texts": int(df["text"].duplicated().sum()),
        "empty_texts":     int((df["text"].str.strip() == "").sum()),
        "label_0":         int((df["label"] == 0).sum()),
        "label_1":         int((df["label"] == 1).sum()),
    }


def _write_summary(stats: dict, summary_path: Path) -> None:
    """Write human-readable quality report to disk."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"WikiMIA_length{stats['length']} Dataset Summary",
        "=" * 45,
        "",
        f"Total rows      : {stats['total_rows']}",
        f"Columns         : {stats['columns']}",
        "",
        "Label meaning:",
        "  0 = non-member / unseen",
        "  1 = member / seen",
        "",
        "Label distribution:",
        f"  label 0 (non-member) : {stats['label_0']}",
        f"  label 1 (member)     : {stats['label_1']}",
        "",
        "Data quality:",
        f"  Missing text values  : {stats['missing_text']}",
        f"  Missing label values : {stats['missing_label']}",
        f"  Duplicate texts      : {stats['duplicate_texts']}",
        f"  Empty texts          : {stats['empty_texts']}",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def process_length(length: int, output_dir: Path, sample_size: int) -> dict:
    """Load, validate, and save one WikiMIA length split."""
    print(f"\n{'='*50}")
    print(f"  Processing WikiMIA_length{length}")
    print(f"{'='*50}")

    df = load_wikimia(length=length)
    stats = _validate(df, length)

    # Warn on data quality issues
    if stats["missing_text"] > 0 or stats["missing_label"] > 0:
        print(f"  [WARN] Missing values detected — check summary for details.")
    if stats["empty_texts"] > 0:
        print(f"  [WARN] {stats['empty_texts']} empty text rows detected.")
    if stats["duplicate_texts"] > 0:
        print(f"  [WARN] {stats['duplicate_texts']} duplicate texts detected.")

    # Save processed CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_path = output_dir / f"wikimia_length{length}_processed.csv"
    df.to_csv(processed_path, index=False, encoding="utf-8")
    print(f"  Saved processed CSV  : {processed_path}")

    # Save balanced, shuffled sample
    half = sample_size // 2
    members     = df[df["label"] == 1].sample(min(half, stats["label_1"]), random_state=42)
    non_members = df[df["label"] == 0].sample(min(half, stats["label_0"]), random_state=42)
    sample_df   = (
        pd.concat([members, non_members], ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    sample_path = output_dir / f"wikimia_length{length}_sample.csv"
    sample_df.to_csv(sample_path, index=False, encoding="utf-8")
    print(f"  Saved sample CSV     : {sample_path}")

    # Write quality summary
    summary_path = output_dir / "summaries" / f"dataset_summary_len{length}.txt"
    _write_summary(stats, summary_path)
    print(f"  Saved quality report : {summary_path}")

    print(f"\n  Rows: {stats['total_rows']}  |  "
          f"label_0: {stats['label_0']}  |  label_1: {stats['label_1']}")

    return stats


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    # DRIVE env var allows Colab to override output dir without editing code
    default_drive = os.environ.get("DRIVE", "")
    default_output = (
        str(Path(default_drive) / "01_dataset") if default_drive
        else "outputs/01_dataset"
    )

    parser = argparse.ArgumentParser(
        description="WikiMIA dataset loading and preparation for Min-K% replication."
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=sorted(_VALID_LENGTHS),
        help=f"Length buckets to process. Valid: {sorted(_VALID_LENGTHS)}. Default: all.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=default_output,
        help="Directory for processed CSVs and summaries.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=10,
        help="Total rows in smoke-test sample (split evenly by label).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate lengths before downloading
    invalid = set(args.lengths) - _VALID_LENGTHS
    if invalid:
        print(f"[ERROR] Invalid lengths: {invalid}. Must be from {sorted(_VALID_LENGTHS)}.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    all_stats  = []

    print(f"\nOutput directory : {output_dir}")
    print(f"Lengths          : {args.lengths}")
    print(f"Sample size      : {args.sample_size}")

    for length in sorted(args.lengths):
        stats = process_length(length, output_dir, args.sample_size)
        all_stats.append(stats)

    # Save master summary across all lengths
    summary_df = pd.DataFrame(all_stats)
    master_path = output_dir / "summaries" / "dataset_summary_all.csv"
    summary_df.to_csv(master_path, index=False)
    print(f"\n{'='*50}")
    print("  All lengths processed.")
    print(f"  Master summary saved : {master_path}")
    print(f"{'='*50}\n")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()