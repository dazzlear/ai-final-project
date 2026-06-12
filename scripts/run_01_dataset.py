"""Dataset loading, validation, and preparation for all configured lengths.

Entry point for Module 01.

Basic usage:
    python scripts/run_01_dataset.py --lengths 32 64 128 256

With paraphrased dataset generation:
    python scripts/run_01_dataset.py --lengths 64 --make_paraphrase --paraphrase_length 64 --paraphrase_limit 100 (None if all rows)
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Ensure src/ is importable from repo root or Colab
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from data import load_wikimia, VALID_LENGTHS  # noqa: E402


def _collect_stats(df: pd.DataFrame, length: int) -> dict:
    """Collect quality statistics for one WikiMIA split."""
    text_series = df["text"].fillna("").astype(str).str.strip()

    return {
        "length": length,
        "total_rows": len(df),
        "columns": df.columns.tolist(),
        "missing_text": int(df["text"].isna().sum()),
        "missing_label": int(df["label"].isna().sum()),
        "duplicate_texts": int(df["text"].duplicated().sum()),
        "empty_texts": int((text_series == "").sum()),
        "label_0": int((df["label"] == 0).sum()),
        "label_1": int((df["label"] == 1).sum()),
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


def _ensure_text_id(df: pd.DataFrame, length: int) -> pd.DataFrame:
    """Ensure every processed dataset has a stable text_id column."""
    df = df.copy()

    if "text_id" not in df.columns:
        df.insert(0, "text_id", [f"len{length}_{i}" for i in range(len(df))])

    return df


def process_length(length: int, output_dir: Path, sample_size: int) -> dict:
    """Load, collect stats, and save one WikiMIA length split."""
    print(f"\n{'=' * 50}")
    print(f"  Processing WikiMIA_length{length}")
    print(f"{'=' * 50}")

    df = load_wikimia(length=length)

    required_columns = {"text", "label"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"WikiMIA_length{length} is missing required columns: {missing_columns}"
        )

    df = _ensure_text_id(df, length)
    stats = _collect_stats(df, length)

    if stats["missing_text"] > 0 or stats["missing_label"] > 0:
        print("  [WARN] Missing values detected — check summary for details.")

    if stats["empty_texts"] > 0:
        print(f"  [WARN] {stats['empty_texts']} empty text rows detected.")

    if stats["duplicate_texts"] > 0:
        print(f"  [WARN] {stats['duplicate_texts']} duplicate texts detected.")

    output_dir.mkdir(parents=True, exist_ok=True)

    processed_path = output_dir / f"wikimia_length{length}_processed.csv"
    df.to_csv(processed_path, index=False, encoding="utf-8")
    print(f"  Saved processed CSV  : {processed_path}")

    half = sample_size // 2
    actual_members = min(half, stats["label_1"])
    actual_non_members = min(half, stats["label_0"])

    if actual_members < half or actual_non_members < half:
        print(
            f"  [WARN] Requested {half} samples per class but only "
            f"{stats['label_1']} members / {stats['label_0']} non-members available. "
            f"Sample will be smaller than requested."
        )

    members = df[df["label"] == 1].sample(actual_members, random_state=42)
    non_members = df[df["label"] == 0].sample(actual_non_members, random_state=42)

    sample_df = (
        pd.concat([members, non_members], ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    sample_path = output_dir / f"wikimia_length{length}_sample.csv"
    sample_df.to_csv(sample_path, index=False, encoding="utf-8")
    print(f"  Saved sample CSV     : {sample_path}  ({len(sample_df)} rows)")

    summary_path = output_dir / "summaries" / f"dataset_summary_len{length}.txt"
    _write_summary(stats, summary_path)
    print(f"  Saved quality report : {summary_path}")

    print(
        f"\n  Rows: {stats['total_rows']}  |  "
        f"label_0: {stats['label_0']}  |  label_1: {stats['label_1']}"
    )

    return stats


def _load_paraphrase_dependencies():
    """Load paraphrase dependencies only when paraphrasing is requested."""
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")

        AutoTokenizer = getattr(transformers, "AutoTokenizer")
        AutoModelForSeq2SeqLM = getattr(transformers, "AutoModelForSeq2SeqLM")

        return torch, AutoTokenizer, AutoModelForSeq2SeqLM

    except ImportError as exc:
        raise ImportError(
            "Paraphrase generation requires extra packages. Install them with:\n"
            "pip install torch transformers sentencepiece"
        ) from exc


def load_paraphraser(model_name: str):
    """Load the paraphrase model used for Module 1B."""
    torch, AutoTokenizer, AutoModelForSeq2SeqLM = _load_paraphrase_dependencies()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    print(f"Paraphrase model loaded: {model_name}")
    print(f"Device: {device}")

    return tokenizer, model, device, torch


def paraphrase_text(
    text: str,
    tokenizer,
    model,
    device: str,
    torch,
    max_input_length: int = 256,
    max_output_length: int = 256,
) -> str:
    """Generate one paraphrased version of the input text."""
    text = str(text).strip()

    if text == "":
        return text

    prompt = f"paraphrase: {text} </s>"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_output_length,
            num_beams=4,
            num_return_sequences=1,
            early_stopping=True,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def make_paraphrased_dataset(
    input_csv: str | Path,
    output_csv: str | Path,
    model_name: str,
    limit: int | None = None,
    max_input_length: int = 256,
    max_output_length: int = 256,
) -> None:
    """Create a paraphrased version of a processed WikiMIA dataset.

    Expected input columns:
    - text_id
    - text
    - label

    Output columns:
    - original columns
    - original_text
    - is_paraphrased
    """

    input_path = Path(input_csv)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    required_columns = {"text_id", "text", "label"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns for paraphrasing: {missing_columns}")

    if limit is not None:
        if limit <= 0:
            raise ValueError("--paraphrase_limit must be positive. Use 0 only from CLI to mean all rows.")
        df = df.head(int(limit)).copy()

    if df["label"].nunique() < 2:
        print(
            "  [WARN] Paraphrased subset contains only one label. "
            "AUC/TPR evaluation later requires both label 0 and label 1."
        )

    tokenizer, model, device, torch = load_paraphraser(model_name)

    paraphrased_texts = []

    for text in tqdm(df["text"].tolist(), desc="Generating paraphrases"):
        paraphrased_text = paraphrase_text(
            text=text,
            tokenizer=tokenizer,
            model=model,
            device=device,
            torch=torch,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
        )
        paraphrased_texts.append(paraphrased_text)

    df["original_text"] = df["text"]
    df["text"] = paraphrased_texts
    df["is_paraphrased"] = True

    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Saved paraphrased dataset to: {output_path}")
    print(f"Rows saved: {len(df)}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    default_drive = os.environ.get("DRIVE", "")
    default_output = (
        str(Path(default_drive) / "01_dataset")
        if default_drive
        else "outputs/01_dataset"
    )

    parser = argparse.ArgumentParser(
        description="WikiMIA dataset loading and preparation for Min-K% replication."
    )

    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=sorted(VALID_LENGTHS),
        help=f"Length buckets to process. Valid: {sorted(VALID_LENGTHS)}. Default: all.",
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
        help="Total rows in smoke-test sample. Must be positive and even.",
    )

    parser.add_argument(
        "--make_paraphrase",
        action="store_true",
        help="Generate a paraphrased dataset after Module 1 dataset preparation.",
    )

    parser.add_argument(
        "--paraphrase_length",
        type=int,
        default=64,
        help="WikiMIA length split to paraphrase. Default: 64.",
    )

    parser.add_argument(
        "--paraphrase_limit",
        type=int,
        default=100,
        help="Number of rows to paraphrase. Use 100 for testing. Use 0 for all rows.",
    )

    parser.add_argument(
        "--paraphrase_model",
        type=str,
        default="Vamsi/T5_Paraphrase_Paws",
        help="Hugging Face paraphrase model.",
    )

    parser.add_argument(
        "--paraphrase_max_input_length",
        type=int,
        default=256,
        help="Maximum input token length for the paraphrase model.",
    )

    parser.add_argument(
        "--paraphrase_max_output_length",
        type=int,
        default=256,
        help="Maximum output token length for the paraphrase model.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate user-provided command-line arguments."""
    valid_lengths = set(VALID_LENGTHS)

    if args.sample_size <= 0:
        print(f"[ERROR] --sample_size must be positive. Got {args.sample_size}.")
        sys.exit(1)

    if args.sample_size % 2 != 0:
        print(
            f"[ERROR] --sample_size must be even for balanced sampling. "
            f"Got {args.sample_size}. Try {args.sample_size + 1}."
        )
        sys.exit(1)

    invalid_lengths = set(args.lengths) - valid_lengths

    if invalid_lengths:
        print(
            f"[ERROR] Invalid lengths: {sorted(invalid_lengths)}. "
            f"Must be from {sorted(valid_lengths)}."
        )
        sys.exit(1)

    if args.make_paraphrase:
        if args.paraphrase_length not in valid_lengths:
            print(
                f"[ERROR] Invalid --paraphrase_length: {args.paraphrase_length}. "
                f"Must be from {sorted(valid_lengths)}."
            )
            sys.exit(1)

        if args.paraphrase_limit < 0:
            print(
                f"[ERROR] --paraphrase_limit must be 0 or positive. "
                f"Got {args.paraphrase_limit}."
            )
            sys.exit(1)

        if args.paraphrase_max_input_length <= 0:
            print("[ERROR] --paraphrase_max_input_length must be positive.")
            sys.exit(1)

        if args.paraphrase_max_output_length <= 0:
            print("[ERROR] --paraphrase_max_output_length must be positive.")
            sys.exit(1)


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_dir = Path(args.output_dir)
    all_stats = []

    print(f"\nOutput directory : {output_dir}")
    print(f"Lengths          : {sorted(args.lengths)}")
    print(f"Sample size      : {args.sample_size}")

    for length in sorted(args.lengths):
        stats = process_length(length, output_dir, args.sample_size)
        all_stats.append(stats)

    summary_df = pd.DataFrame(all_stats)
    master_path = output_dir / "summaries" / "dataset_summary_all.csv"
    master_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(master_path, index=False)

    print(f"\n{'=' * 50}")
    print("  All requested lengths processed.")
    print(f"  Master summary saved : {master_path}")
    print(f"{'=' * 50}\n")
    print(summary_df.to_string(index=False))

    if args.make_paraphrase:
        input_csv = output_dir / f"wikimia_length{args.paraphrase_length}_processed.csv"
        output_csv = output_dir / f"wikimia_length{args.paraphrase_length}_paraphrased.csv"

        paraphrase_limit = None if args.paraphrase_limit == 0 else args.paraphrase_limit

        print(f"\n{'=' * 50}")
        print("  Module 1B — Paraphrased Dataset Generation")
        print(f"{'=' * 50}")
        print(f"Input CSV         : {input_csv}")
        print(f"Output CSV        : {output_csv}")
        print(f"Paraphrase model  : {args.paraphrase_model}")
        print(f"Paraphrase limit  : {'all rows' if paraphrase_limit is None else paraphrase_limit}")

        make_paraphrased_dataset(
            input_csv=input_csv,
            output_csv=output_csv,
            model_name=args.paraphrase_model,
            limit=paraphrase_limit,
            max_input_length=args.paraphrase_max_input_length,
            max_output_length=args.paraphrase_max_output_length,
        )

        print(f"\n{'=' * 50}")
        print("  Module 1B completed.")
        print(f"  Paraphrased CSV saved : {output_csv}")
        print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()