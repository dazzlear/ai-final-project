"""Create and validate separate WikiMIA paraphrase CSV files.

This script never writes to or overwrites the processed CSV files.

Input files (read only):
    wikimia_length{N}_processed.csv

Output files:
    wikimia_length{N}_paraphrased.csv

Standard output schema:
    text_id, original_text, label, paraphrase_text

The script can also standardize the legacy length-64 paraphrase file whose
schema is commonly:
    text_id, text, label, original_text, is_paraphrased
where ``text`` contains the generated paraphrase.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from tqdm.auto import tqdm

VALID_LENGTHS = {32, 64, 128, 256}
OUTPUT_COLUMNS = ["text_id", "original_text", "label", "paraphrase_text"]


def normalize_text_id(value: Any) -> str:
    """Normalize text IDs without turning integer IDs into strings like '1.0'."""
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def normalize_ids(series: pd.Series) -> pd.Series:
    return series.map(normalize_text_id)


def nonempty_string_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV through a temporary file so interrupted runs keep the old file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    df.to_csv(temp_path, index=False, encoding="utf-8")
    os.replace(temp_path, path)


def required_columns(df: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{source.name} is missing required columns: {sorted(missing)}")


def load_processed(processed_path: Path) -> pd.DataFrame:
    """Load one processed split without modifying it."""
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed CSV not found: {processed_path}")

    df = pd.read_csv(processed_path)
    required_columns(df, ["text_id", "text", "label"], processed_path)

    result = df[["text_id", "text", "label"]].copy()
    result["text_id"] = normalize_ids(result["text_id"])

    if result["text_id"].eq("").any():
        raise ValueError(f"{processed_path.name} contains empty text_id values.")
    if result["text_id"].duplicated().any():
        duplicates = result.loc[result["text_id"].duplicated(False), "text_id"].tolist()
        raise ValueError(
            f"{processed_path.name} contains duplicate text_id values: {duplicates[:10]}"
        )

    return result


def infer_existing_paraphrase_column(existing: pd.DataFrame, path: Path) -> str:
    """Identify the column holding paraphrased text in current or legacy files."""
    if "paraphrase_text" in existing.columns:
        return "paraphrase_text"
    if "paraphrased_text" in existing.columns:
        return "paraphrased_text"
    if "original_text" in existing.columns and "text" in existing.columns:
        # Legacy project format: original_text=source passage, text=paraphrase.
        return "text"
    raise ValueError(
        f"Cannot identify the paraphrase column in {path.name}. "
        f"Available columns: {existing.columns.tolist()}"
    )


def prepare_paraphrase_dataframe(
    processed: pd.DataFrame,
    paraphrase_path: Path,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, bool]:
    """Create the standardized output frame and reuse any existing paraphrases.

    Returns:
        (standardized_dataframe, existing_file_was_legacy)
    """
    standardized = pd.DataFrame(
        {
            "text_id": processed["text_id"],
            "original_text": processed["text"],
            "label": processed["label"],
            "paraphrase_text": "",
        }
    )

    if overwrite or not paraphrase_path.exists():
        return standardized[OUTPUT_COLUMNS], False

    existing = pd.read_csv(paraphrase_path)
    required_columns(existing, ["text_id"], paraphrase_path)
    existing = existing.copy()
    existing["text_id"] = normalize_ids(existing["text_id"])

    if existing["text_id"].eq("").any():
        raise ValueError(f"{paraphrase_path.name} contains empty text_id values.")
    if existing["text_id"].duplicated().any():
        duplicates = existing.loc[
            existing["text_id"].duplicated(False), "text_id"
        ].tolist()
        raise ValueError(
            f"{paraphrase_path.name} contains duplicate text_id values: {duplicates[:10]}"
        )

    paraphrase_col = infer_existing_paraphrase_column(existing, paraphrase_path)
    legacy = list(existing.columns) != OUTPUT_COLUMNS or paraphrase_col != "paraphrase_text"

    processed_ids = set(processed["text_id"])
    extra_ids = sorted(set(existing["text_id"]) - processed_ids)
    if extra_ids:
        raise ValueError(
            f"{paraphrase_path.name} contains text_id values not found in the processed split: "
            f"{extra_ids[:10]}"
        )

    # Confirm existing labels agree with the processed split when available.
    if "label" in existing.columns:
        aligned = processed[["text_id", "label"]].merge(
            existing[["text_id", "label"]],
            on="text_id",
            how="inner",
            validate="one_to_one",
            suffixes=("_processed", "_existing"),
        )
        if not aligned["label_processed"].astype(str).equals(
            aligned["label_existing"].astype(str)
        ):
            raise ValueError(
                f"Label alignment failed between {paraphrase_path.name} and its processed split."
            )

    # Confirm the legacy/current original text agrees when available.
    existing_original_col = None
    if "original_text" in existing.columns:
        existing_original_col = "original_text"
    elif paraphrase_col != "text" and "text" in existing.columns:
        # Some files may use text=original and paraphrase_text=generated text.
        existing_original_col = "text"

    if existing_original_col is not None:
        aligned = processed[["text_id", "text"]].merge(
            existing[["text_id", existing_original_col]],
            on="text_id",
            how="inner",
            validate="one_to_one",
        )
        left = aligned["text"].fillna("").astype(str)
        right = aligned[existing_original_col].fillna("").astype(str)
        if not left.equals(right):
            mismatch = aligned.loc[left.ne(right), "text_id"].tolist()
            raise ValueError(
                f"Original-text alignment failed in {paraphrase_path.name}; "
                f"mismatched text_id values include {mismatch[:10]}."
            )

    mapping = existing.set_index("text_id")[paraphrase_col]
    standardized["paraphrase_text"] = (
        standardized["text_id"].map(mapping).fillna("").astype(str)
    )

    return standardized[OUTPUT_COLUMNS], legacy


def load_paraphrase_model(model_name: str):
    """Load Hugging Face paraphrase dependencies lazily."""
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Paraphrase generation requires torch, transformers, sentencepiece, and tqdm. "
            "Install them with: pip install torch transformers sentencepiece tqdm"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    print(f"Paraphrase model: {model_name}")
    print(f"Device: {device}")
    return tokenizer, model, device, torch


def generate_batch(
    texts: list[str],
    tokenizer,
    model,
    device: str,
    torch,
    max_input_length: int,
    max_output_length: int,
) -> list[str]:
    """Generate one paraphrase per input. Splits a batch if GPU memory is exhausted."""
    prompts = [f"paraphrase: {text.strip()} </s>" for text in texts]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    try:
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_length=max_output_length,
                num_beams=4,
                num_return_sequences=1,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )
        return [
            tokenizer.decode(sequence, skip_special_tokens=True).strip()
            for sequence in outputs
        ]
    except RuntimeError as exc:
        is_oom = "out of memory" in str(exc).lower()
        if not is_oom or len(texts) == 1:
            raise

        if device == "cuda":
            torch.cuda.empty_cache()
        midpoint = len(texts) // 2
        print(
            f"[WARN] GPU memory limit reached for batch size {len(texts)}; "
            f"retrying as {midpoint} + {len(texts) - midpoint}."
        )
        return generate_batch(
            texts[:midpoint],
            tokenizer,
            model,
            device,
            torch,
            max_input_length,
            max_output_length,
        ) + generate_batch(
            texts[midpoint:],
            tokenizer,
            model,
            device,
            torch,
            max_input_length,
            max_output_length,
        )


def validate_pair(
    processed_path: Path,
    paraphrase_path: Path,
    require_complete: bool,
) -> dict[str, Any]:
    """Validate one paraphrase CSV against its corresponding processed CSV."""
    processed = load_processed(processed_path)

    if not paraphrase_path.exists():
        return {
            "length": int(processed_path.stem.split("length")[1].split("_")[0]),
            "status": "FAIL",
            "reason": "paraphrase file missing",
        }

    paraphrased = pd.read_csv(paraphrase_path)
    missing_columns = set(OUTPUT_COLUMNS) - set(paraphrased.columns)
    exact_columns = paraphrased.columns.tolist() == OUTPUT_COLUMNS

    if "text_id" in paraphrased.columns:
        paraphrased["text_id"] = normalize_ids(paraphrased["text_id"])

    metrics: dict[str, Any] = {
        "length": int(processed_path.stem.split("length")[1].split("_")[0]),
        "processed_rows": len(processed),
        "paraphrase_rows": len(paraphrased),
        "exact_standard_columns": exact_columns,
        "missing_required_columns": ",".join(sorted(missing_columns)),
    }

    if missing_columns:
        metrics.update({"status": "FAIL", "reason": "missing required columns"})
        return metrics

    metrics.update(
        {
            "null_text_id": int(paraphrased["text_id"].isna().sum()),
            "empty_text_id": int(nonempty_string_series(paraphrased["text_id"]).eq("").sum()),
            "duplicate_text_id": int(paraphrased["text_id"].duplicated().sum()),
            "null_original_text": int(paraphrased["original_text"].isna().sum()),
            "empty_original_text": int(
                nonempty_string_series(paraphrased["original_text"]).eq("").sum()
            ),
            "null_label": int(paraphrased["label"].isna().sum()),
            "null_paraphrase_text": int(paraphrased["paraphrase_text"].isna().sum()),
            "empty_paraphrase_text": int(
                nonempty_string_series(paraphrased["paraphrase_text"]).eq("").sum()
            ),
            "row_count_parity": len(processed) == len(paraphrased),
            "text_id_set_match": set(processed["text_id"])
            == set(paraphrased["text_id"]),
            "text_id_order_match": processed["text_id"].tolist()
            == paraphrased["text_id"].tolist(),
        }
    )

    one_to_one = False
    labels_aligned = False
    original_text_aligned = False

    try:
        aligned = processed.merge(
            paraphrased,
            on="text_id",
            how="outer",
            validate="one_to_one",
            indicator=True,
            suffixes=("_processed", "_paraphrased"),
        )
        one_to_one = bool((aligned["_merge"] == "both").all())
        if one_to_one:
            labels_aligned = aligned["label_processed"].astype(str).equals(
                aligned["label_paraphrased"].astype(str)
            )
            original_text_aligned = (
                aligned["text"].fillna("").astype(str).equals(
                    aligned["original_text"].fillna("").astype(str)
                )
            )
    except pd.errors.MergeError:
        one_to_one = False

    metrics["one_to_one_text_id"] = one_to_one
    metrics["label_alignment"] = labels_aligned
    metrics["original_text_alignment"] = original_text_aligned
    metrics["identical_paraphrase_count"] = int(
        paraphrased["original_text"].fillna("").astype(str).eq(
            paraphrased["paraphrase_text"].fillna("").astype(str)
        ).sum()
    )

    core_checks = [
        metrics["exact_standard_columns"],
        metrics["null_text_id"] == 0,
        metrics["empty_text_id"] == 0,
        metrics["duplicate_text_id"] == 0,
        metrics["null_original_text"] == 0,
        metrics["empty_original_text"] == 0,
        metrics["null_label"] == 0,
        metrics["row_count_parity"],
        metrics["text_id_set_match"],
        metrics["one_to_one_text_id"],
        metrics["label_alignment"],
        metrics["original_text_alignment"],
    ]
    if require_complete:
        core_checks.extend(
            [
                metrics["null_paraphrase_text"] == 0,
                metrics["empty_paraphrase_text"] == 0,
            ]
        )

    metrics["status"] = "PASS" if all(core_checks) else "FAIL"
    metrics["reason"] = "" if metrics["status"] == "PASS" else "one or more checks failed"
    return metrics


def write_validation_reports(results: list[dict[str, Any]], output_dir: Path) -> Path:
    summaries = output_dir / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)

    summary_path = summaries / "paraphrase_validation_all.csv"
    pd.DataFrame(results).to_csv(summary_path, index=False)

    for result in results:
        length = result.get("length", "unknown")
        json_path = summaries / f"paraphrase_validation_len{length}.json"
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create separate WikiMIA paraphrase CSVs without changing processed CSVs."
    )
    parser.add_argument("--input_dir", required=True, help="Folder containing processed CSVs.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Folder for paraphrase CSVs. Defaults to --input_dir.",
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=sorted(VALID_LENGTHS),
        help="WikiMIA lengths to process.",
    )
    parser.add_argument(
        "--model_name",
        default="Vamsi/T5_Paraphrase_Paws",
        help="Hugging Face paraphrase model.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=1, help="Checkpoint every N batches.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum missing rows generated per length. 0 means all.",
    )
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_output_length", type=int, default=512)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard existing paraphrases and regenerate them. Use cautiously.",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Do not generate; only validate standardized output files.",
    )
    parser.add_argument(
        "--require_complete",
        action="store_true",
        help="Fail validation if any paraphrase_text is null or empty.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    invalid_lengths = set(args.lengths) - VALID_LENGTHS
    if invalid_lengths:
        raise ValueError(
            f"Invalid lengths: {sorted(invalid_lengths)}. "
            f"Valid lengths: {sorted(VALID_LENGTHS)}"
        )
    if args.batch_size <= 0 or args.save_every <= 0 or args.limit < 0:
        raise ValueError("batch_size and save_every must be positive; limit must be >= 0.")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_bundle = None

    if not args.validate_only:
        for length in args.lengths:
            processed_path = input_dir / f"wikimia_length{length}_processed.csv"
            paraphrase_path = output_dir / f"wikimia_length{length}_paraphrased.csv"

            print(f"\n{'=' * 72}")
            print(f"WikiMIA length {length}")
            print(f"Processed (read only): {processed_path}")
            print(f"Paraphrase output    : {paraphrase_path}")
            print(f"{'=' * 72}")

            processed = load_processed(processed_path)
            output_df, was_legacy = prepare_paraphrase_dataframe(
                processed, paraphrase_path, overwrite=args.overwrite
            )

            if was_legacy:
                backup_dir = output_dir / "backups_paraphrase"
                backup_dir.mkdir(parents=True, exist_ok=True)
                backup_path = backup_dir / f"wikimia_length{length}_paraphrased_legacy.csv"
                if paraphrase_path.exists() and not backup_path.exists():
                    shutil.copy2(paraphrase_path, backup_path)
                    print(f"Legacy backup saved : {backup_path}")

            # Standardize immediately, including the legacy length-64 file.
            atomic_write_csv(output_df, paraphrase_path)

            pending_mask = nonempty_string_series(output_df["paraphrase_text"]).eq("")
            pending_indices = output_df.index[pending_mask].tolist()
            if args.limit > 0:
                pending_indices = pending_indices[: args.limit]

            completed_count = int((~pending_mask).sum())
            print(f"Rows in processed split : {len(output_df)}")
            print(f"Existing paraphrases    : {completed_count}")
            print(f"Rows selected to create : {len(pending_indices)}")

            if not pending_indices:
                print("No missing paraphrases selected; generation skipped.")
                continue

            if model_bundle is None:
                model_bundle = load_paraphrase_model(args.model_name)
            tokenizer, model, device, torch = model_bundle

            progress = tqdm(total=len(pending_indices), desc=f"length {length}")
            try:
                for batch_number, start in enumerate(
                    range(0, len(pending_indices), args.batch_size), start=1
                ):
                    batch_indices = pending_indices[start : start + args.batch_size]
                    batch_texts = [
                        str(output_df.at[index, "original_text"]).strip()
                        for index in batch_indices
                    ]
                    generated = generate_batch(
                        batch_texts,
                        tokenizer,
                        model,
                        device,
                        torch,
                        args.max_input_length,
                        args.max_output_length,
                    )
                    if len(generated) != len(batch_indices):
                        raise RuntimeError(
                            "Generated output count does not match the input batch count."
                        )
                    for index, paraphrase in zip(batch_indices, generated):
                        output_df.at[index, "paraphrase_text"] = paraphrase

                    progress.update(len(batch_indices))
                    if batch_number % args.save_every == 0:
                        atomic_write_csv(output_df[OUTPUT_COLUMNS], paraphrase_path)
            finally:
                atomic_write_csv(output_df[OUTPUT_COLUMNS], paraphrase_path)
                progress.close()

            remaining = int(
                nonempty_string_series(output_df["paraphrase_text"]).eq("").sum()
            )
            print(f"Saved: {paraphrase_path}")
            print(f"Remaining empty paraphrases: {remaining}")

    print("\nRunning validation...")
    results = []
    for length in args.lengths:
        processed_path = input_dir / f"wikimia_length{length}_processed.csv"
        paraphrase_path = output_dir / f"wikimia_length{length}_paraphrased.csv"
        result = validate_pair(
            processed_path,
            paraphrase_path,
            require_complete=args.require_complete,
        )
        results.append(result)
        print(
            f"Length {length}: {result.get('status')} | "
            f"rows={result.get('paraphrase_rows', 'missing')}/"
            f"{result.get('processed_rows', 'unknown')} | "
            f"empty paraphrases={result.get('empty_paraphrase_text', 'n/a')}"
        )

    summary_path = write_validation_reports(results, output_dir)
    print(f"Validation summary: {summary_path}")

    failed = [result for result in results if result.get("status") != "PASS"]
    if failed:
        print("\nFAILED validation lengths:", [item.get("length") for item in failed])
        sys.exit(1)

    print("\nPASS: all requested paraphrase CSVs satisfy the validation rules.")


if __name__ == "__main__":
    main()
