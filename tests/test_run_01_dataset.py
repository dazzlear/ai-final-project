"""
Tests for scripts/run_01_dataset.py
Run from repo root: python -m pytest tests/test_run_01_dataset.py -v
"""
import os
import sys
import pytest
import pandas as pd
sys.path.insert(0, os.path.abspath("."))


class TestDataModule:
    """Tests for src/data.py functions used by run_01_dataset.py."""

    def test_load_wikimia_import(self):
        """src/data.py should import without errors."""
        try:
            from src.data import load_wikimia
        except ImportError as e:
            pytest.fail(f"Could not import load_wikimia from src.data: {e}")

    def test_processed_csv_exists(self):
        """
        The processed dataset CSV must exist.
        This is Dazel's deliverable and everything else depends on it.
        """
        path = "data/wikimia_length64_processed.csv"
        assert os.path.isfile(path), (
            f"{path} not found. "
            "Run scripts/run_01_dataset.py to generate it."
        )

    def test_processed_csv_has_required_columns(self):
        """Processed CSV must have 'text' and 'label' columns."""
        df = pd.read_csv("data/wikimia_length64_processed.csv")
        assert "text"  in df.columns, "Missing 'text' column"
        assert "label" in df.columns, "Missing 'label' column"

    def test_processed_csv_label_values(self):
        """Labels must be binary: 0 (non-member) or 1 (member) only."""
        df = pd.read_csv("data/wikimia_length64_processed.csv")
        unique = set(df["label"].unique())
        assert unique <= {0, 1}, (
            f"Unexpected label values: {unique}. Expected only 0 and 1."
        )

    def test_processed_csv_balanced(self):
        """
        Dataset should be roughly balanced (±10% tolerance).
        WikiMIA is constructed to be 50/50 member vs non-member.
        """
        df = pd.read_csv("data/wikimia_length64_processed.csv")
        counts = df["label"].value_counts()
        n_member     = counts.get(1, 0)
        n_nonmember  = counts.get(0, 0)
        total        = len(df)
        ratio        = abs(n_member - n_nonmember) / total
        assert ratio <= 0.10, (
            f"Dataset imbalanced: {n_member} members, {n_nonmember} non-members. "
            f"Imbalance ratio: {ratio:.2%} (tolerance: 10%)"
        )

    def test_processed_csv_no_empty_texts(self):
        """No row should have an empty or whitespace-only text."""
        df = pd.read_csv("data/wikimia_length64_processed.csv")
        empty = df["text"].isna() | (df["text"].str.strip() == "")
        n_empty = empty.sum()
        assert n_empty == 0, (
            f"{n_empty} rows have empty text. "
            "Run data validation in run_01_dataset.py."
        )

    def test_sample_csv_exists(self):
        """Sample CSV should also exist (used for smoke tests)."""
        assert os.path.isfile("data/wikimia_length64_sample.csv"), (
            "data/wikimia_length64_sample.csv not found."
        )

    def test_run_01_dataset_script_exists(self):
        """scripts/run_01_dataset.py must exist."""
        assert os.path.isfile("scripts/run_01_dataset.py"), (
            "scripts/run_01_dataset.py not found. "
            "Move full_dataset.py → scripts/run_01_dataset.py."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])