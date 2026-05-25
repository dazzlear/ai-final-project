"""
Tests for src/methods.py
Run from repo root: python -m pytest tests/test_methods.py -v
"""
import os
import pickle
import tempfile
import pytest
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, os.path.abspath("."))

from src.methods import (
    min_k_prob,
    save_logprobs_cache,
    load_logprobs_cache,
)


# ── min_k_prob ─────────────────────────────────────────────────────────────

class TestMinKProb:

    def test_empty_returns_nan(self):
        """Empty logprob list should return nan, not crash."""
        result = min_k_prob([])
        assert np.isnan(result), f"Expected nan, got {result}"

    def test_sign_convention_member(self):
        """
        Member texts have higher (less negative) min-k% scores.
        Simulate: member = mostly high-prob tokens, one low-prob outlier.
        Non-member = several very low-prob tokens.
        """
        member_logprobs     = [-0.1, -0.2, -0.1, -0.15, -5.0]  # one outlier
        nonmember_logprobs  = [-4.0, -5.0, -4.5, -6.0, -4.8]   # all surprising

        member_score    = min_k_prob(member_logprobs,    k=20)
        nonmember_score = min_k_prob(nonmember_logprobs, k=20)

        assert member_score > nonmember_score, (
            f"Sign convention broken: member ({member_score:.4f}) should be "
            f"> non-member ({nonmember_score:.4f})"
        )

    def test_selects_bottom_k_percent(self):
        """k=20 on 10 tokens should select bottom 2 tokens."""
        logprobs = [-0.1, -0.2, -0.3, -0.4, -0.5,
                    -0.6, -0.7, -0.8, -0.9, -1.0]
        result = min_k_prob(logprobs, k=20)
        expected = np.mean([-1.0, -0.9])  # bottom 2 of 10
        assert abs(result - expected) < 1e-6, (
            f"Expected {expected:.6f}, got {result:.6f}"
        )

    def test_k_at_least_one_token(self):
        """Even with k=1 and a short list, at least 1 token is selected."""
        result = min_k_prob([-0.5], k=1)
        assert not np.isnan(result), "Should not return nan for single token"
        assert abs(result - (-0.5)) < 1e-6

    def test_full_list_k100(self):
        """k=100 should use all tokens — result equals mean of all logprobs."""
        logprobs = [-0.3, -0.5, -0.7, -1.2]
        result   = min_k_prob(logprobs, k=100)
        expected = np.mean(logprobs)
        assert abs(result - expected) < 1e-6, (
            f"Expected {expected:.6f}, got {result:.6f}"
        )

    def test_not_negated(self):
        """
        Confirm the old negation bug is gone.
        min_k_prob should return a negative number for typical log-probs
        (log-probs are always <= 0).
        """
        logprobs = [-0.5, -1.0, -2.0, -0.3]
        result = min_k_prob(logprobs, k=20)
        assert result < 0, (
            f"Expected negative value (log-prob), got {result}. "
            "Negation bug may have returned."
        )


# ── cache functions ────────────────────────────────────────────────────────

class TestCacheFunctions:

    def test_save_and_load_roundtrip(self):
        """Saved cache should load back identically."""
        data = [[-0.1, -0.5, -0.3], [-0.9, -1.2], [-0.4]]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_cache.pkl")
            save_logprobs_cache(data, path)
            loaded = load_logprobs_cache(path)
        assert loaded == data, "Cache roundtrip failed"

    def test_load_missing_returns_none(self):
        """Loading a nonexistent cache should return None, not raise."""
        result = load_logprobs_cache("/nonexistent/path/cache.pkl")
        assert result is None, f"Expected None, got {result}"

    def test_save_creates_directory(self):
        """save_logprobs_cache should create missing parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested", "dir", "cache.pkl")
            save_logprobs_cache([[-0.1]], path)
            assert os.path.exists(path), "File was not created"


# ── cache mismatch ─────────────────────────────────────────────────────────

class TestCacheMismatch:

    def test_mismatch_triggers_recompute(self, capsys):
        """
        score_dataset should detect cache length mismatch and discard cache.
        We mock get_token_logprobs to avoid loading a real model.
        """
        from unittest.mock import patch, MagicMock

        # 3-row dataset
        df = pd.DataFrame({
            "text":  ["text one", "text two", "text three"],
            "label": [1, 0, 1],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.pkl")

            # Save a cache with 5 entries (mismatches 3-row dataset)
            save_logprobs_cache(
                [[-0.1], [-0.2], [-0.3], [-0.4], [-0.5]],
                cache_path
            )

            mock_model     = MagicMock()
            mock_tokenizer = MagicMock()

            with patch("src.methods.get_token_logprobs",
                       return_value=[-0.5, -0.3]) as mock_lp:
                from src.methods import score_dataset
                result = score_dataset(
                    df, mock_model, mock_tokenizer,
                    k=20, cache_path=cache_path
                )

            captured = capsys.readouterr()
            assert "Cache" in captured.out or "mismatch" in captured.out.lower(), (
                "Expected cache mismatch warning in output"
            )
            assert len(result) == 3, "Result should have 3 rows matching the dataset"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])