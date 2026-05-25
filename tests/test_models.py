"""
Tests for src/models.py
Run from repo root: python -m pytest tests/test_models.py -v

NOTE: Full model load tests are skipped by default (too slow for CI).
      Run with --run-slow to include them:
      python -m pytest tests/test_models.py -v --run-slow
"""
import os
import sys
import pytest
import torch
sys.path.insert(0, os.path.abspath("."))


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False,
        help="Run tests that load real models (slow)"
    )


# ── get_token_logprobs logic (no real model needed) ───────────────────────

class TestGetTokenLogprobsLogic:
    """
    Test the mathematical logic of get_token_logprobs using a
    tiny mock model so we don't need to download anything.
    """

    def _make_mock_model_and_tokenizer(self, vocab_size=50, seq_len=5):
        """Build a minimal mock that mimics HuggingFace CausalLM output."""
        from unittest.mock import MagicMock, patch
        import torch

        # Mock tokenizer: encode returns fixed token ids
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(seq_len))

        # Mock model: returns logits of shape (1, seq_len, vocab_size)
        # Use uniform logits so log_softmax gives predictable values
        mock_logits = torch.zeros(1, seq_len, vocab_size)
        mock_output = MagicMock()
        mock_output.logits = mock_logits

        mock_model = MagicMock()
        mock_model.return_value = mock_output
        mock_model.device = torch.device("cpu")

        # Make next(model.parameters()).device work
        param = torch.nn.Parameter(torch.tensor([0.0]))
        mock_model.parameters = MagicMock(return_value=iter([param]))

        return mock_model, mock_tokenizer

    def test_output_length(self):
        """Output should have seq_len - 1 values (first token excluded)."""
        from src.models import get_token_logprobs
        mock_model, mock_tokenizer = self._make_mock_model_and_tokenizer(
            vocab_size=50, seq_len=6
        )
        result = get_token_logprobs("test text", mock_model, mock_tokenizer)
        assert len(result) == 5, (
            f"Expected 5 log-probs (seq_len-1), got {len(result)}"
        )

    def test_output_all_negative_or_zero(self):
        """Log-probabilities must be <= 0 (they are log of values in [0,1])."""
        from src.models import get_token_logprobs
        mock_model, mock_tokenizer = self._make_mock_model_and_tokenizer(
            vocab_size=50, seq_len=5
        )
        result = get_token_logprobs("test text", mock_model, mock_tokenizer)
        for i, lp in enumerate(result):
            assert lp <= 0.0, (
                f"Token {i} has log-prob {lp} > 0, which is mathematically impossible"
            )

    def test_output_is_list_of_floats(self):
        """Return type should be a plain Python list of floats."""
        from src.models import get_token_logprobs
        mock_model, mock_tokenizer = self._make_mock_model_and_tokenizer()
        result = get_token_logprobs("test text", mock_model, mock_tokenizer)
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        for v in result:
            assert isinstance(v, float), f"Expected float, got {type(v)}"

    def test_uses_outputs_logits_not_index(self):
        """
        Confirm we access outputs.logits (named attr) not outputs[1].
        If outputs[1] were used, this test would fail because our mock
        does not support integer indexing.
        """
        from src.models import get_token_logprobs
        from unittest.mock import MagicMock
        import torch

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4]

        mock_output = MagicMock()
        mock_output.logits = torch.zeros(1, 4, 50)
        # Disable integer indexing — if code uses outputs[1] this raises
        mock_output.__getitem__ = MagicMock(
            side_effect=AssertionError("outputs[1] used instead of outputs.logits")
        )

        mock_model = MagicMock()
        mock_model.return_value = mock_output
        param = torch.nn.Parameter(torch.tensor([0.0]))
        mock_model.parameters = MagicMock(return_value=iter([param]))

        # Should not raise
        result = get_token_logprobs("test", mock_model, mock_tokenizer)
        assert len(result) == 3


# ── device resolution ──────────────────────────────────────────────────────

class TestDeviceResolution:

    def test_uses_next_parameters_device(self):
        """
        Confirm input_ids are moved to next(model.parameters()).device
        not model.device, which can be ambiguous under device_map='auto'.
        """
        import torch
        from unittest.mock import MagicMock, patch
        from src.models import get_token_logprobs

        target_device = torch.device("cpu")

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]

        mock_output = MagicMock()
        mock_output.logits = torch.zeros(1, 3, 50)

        mock_model = MagicMock()
        mock_model.return_value = mock_output

        # next(model.parameters()).device should be used
        param = torch.nn.Parameter(torch.zeros(1).to(target_device))
        mock_model.parameters = MagicMock(return_value=iter([param]))

        # model.device set to something different to catch if it's used instead
        mock_model.device = torch.device("meta")

        # Should complete without error — if model.device were used,
        # moving tensors to 'meta' would raise
        result = get_token_logprobs("test", mock_model, mock_tokenizer)
        assert isinstance(result, list)


# ── slow tests (real model load) ───────────────────────────────────────────

@pytest.mark.slow
class TestLoadModelSlow:
    """Only runs with --run-slow flag."""

    def test_load_pythia_410m(self, request):
        if not request.config.getoption("--run-slow", default=False):
            pytest.skip("Skipping slow model load test")

        from src.models import load_model
        model, tokenizer = load_model("EleutherAI/pythia-410m")

        assert model is not None
        assert tokenizer is not None
        assert not model.training, "Model should be in eval mode"
        assert tokenizer.pad_token is not None, "pad_token should be set"

    def test_logprobs_on_real_model(self, request):
        if not request.config.getoption("--run-slow", default=False):
            pytest.skip("Skipping slow model load test")

        from src.models import load_model, get_token_logprobs
        model, tokenizer = load_model("EleutherAI/pythia-410m")
        text = "The quick brown fox"
        result = get_token_logprobs(text, model, tokenizer)

        assert len(result) > 0
        assert all(lp <= 0 for lp in result), "All log-probs must be <= 0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])