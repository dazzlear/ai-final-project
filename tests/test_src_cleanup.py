"""
Standalone test script for src/data.py, src/baselines.py, and src/metrics.py.

Run from the repo root in PowerShell:

    python tests/test_src_cleanup.py -v

Also compatible with pytest:

    python -m pytest tests/test_src_cleanup.py -v

No GPU, no HuggingFace downloads, no model loading required.
All external calls are mocked.
"""

import importlib.machinery
import os
import sys
import tempfile
import types
import unittest
import zlib
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make src/ importable regardless of where the script is run from
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

# ---------------------------------------------------------------------------
# Mock heavy optional dependencies BEFORE any src module is imported.
#
# Why this block exists:
#   baselines.py imports torch and transformers at the top level.
#   datasets (used by data.py) probes for torch via importlib.util.find_spec.
#   scipy (a sklearn dependency) calls issubclass(cls, torch.Tensor), which
#   requires torch.Tensor to be a real class, not a MagicMock.
#
# These mocks are only needed in environments without torch installed
# (e.g. CI, fresh Colab before !pip install). In normal Colab with torch
# installed, they have no effect.
# ---------------------------------------------------------------------------
_mock_torch = MagicMock()
_mock_torch.__spec__ = importlib.machinery.ModuleSpec("torch", None)
_mock_torch.Tensor = type("Tensor", (), {})   # real class — scipy needs issubclass()
_mock_torch.cuda.is_available.return_value = False
sys.modules.setdefault("torch", _mock_torch)
sys.modules.setdefault("torch.cuda", _mock_torch.cuda)
sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("datasets", MagicMock())


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _make_mock_hf_dataset(n_members: int = 4, n_non_members: int = 4) -> MagicMock:
    """Return a mock that mimics a HuggingFace Dataset with .to_pandas()."""
    rows = (
        [{"input": f"member text {i}",     "label": 1} for i in range(n_members)] +
        [{"input": f"non-member text {i}", "label": 0} for i in range(n_non_members)]
    )
    mock = MagicMock()
    mock.to_pandas.return_value = pd.DataFrame(rows)
    return mock


# ===========================================================================
# src/data.py
# ===========================================================================

class TestLoadWikiMIA(unittest.TestCase):
    """Tests for load_wikimia()."""

    def test_invalid_length_raises(self):
        """Non-supported length must raise ValueError immediately."""
        from data import load_wikimia
        with self.assertRaises(ValueError) as ctx:
            load_wikimia(length=999)
        self.assertIn("Unsupported length", str(ctx.exception))

    def test_invalid_length_message_shows_valid_options(self):
        """Error message must include all valid lengths so the caller knows what to use."""
        from data import load_wikimia
        with self.assertRaises(ValueError) as ctx:
            load_wikimia(length=100)
        msg = str(ctx.exception)
        for valid in ("32", "64", "128", "256"):
            self.assertIn(valid, msg)

    @patch("data.load_dataset")
    def test_returns_correct_columns(self, mock_load):
        """Output DataFrame must have exactly text_id, text, label."""
        from data import load_wikimia
        mock_load.return_value = _make_mock_hf_dataset()
        df = load_wikimia(length=64)
        self.assertListEqual(list(df.columns), ["text_id", "text", "label"])

    @patch("data.load_dataset")
    def test_input_column_renamed_to_text(self, mock_load):
        """WikiMIA HuggingFace upload uses 'input'; must be renamed to 'text'."""
        from data import load_wikimia
        mock_load.return_value = _make_mock_hf_dataset()
        df = load_wikimia(length=64)
        self.assertIn("text", df.columns)
        self.assertNotIn("input", df.columns)

    @patch("data.load_dataset")
    def test_text_id_is_sequential(self, mock_load):
        """text_id must be 0-indexed and sequential for row tracking."""
        from data import load_wikimia
        mock_load.return_value = _make_mock_hf_dataset(n_members=3, n_non_members=3)
        df = load_wikimia(length=64)
        self.assertEqual(list(df["text_id"]), list(range(len(df))))

    @patch("data.load_dataset")
    def test_load_dataset_called_with_config_not_split(self, mock_load):
        """Bug fix: config name must be 2nd positional arg, split must be 'train'.

        The old (broken) call was:
            load_dataset('swj0419/WikiMIA', split='WikiMIA_length64')

        The correct call passes the config name as the second positional argument:
            load_dataset('swj0419/WikiMIA', 'WikiMIA_length128', split='train')
        """
        from data import load_wikimia
        mock_load.return_value = _make_mock_hf_dataset()
        load_wikimia(length=128)
        mock_load.assert_called_once_with(
            "swj0419/WikiMIA", "WikiMIA_length128", split="train"
        )


class TestSaveWikiMIASample(unittest.TestCase):
    """Tests for save_wikimia_sample()."""

    @patch("data.load_dataset")
    def test_sample_is_balanced(self, mock_load):
        """Sample must have exactly half members and half non-members."""
        from data import save_wikimia_sample
        mock_load.return_value = _make_mock_hf_dataset(n_members=20, n_non_members=20)
        # NOTE: os.chdir() is restored in a finally block INSIDE the with block.
        # On Windows, restoring cwd after TemporaryDirectory.__exit__ causes
        # [WinError 32] because the process still holds the directory open.
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                df = save_wikimia_sample(length=64, sample_size=10)
                self.assertEqual(int((df["label"] == 1).sum()), 5)
                self.assertEqual(int((df["label"] == 0).sum()), 5)
            finally:
                os.chdir(old_cwd)  # restore BEFORE TemporaryDirectory cleanup

    @patch("data.load_dataset")
    def test_sample_is_shuffled(self, mock_load):
        """Shuffled sample must not keep all members before all non-members."""
        from data import save_wikimia_sample
        mock_load.return_value = _make_mock_hf_dataset(n_members=20, n_non_members=20)
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                df = save_wikimia_sample(length=64, sample_size=10)
                labels = list(df["label"])
                unshuffled = [1] * 5 + [0] * 5
                self.assertNotEqual(
                    labels, unshuffled,
                    "Labels are in unshuffled order — shuffle is not working"
                )
            finally:
                os.chdir(old_cwd)  # restore BEFORE TemporaryDirectory cleanup


# ===========================================================================
# src/baselines.py
# ===========================================================================

class TestPplScore(unittest.TestCase):
    """Tests for ppl_score()."""

    def test_empty_raises(self):
        from baselines import ppl_score
        with self.assertRaises(ValueError):
            ppl_score([])

    def test_known_value(self):
        """mean([-1, -2, -3]) must be exactly -2.0."""
        from baselines import ppl_score
        self.assertAlmostEqual(ppl_score([-1.0, -2.0, -3.0]), -2.0)

    def test_member_scores_higher_than_non_member(self):
        """Member has less-negative log-probs → higher (less negative) score."""
        from baselines import ppl_score
        self.assertGreater(ppl_score([-0.5, -0.5]), ppl_score([-3.0, -3.0]))


class TestZlibScore(unittest.TestCase):
    """Tests for zlib_score()."""

    def test_empty_logprobs_raises(self):
        from baselines import zlib_score
        with self.assertRaises(ValueError):
            zlib_score([], "hello world")

    def test_empty_text_raises(self):
        from baselines import zlib_score
        with self.assertRaises(ValueError):
            zlib_score([-1.0, -2.0], "")

    def test_known_formula(self):
        """Verify -log_ppl / zlib_len against hand-computed values.

        logprobs = [-1.0, -2.0]  →  mean = -1.5  →  log_ppl = 1.5
        zlib.compress(b'hello world') = 19 bytes
        result = -1.5 / 19 ≈ -0.07895
        """
        from baselines import zlib_score
        text = "hello world"
        expected = -1.5 / len(zlib.compress(text.encode("utf-8")))
        self.assertAlmostEqual(zlib_score([-1.0, -2.0], text), expected, places=6)

    def test_member_scores_higher_than_non_member(self):
        """Member (less-negative logprobs) → higher zlib_score."""
        from baselines import zlib_score
        text = "the quick brown fox jumps over the lazy dog"
        self.assertGreater(
            zlib_score([-0.2] * 10, text),
            zlib_score([-4.0] * 10, text),
        )


class TestLowercaseScore(unittest.TestCase):
    """Tests for lowercase_score()."""

    def test_member_scores_positive(self):
        """Member: model knows original casing better than lowercased → positive score.

        logprob_fn("Hello World") → [-0.5, -0.5]  log_ppl_orig  = 0.5
        logprob_fn("hello world") → [-2.0, -2.0]  log_ppl_lower = 2.0
        result = -(0.5 - 2.0) = 1.5  (positive → member)
        """
        from baselines import lowercase_score
        text = "Hello World"

        def logprob_fn(t):
            return [-0.5, -0.5] if t == text else [-2.0, -2.0]

        self.assertAlmostEqual(lowercase_score(text, logprob_fn), 1.5)


class TestSmallerRefScore(unittest.TestCase):
    """Tests for smaller_ref_score()."""

    def test_empty_target_raises(self):
        from baselines import smaller_ref_score
        with self.assertRaises(ValueError):
            smaller_ref_score([], [-1.0, -2.0])

    def test_empty_ref_raises(self):
        from baselines import smaller_ref_score
        with self.assertRaises(ValueError):
            smaller_ref_score([-1.0, -2.0], [])

    def test_length_mismatch_raises(self):
        """Different-length logprob lists must raise with a clear message."""
        from baselines import smaller_ref_score
        with self.assertRaises(ValueError) as ctx:
            smaller_ref_score([-1.0, -2.0], [-1.0])
        self.assertIn("differ in length", str(ctx.exception))

    def test_known_value_member(self):
        """Member: target PPL lower than ref → positive score of 2.0.

        log_ppl_target = -mean([-1,-2]) = 1.5
        log_ppl_ref    = -mean([-3,-4]) = 3.5
        result         = -(1.5 - 3.5)  = 2.0
        """
        from baselines import smaller_ref_score
        self.assertAlmostEqual(
            smaller_ref_score([-1.0, -2.0], [-3.0, -4.0]), 2.0
        )

    def test_non_member_scores_negative(self):
        """Non-member: target PPL higher than ref → negative score."""
        from baselines import smaller_ref_score
        self.assertLess(
            smaller_ref_score([-3.0, -4.0], [-1.0, -2.0]), 0.0
        )


class TestNeighborScore(unittest.TestCase):
    """Tests for neighbor_score()."""

    def test_degenerate_neighbors_returns_zero(self):
        """If neighbors produce empty logprobs, return 0.0 — not raise.

        The original text must have valid logprobs (so log_ppl_text computes fine).
        Only the neighbors are degenerate. This is the documented edge case.
        """
        from baselines import neighbor_score
        with patch("baselines._generate_neighbors", return_value=["nbr_a", "nbr_b"]):
            # Original: valid. Neighbors: degenerate (empty).
            def logprob_fn(t):
                return [-1.0, -1.0] if t == "some text" else []
            score = neighbor_score("some text", logprob_fn=logprob_fn)
            self.assertEqual(score, 0.0)

    def test_member_scores_positive(self):
        """Member sits at local PPL minimum → neighbors have higher PPL → positive score.

        logprob_fn("original text") → [-0.5,-0.5]  log_ppl = 0.5
        logprob_fn(neighbor)        → [-3.0,-3.0]  log_ppl = 3.0
        result = mean([3.0, 3.0]) - 0.5 = 2.5
        """
        from baselines import neighbor_score

        def logprob_fn(t):
            return [-0.5, -0.5] if t == "original text" else [-3.0, -3.0]

        with patch("baselines._generate_neighbors", return_value=["n1", "n2"]):
            self.assertAlmostEqual(neighbor_score("original text", logprob_fn), 2.5)

    def test_non_member_scores_zero(self):
        """Non-member has same PPL as neighbors → score = 0.0."""
        from baselines import neighbor_score
        with patch("baselines._generate_neighbors", return_value=["n1"]):
            score = neighbor_score("original text", logprob_fn=lambda t: [-2.0, -2.0])
            self.assertAlmostEqual(score, 0.0)


# ===========================================================================
# src/metrics.py
# ===========================================================================

class TestValidateInputs(unittest.TestCase):
    """Tests for the internal _validate_inputs() helper."""

    def _call(self, scores, labels):
        from metrics import _validate_inputs
        return _validate_inputs(scores, labels)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._call([0.5, 0.8], [1])
        self.assertIn("Length mismatch", str(ctx.exception))

    def test_nan_in_scores_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._call([float("nan"), 0.5], [1, 0])
        self.assertIn("NaN or inf", str(ctx.exception))

    def test_inf_in_scores_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._call([float("inf"), 0.5], [1, 0])
        self.assertIn("NaN or inf", str(ctx.exception))

    def test_invalid_label_value_raises(self):
        """Label value of 2 is not 0 or 1 — must raise with a clear message."""
        with self.assertRaises(ValueError) as ctx:
            self._call([0.5, 0.8], [1, 2])
        self.assertIn("only 0 and 1", str(ctx.exception))

    def test_single_class_raises_distinct_message(self):
        """[1,1,1] are valid values but single-class — must be a DISTINCT error from
        invalid-value errors so callers can tell the two failures apart."""
        with self.assertRaises(ValueError) as ctx:
            self._call([0.5, 0.8, 0.3], [1, 1, 1])
        self.assertIn("Both classes", str(ctx.exception))
        self.assertNotIn("only 0 and 1", str(ctx.exception))

    def test_valid_inputs_return_correct_dtypes(self):
        scores_arr, labels_arr = self._call([0.5, 0.8], [1, 0])
        self.assertEqual(scores_arr.dtype, np.float64)
        self.assertEqual(labels_arr.dtype, np.int64)


class TestComputeAuc(unittest.TestCase):
    """Tests for compute_auc()."""

    def test_perfect_classifier_returns_one(self):
        from metrics import compute_auc
        self.assertAlmostEqual(
            compute_auc([1.0, 1.0, 0.0, 0.0], [1, 1, 0, 0]), 1.0
        )

    def test_inverted_classifier_returns_zero(self):
        """Scores perfectly backwards → AUC = 0.0."""
        from metrics import compute_auc
        self.assertAlmostEqual(
            compute_auc([0.0, 0.0, 1.0, 1.0], [1, 1, 0, 0]), 0.0
        )

    def test_result_is_in_unit_interval(self):
        from metrics import compute_auc
        result = compute_auc([0.6, 0.4, 0.6, 0.4], [1, 0, 1, 0])
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_inherits_validation(self):
        """compute_auc must propagate _validate_inputs errors."""
        from metrics import compute_auc
        with self.assertRaises(ValueError):
            compute_auc([float("nan"), 0.5], [1, 0])


class TestTprAtFpr(unittest.TestCase):
    """Tests for tpr_at_fpr()."""

    def test_fpr_above_one_raises(self):
        from metrics import tpr_at_fpr
        with self.assertRaises(ValueError) as ctx:
            tpr_at_fpr([0.8, 0.2], [1, 0], fpr=3.0)
        self.assertIn("fpr must be in [0, 1]", str(ctx.exception))

    def test_fpr_below_zero_raises(self):
        from metrics import tpr_at_fpr
        with self.assertRaises(ValueError):
            tpr_at_fpr([0.8, 0.2], [1, 0], fpr=-0.1)

    def test_perfect_classifier_achieves_full_tpr(self):
        """Perfect scores → TPR = 1.0 even at the tightest threshold (5% FPR)."""
        from metrics import tpr_at_fpr
        self.assertAlmostEqual(
            tpr_at_fpr([1.0, 1.0, 0.0, 0.0], [1, 1, 0, 0], fpr=0.05), 1.0
        )

    def test_result_is_in_unit_interval(self):
        from metrics import tpr_at_fpr
        result = tpr_at_fpr([0.9, 0.4, 0.7, 0.2], [1, 0, 1, 0], fpr=0.05)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)


class TestRocPoints(unittest.TestCase):
    """Tests for roc_points()."""

    def _inputs(self):
        return [0.9, 0.4, 0.7, 0.2], [1, 0, 1, 0]

    def test_returns_two_numpy_arrays(self):
        from metrics import roc_points
        fpr, tpr = roc_points(*self._inputs())
        self.assertIsInstance(fpr, np.ndarray)
        self.assertIsInstance(tpr, np.ndarray)

    def test_fpr_and_tpr_same_length(self):
        from metrics import roc_points
        fpr, tpr = roc_points(*self._inputs())
        self.assertEqual(len(fpr), len(tpr))

    def test_fpr_starts_at_zero(self):
        from metrics import roc_points
        fpr, _ = roc_points(*self._inputs())
        self.assertAlmostEqual(fpr[0], 0.0)

    def test_fpr_ends_at_one(self):
        from metrics import roc_points
        fpr, _ = roc_points(*self._inputs())
        self.assertAlmostEqual(fpr[-1], 1.0)

    def test_inherits_validation(self):
        """Single-class labels must raise through roc_points."""
        from metrics import roc_points
        with self.assertRaises(ValueError):
            roc_points([0.5, 0.5], [1, 1])


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)