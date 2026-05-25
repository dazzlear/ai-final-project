"""
Tests for reference/ folder integrity.
Confirms the paper's original files were moved correctly and are untouched.
Run from repo root: python -m pytest tests/test_reference.py -v
"""
import os
import sys
import pytest
sys.path.insert(0, os.path.abspath("."))


class TestReferenceFolder:

    def test_reference_folder_exists(self):
        """reference/ folder must exist."""
        assert os.path.isdir("reference"), (
            "reference/ folder not found. "
            "Create it and move run_original.py, eval_original.py, "
            "options_original.py into it."
        )

    def test_run_original_exists(self):
        """Paper's original run.py must be in reference/."""
        assert os.path.isfile("reference/run_original.py"), (
            "reference/run_original.py not found. "
            "Move src/run.py → reference/run_original.py."
        )

    def test_eval_original_exists(self):
        """Paper's original eval.py must be in reference/."""
        assert os.path.isfile("reference/eval_original.py"), (
            "reference/eval_original.py not found. "
            "Move src/eval.py → reference/eval_original.py."
        )

    def test_options_original_exists(self):
        """Paper's original options.py must be in reference/."""
        assert os.path.isfile("reference/options_original.py"), (
            "reference/options_original.py not found. "
            "Move src/options.py → reference/options_original.py."
        )

    def test_run_original_has_warning_comment(self):
        """run_original.py should have a REFERENCE ONLY comment at the top."""
        with open("reference/run_original.py", "r") as f:
            content = f.read()
        assert "REFERENCE" in content.upper(), (
            "run_original.py is missing the REFERENCE ONLY warning comment. "
            "Add it so teammates know not to import from this file."
        )

    def test_paper_files_not_in_src(self):
        """
        Original paper files should NOT remain in src/.
        After moving to reference/, src/ should be clean.
        """
        paper_files = ["src/run.py", "src/eval.py", "src/options.py"]
        found = [f for f in paper_files if os.path.isfile(f)]
        assert not found, (
            f"Paper files still in src/: {found}. "
            "Move them to reference/ and remove from src/."
        )

    def test_evaluate_py_removed(self):
        """src/evaluate.py should be deleted (unused re-export shim)."""
        assert not os.path.isfile("src/evaluate.py"), (
            "src/evaluate.py still exists. "
            "It is an unused re-export shim — delete it."
        )


class TestSrcStructure:
    """Confirm src/ has exactly the right files after cleanup."""

    EXPECTED = {
        "__init__.py",
        "data.py",
        "models.py",
        "methods.py",
        "baselines.py",
        "metrics.py",
    }

    def test_src_has_expected_files(self):
        """src/ should contain exactly the expected module files."""
        actual = {
            f for f in os.listdir("src")
            if f.endswith(".py") and not f.startswith(".")
        }
        missing = self.EXPECTED - actual
        assert not missing, f"Missing from src/: {missing}"

    def test_src_has_no_unexpected_files(self):
        """src/ should not contain any unexpected .py files."""
        actual = {
            f for f in os.listdir("src")
            if f.endswith(".py") and not f.startswith(".")
        }
        unexpected = actual - self.EXPECTED
        assert not unexpected, (
            f"Unexpected files in src/: {unexpected}. "
            "Move them to reference/ or scripts/ as appropriate."
        )


class TestOutputsStructure:
    """Confirm outputs/ has the module-based folder structure."""

    EXPECTED_DIRS = [
        "outputs/01_dataset",
        "outputs/02_logprobs",
        "outputs/03_mink_scores",
        "outputs/04_baseline_scores",
        "outputs/05_evaluation",
    ]

    def test_output_dirs_exist(self):
        """All five module output directories must exist."""
        missing = [d for d in self.EXPECTED_DIRS if not os.path.isdir(d)]
        assert not missing, (
            f"Missing output directories: {missing}. "
            "Create them with mkdir."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])