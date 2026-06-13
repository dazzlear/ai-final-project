"""Module 03 — Min-K% Prob Scoring.

Reads dataset CSVs from Module 01, computes Min-K% Prob scores for each
model length setting, writes scores to DRIVE_03.
"""

import argparse
import os
import sys
import pickle
import time
import traceback
from pathlib import Path

import pandas as pd
import torch

# Resolve src/ on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / 'src'))

from models  import load_model
from methods import min_k_prob, select_min_k_tokens
import log_probability_compute_auto as auto_impl
import log_probability_compute_manual as manual_impl

# Maps --implementation choices to the modules providing
# compute_token_logprobs(). Both expose an identical signature, so the
# rest of the pipeline doesn't need to know which one is active.
_LOGPROB_IMPLS = {
    'auto'  : auto_impl,
    'manual': manual_impl,
}


def _log(msg: str, log_fh=None):
    """Print to stdout and optionally log file."""
    print(msg, flush=True)
    if log_fh:
        print(msg, file=log_fh, flush=True)


def _score_dataframe(
    df: pd.DataFrame,
    model,
    tokenizer,
    k: int,
    device: str,
    compute_token_logprobs,
    log_fh=None,
) -> pd.DataFrame:
    """Score all rows, compute Min-K% scores, return augmented DataFrame."""
    texts      = df['text'].tolist()
    total      = len(texts)
    all_lp     = []
    min_k_scores, n_tokens_list, n_selected_list = [], [], []

    t0 = time.time()
    for i, text in enumerate(texts, 1):
        # Compute log-probs and Min-K% score for this row
        lp    = compute_token_logprobs(text, model, tokenizer, device=device)
        score = min_k_prob(lp, k=k)
        _, sel, _ = select_min_k_tokens(lp, k=k)

        all_lp.append(lp)
        min_k_scores.append(score)
        n_tokens_list.append(len(lp))
        n_selected_list.append(len(sel))

        # Progress update every 100 rows
        if i % 100 == 0 or i == total:
            elapsed = time.time() - t0
            rate    = i / elapsed
            eta     = (total - i) / rate if rate > 0 else 0
            _log(f'  [{i:>4}/{total}]  elapsed {elapsed:.0f}s  ETA {eta:.0f}s', log_fh)

    out = df.copy()
    out['min_k_score'] = min_k_scores
    out['n_tokens']    = n_tokens_list
    out['n_selected']  = n_selected_list

    return out, all_lp


def main():
    parser = argparse.ArgumentParser(description='Module 03 — Min-K% Prob Scoring')
    parser.add_argument('--input_dir',    required=True,
                        help='Path to DRIVE_01 (processed dataset CSVs)')
    parser.add_argument('--output_dir',   required=True,
                        help='Path to DRIVE_03 (Min-K% score output)')
    parser.add_argument('--models',       nargs='+', required=True,
                        help='HuggingFace model IDs (space-separated)')
    parser.add_argument('--model_keys',   nargs='+', required=True,
                        help='Short keys matching --models')
    parser.add_argument('--lengths',      nargs='+', type=int, default=[64],
                        help='WikiMIA length buckets to process')
    parser.add_argument('--settings',     nargs='+', default=['original'],
                        choices=['original', 'paraphrase'],
                        help='Dataset settings to process')
    parser.add_argument('--k',            type=int, default=20,
                        help='Min-K%% percentage (default: 20)')
    parser.add_argument('--implementation', choices=['auto', 'manual'], default='auto',
                        help='Log-probability computation backend. "auto" uses '
                             'torch.nn.functional.log_softmax / Tensor.gather() '
                             '(fast, for full dataset runs). "manual" computes '
                             'softmax/log/gather with plain Python loops '
                             '(slow, for smoke tests and verification only).')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip combos whose output CSV already exists')
    parser.add_argument('--cache_logprobs', action='store_true',
                        help='Save raw log-prob lists as .pkl files')
    parser.add_argument('--smoke',         action='store_true',
                        help='Run on first 10 rows only (smoke test)')
    args = parser.parse_args()

    # Validate model args match
    if len(args.models) != len(args.model_keys):
        parser.error('--models and --model_keys must have the same number of entries')

    model_map = dict(zip(args.model_keys, args.models))

    # Resolve the log-probability implementation once for the whole run
    compute_token_logprobs = _LOGPROB_IMPLS[args.implementation].compute_token_logprobs

    # Create output directories
    out_dir      = Path(args.output_dir)
    logprobs_dir = out_dir / 'logprobs'
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.cache_logprobs:
        logprobs_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    log_path = out_dir / 'run_log.txt'
    with open(log_path, 'a') as log_fh:
        _log(f'\n{"="*60}', log_fh)
        _log(f'Module 03 — Min-K% Prob Scoring', log_fh)
        _log(f'k={args.k}  device={device}  smoke={args.smoke}  '
             f'implementation={args.implementation}', log_fh)
        _log(f'models   : {list(model_map.keys())}', log_fh)
        _log(f'lengths  : {args.lengths}', log_fh)
        _log(f'settings : {args.settings}', log_fh)
        _log(f'{"="*60}', log_fh)

        completed, skipped, failed = [], [], []

        for model_key, model_hf_id in model_map.items():
            _log(f'\nModel: {model_key} ({model_hf_id})', log_fh)

            # Collect all combos for this model
            combos_to_run = []
            for length in args.lengths:
                for setting in args.settings:
                    # Tag non-"auto" implementations with a suffix so manual
                    # and auto outputs don't collide / overwrite each other.
                    suffix = '' if args.implementation == 'auto' else f'_{args.implementation}'
                    out_csv = out_dir / f'mink_scores_{model_key}_len{length}_{setting}{suffix}.csv'
                    if args.skip_existing and out_csv.exists():
                        _log(f'  SKIP (exists): {out_csv.name}', log_fh)
                        skipped.append(str(out_csv.name))
                        continue
                    combos_to_run.append((length, setting, out_csv))

            if not combos_to_run:
                _log(f'  All combos done for {model_key}. Skipping model load.', log_fh)
                continue

            # Load model once per model_key
            try:
                _log(f'  Loading model ...', log_fh)
                t_load = time.time()
                model, tokenizer = load_model(model_hf_id, device=device)
                _log(f'  Loaded in {time.time() - t_load:.1f}s', log_fh)
            except Exception as e:
                _log(f'  ERROR loading {model_key}: {e}', log_fh)
                for _, _, out_csv in combos_to_run:
                    failed.append(str(out_csv.name))
                continue

            # Score each length × setting combo
            for length, setting, out_csv in combos_to_run:
                _log(f'\n  combo: len={length}  setting={setting}', log_fh)

                # Determine text column for this setting
                text_col = 'text' if setting == 'original' else 'paraphrase_text'
                in_csv   = Path(args.input_dir) / f'wikimia_length{length}_processed.csv'

                if not in_csv.exists():
                    _log(f'  ERROR: input file not found: {in_csv}', log_fh)
                    failed.append(out_csv.name)
                    continue

                try:
                    df_in = pd.read_csv(in_csv)

                    # Smoke test: use first 10 rows
                    if args.smoke:
                        df_in = df_in.head(10).copy()
                        _log(f'  [SMOKE] using first {len(df_in)} rows', log_fh)

                    # Fall back to 'text' if paraphrase column missing
                    if setting == 'paraphrase' and text_col not in df_in.columns:
                        _log(f'  WARN: column "{text_col}" not found; falling back to "text"', log_fh)
                        text_col = 'text'

                    df_work = df_in[['text_id', text_col, 'label']].copy()
                    df_work = df_work.rename(columns={text_col: 'text'})

                    _log(f'  scoring {len(df_work)} rows ...', log_fh)
                    df_scored, all_lp = _score_dataframe(
                        df_work, model, tokenizer,
                        k=args.k, device=device,
                        compute_token_logprobs=compute_token_logprobs,
                        log_fh=log_fh,
                    )

                    # Save scores CSV
                    df_scored.to_csv(out_csv, index=False)
                    _log(f'  saved: {out_csv}', log_fh)

                    # Optional: cache raw log-probs
                    if args.cache_logprobs:
                        pkl_path = logprobs_dir / f'logprobs_{model_key}_len{length}_{setting}.pkl'
                        with open(pkl_path, 'wb') as f:
                            pickle.dump({'logprobs': all_lp, 'text_ids': df_work['text_id'].tolist()}, f)
                        _log(f'  cached logprobs: {pkl_path}', log_fh)

                    # Sanity check: member mean > non-member mean
                    m1 = df_scored[df_scored['label'] == 1]['min_k_score'].mean()
                    m0 = df_scored[df_scored['label'] == 0]['min_k_score'].mean()
                    direction = '(member > non-member)' if m1 > m0 else 'unexpected direction'
                    _log(f'  member_mean={m1:.4f}  non_member_mean={m0:.4f}  {direction}', log_fh)

                    completed.append(out_csv.name)

                except Exception:
                    _log(f'  ERROR scoring {model_key} len={length} {setting}:', log_fh)
                    _log(traceback.format_exc(), log_fh)
                    failed.append(out_csv.name)

            # Unload model before next model_key
            del model, tokenizer
            torch.cuda.empty_cache()
            _log(f'\n  GPU cache cleared.', log_fh)

        # Final summary
        _log(f'\n{"="*60}', log_fh)
        _log(f'DONE.  completed={len(completed)}  skipped={len(skipped)}  failed={len(failed)}', log_fh)
        if failed:
            _log(f'FAILED: {failed}', log_fh)
        _log(f'{"="*60}', log_fh)

    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()