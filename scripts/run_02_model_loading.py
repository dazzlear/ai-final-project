"""Module 02 — model loading and verification.

Loads each model, captures diagnostics, runs tokenization roundtrip
and forward pass checks, then unloads. Outputs: model_verification_report.csv
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR    = SCRIPT_DIR.parent / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models import load_model  # noqa: E402

PROBE_TEXT = 'The quick brown fox jumps over the lazy dog.'


def verify_model(model_name: str) -> dict:
    """Load model, capture diagnostics, run checks, unload. Returns result dict."""
    model_key = model_name.split('/')[-1].replace('-', '').replace('.', '')
    result = {
        'model_key'      : model_key,
        'model_name'     : model_name,
        'architecture'   : '',
        'parameters_M'   : 0.0,
        'device'         : '',
        'vram_mb'        : 0.0,
        'vocab_size'     : 0,
        'roundtrip_ok'   : False,
        'forward_pass_ok': False,
        'error'          : '',
        'timestamp'      : datetime.now().isoformat(timespec='seconds'),
    }

    try:
        model, tokenizer = load_model(model_name)

        # Capture architecture and parameter count
        result['architecture'] = type(model).__name__
        result['parameters_M'] = sum(p.numel() for p in model.parameters()) / 1e6

        # Capture device and VRAM usage
        device = next(model.parameters()).device
        result['device'] = str(device)
        if device.type == 'cuda':
            result['vram_mb'] = torch.cuda.memory_allocated(device) / 1e6

        result['vocab_size'] = tokenizer.vocab_size

        # Test tokenization roundtrip
        ids     = tokenizer.encode(PROBE_TEXT)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        result['roundtrip_ok'] = (decoded.strip() == PROBE_TEXT.strip())

        # Test forward pass produces valid logits
        input_ids = torch.tensor([ids]).to(device)
        with torch.no_grad():
            outputs = model(input_ids)
        logits = outputs.logits
        # Shape must be [1, seq_len, vocab_size] with no NaN/Inf
        shape_ok = (
            logits.ndim == 3
            and logits.shape[0] == 1
            and logits.shape[2] == tokenizer.vocab_size
        )
        values_ok = torch.isfinite(logits).all().item()
        result['forward_pass_ok'] = bool(shape_ok and values_ok)

        # Print diagnostics
        print(f'\n  Model        : {model_name}')
        print(f'  Architecture : {result["architecture"]}')
        print(f'  Parameters   : {result["parameters_M"]:.1f}M')
        print(f'  Device       : {result["device"]}')
        print(f'  VRAM used    : {result["vram_mb"]:.0f} MB')
        print(f'  Vocab size   : {result["vocab_size"]:,}')
        print(f'  Roundtrip    : {"✅" if result["roundtrip_ok"] else "❌"}')
        print(f'  Forward pass : {"✅" if result["forward_pass_ok"] else "❌"}  '
              f'(logits shape: {list(logits.shape)})')

    except Exception as exc:
        result['error'] = str(exc)
        print(f'\n  ❌ {model_name} failed: {exc}')

    finally:
        # Always unload to free VRAM for next model
        try:
            del model
        except NameError:
            pass
        torch.cuda.empty_cache()
        print('  GPU cache cleared.')

    return result


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"═"*60}')
    print(f'  Module 02 — Model Loading and Verification')
    print(f'  Models to verify: {len(args.models)}')
    print(f'{"═"*60}')

    results = []
    for model_name in args.models:
        results.append(verify_model(model_name))

    # Save verification report
    report_path = output_dir / 'model_verification_report.csv'
    pd.DataFrame(results).to_csv(report_path, index=False)
    print(f'\n{"═"*60}')
    print(f'  Report saved → {report_path}')

    passed = sum(
        r['roundtrip_ok'] and r['forward_pass_ok'] and not r['error']
        for r in results
    )
    print(f'  {passed} / {len(results)} models passed all checks.')
    if passed == len(results):
        print('✅  Module 02 complete — all models verified.')
    else:
        print('❌  Some models failed — check report for details.')


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description='Module 02 — verify model loading for all configured models.'
    )
    p.add_argument(
        '--models',
        nargs='+',
        required=True,
        help='Space-separated list of HuggingFace model IDs to verify.',
    )
    p.add_argument(
        '--output_dir',
        required=True,
        help='Path to Module 02 output directory.',
    )
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())