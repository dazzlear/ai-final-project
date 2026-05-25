"""
smoke_test_module03.py
======================
Offline smoke test for Module 03 — no GPU, no large models, no internet.

Four test groups:
  [A] Pure algorithm  — select_min_k_tokens, min_k_prob (no model needed)
  [B] Mock model      — tokenize_text, compute_token_logprobs with injected
                        tiny model and mock tokenizer
  [C] Full pipeline   — score_texts() end-to-end with the mock
  [D] Script CLI      — run_03_mink.py via subprocess with a tiny CSV
                        and a patched load_model

Run:
    python smoke_test_module03.py
"""

import sys, os, math, csv, subprocess, textwrap, traceback
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch

# ── path setup ────────────────────────────────────────────────────────────────
# Works whether the script lives at project root OR inside tests/
# Strategy: walk up from __file__ until we find a directory that contains src/
def _find_project_root(start: Path) -> Path:
    for candidate in [start, start.parent, start.parent.parent]:
        if (candidate / 'src').is_dir():
            return candidate
    raise RuntimeError(
        f"Could not find project root (a directory containing src/) "
        f"starting from {start}. "
        f"Make sure src/methods.py exists."
    )

ROOT = _find_project_root(Path(__file__).resolve().parent)
SRC  = ROOT / 'src'
sys.path.insert(0, str(SRC))
print(f'Project root : {ROOT}')
print(f'src/         : {SRC}')

import methods

# ── shared helper: TensorDict (dict that supports .to() and ** unpacking) ──────
class TensorDict(dict):
    """dict subclass that supports .to(device) so it behaves like HF BatchEncoding."""
    def to(self, device):
        return TensorDict({k: (v.to(device) if hasattr(v, 'to') else v)
                           for k, v in self.items()})

# ── tiny mock tokenizer ───────────────────────────────────────────────────────
class MockTokenizer:
    """
    Minimal tokenizer mock.  Words split on whitespace; each word maps to its
    character-sum mod 1000.  Does NOT download anything.
    """
    vocab_size = 1000

    def __call__(self, text, return_tensors=None,
                 truncation=None, max_length=None):
        words = text.split()
        if max_length:
            words = words[:max_length]
        ids = [sum(ord(c) for c in w) % self.vocab_size for w in words]
        if not ids:
            ids = [0]
        return TensorDict({'input_ids'      : torch.tensor([ids]),
                           'attention_mask' : torch.ones(1, len(ids), dtype=torch.long)})

    def decode(self, ids, clean_up_tokenization_spaces=False):
        return ' '.join(f'tok{i}' for i in ids)


# ── tiny mock model ───────────────────────────────────────────────────────────
class TinyModel(torch.nn.Module):
    """
    Tiny trainable model that produces real logits.
    Input: (1, T) int tensor.  Output: MagicMock with .logits of shape (1, T, V).
    """
    def __init__(self, vocab: int = 1000, dim: int = 8):
        super().__init__()
        self._emb = torch.nn.Embedding(vocab, dim)
        self._lm  = torch.nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids=None, attention_mask=None, **kw):
        h      = self._emb(input_ids)    # (1, T, dim)
        logits = self._lm(h)             # (1, T, vocab)
        out    = MagicMock()
        out.logits = logits
        return out

    def to(self, *a, **kw):
        return self


# ── test helpers ──────────────────────────────────────────────────────────────
PASS_MARK = '  ✅'
FAIL_MARK = '  ❌'
_results  = []

def check(name: str, condition: bool, detail: str = ''):
    mark = PASS_MARK if condition else FAIL_MARK
    line = f'{mark}  {name}'
    if detail:
        line += f'  ({detail})'
    print(line)
    _results.append((name, condition))
    return condition

def section(title: str):
    print(f'\n{"─"*60}')
    print(f'  {title}')
    print(f'{"─"*60}')


# ══════════════════════════════════════════════════════════════════════════════
# GROUP A — Pure algorithm functions (no model, no tokenizer)
# ══════════════════════════════════════════════════════════════════════════════

section('[A] Pure algorithm — select_min_k_tokens & min_k_prob')

lp_10 = [-1.0, -0.5, -3.0, -0.2, -4.5, -0.8, -2.1, -0.1, -3.8, -1.5]

# A1: ceil(10 * 0.20) = 2 tokens selected
sel_idx, sel_lp, rank = methods.select_min_k_tokens(lp_10, k=20)
check('A1  n_selected = ceil(10 * 0.20) = 2',
      len(sel_idx) == 2, f'got {len(sel_idx)}')

# A2: selected are the two most negative values
expected_bottom2 = sorted(lp_10)[:2]
check('A2  selected values are the two lowest log-probs',
      sorted(sel_lp) == expected_bottom2,
      f'selected={sorted(sel_lp)}  expected={expected_bottom2}')

# A3: rank_order[0] points to the min log-prob token
check('A3  rank_order[0] is the most surprising token',
      lp_10[rank[0]] == min(lp_10), f'lp[rank[0]]={lp_10[rank[0]]}')

# A4: k=100 selects all tokens
sel_all, _, _ = methods.select_min_k_tokens(lp_10, k=100)
check('A4  k=100 selects all tokens',
      len(sel_all) == len(lp_10), f'got {len(sel_all)}')

# A5: single token → minimum is 1
sel_one, _, _ = methods.select_min_k_tokens([-2.0], k=20)
check('A5  single-token input → 1 selected', len(sel_one) == 1)

# A6: empty input → empty output, no crash
sel_e, slp_e, rank_e = methods.select_min_k_tokens([], k=20)
check('A6  empty input → empty output (no crash)',
      sel_e == [] and slp_e == [] and rank_e == [])

# A7: min_k_prob == mean(selected)
score = methods.min_k_prob(lp_10, k=20)
expected_score = float(np.mean(expected_bottom2))
check('A7  min_k_prob == mean(selected_logprobs)',
      abs(score - expected_score) < 1e-6,
      f'got {score:.6f}  expected {expected_score:.6f}')

# A8: member text (high probs) scores higher than non-member (low probs)
lp_member     = [-0.3, -0.1, -0.5, -0.2, -0.4, -0.1, -0.3, -0.2, -0.4, -0.3]
lp_non_member = [-3.1, -4.5, -2.8, -5.2, -3.7, -4.1, -2.9, -5.5, -3.3, -4.0]
score_m  = methods.min_k_prob(lp_member, k=20)
score_nm = methods.min_k_prob(lp_non_member, k=20)
check('A8  member score > non-member score',
      score_m > score_nm,
      f'member={score_m:.4f}  non-member={score_nm:.4f}')

# A9: empty input → 0.0, no crash
check('A9  min_k_prob([]) == 0.0', methods.min_k_prob([]) == 0.0)

# A10: score increases monotonically as k grows
lp_mixed = [-5.0, -4.0, -3.0, -2.0, -1.0, -0.5, -0.3, -0.2, -0.1, -0.05]
scores_k = [methods.min_k_prob(lp_mixed, k=p) for p in [10, 20, 30, 50, 100]]
check('A10 score increases as k grows',
      all(scores_k[i] <= scores_k[i+1] for i in range(len(scores_k)-1)),
      f'{[round(s,3) for s in scores_k]}')

# A11: n_selected formula holds across several (N, k) pairs
for n_tok, k_pct, exp_n in [(50, 20, 10), (50, 10, 5), (7, 20, 2), (3, 20, 1)]:
    lp_t = [-float(i) for i in range(n_tok)]
    s_idx, _, _ = methods.select_min_k_tokens(lp_t, k=k_pct)
    check(f'A11 n_selected(N={n_tok}, k={k_pct}%) = {exp_n}',
          len(s_idx) == exp_n, f'got {len(s_idx)}')


# ══════════════════════════════════════════════════════════════════════════════
# GROUP B — Stages 1 & 2 with mock model/tokenizer
# ══════════════════════════════════════════════════════════════════════════════

section('[B] Staged pipeline with mock model/tokenizer')

_tok   = MockTokenizer()
_model = TinyModel(vocab=_tok.vocab_size)

TEXT_MEMBER     = (
    "The Battle of Hastings was fought on 14 October 1066 between the Norman "
    "army of William the Conqueror and an English army under King Harold II."
)
TEXT_NON_MEMBER = "zxqvw blorple flumtrix snorkel quibble wazzle frumble."

# B1: tokenize_text returns all required keys
tok_result = methods.tokenize_text(TEXT_MEMBER, _tok)
for key in ('input_ids', 'tokens', 'n_tokens', 'was_truncated'):
    check(f'B1  tokenize_text has key: {key}', key in tok_result)

# B2: n_tokens == len(input_ids)
check('B2  n_tokens == len(input_ids)',
      tok_result['n_tokens'] == len(tok_result['input_ids']))

# B3: tokens list same length as input_ids
check('B3  len(tokens) == len(input_ids)',
      len(tok_result['tokens']) == len(tok_result['input_ids']))

# B4: short text not truncated
check('B4  short text → was_truncated=False',
      tok_result['was_truncated'] == False)

# B5: truncation flag fires for a very long text
long_text = ' '.join(['word'] * 2000)
tok_long  = methods.tokenize_text(long_text, _tok, max_length=20)
check('B5  long text → was_truncated=True', tok_long['was_truncated'] == True)

# B6: compute_token_logprobs returns a list
lp_result = methods.compute_token_logprobs(
    TEXT_MEMBER, _model, _tok, device='cpu'
)
check('B6  compute_token_logprobs returns a list', isinstance(lp_result, list))

# B7: length = n_tokens - 1
N = tok_result['n_tokens']
check('B7  len(logprobs) == n_tokens - 1',
      len(lp_result) == N - 1, f'got {len(lp_result)}  expected {N - 1}')

# B8: all values are floats
check('B8  all logprobs are floats',
      all(isinstance(v, float) for v in lp_result))

# B9: all values are ≤ 0 (valid log-probs from log_softmax)
check('B9  all logprobs ≤ 0',
      all(v <= 0.01 for v in lp_result),
      f'min={min(lp_result):.4f}  max={max(lp_result):.4f}')

# B10: different texts produce different log-prob lists
lp_nm = methods.compute_token_logprobs(TEXT_NON_MEMBER, _model, _tok, device='cpu')
check('B10 different texts produce different logprob lists',
      lp_result[:5] != lp_nm[:5])


# ══════════════════════════════════════════════════════════════════════════════
# GROUP C — score_texts() end-to-end
# ══════════════════════════════════════════════════════════════════════════════

section('[C] score_texts() full pipeline (mock model)')

TEXTS  = [TEXT_MEMBER, TEXT_NON_MEMBER,
          "Paris is the capital of France.",
          "norp flizzle quork wazzle mibble."]
LABELS = [1, 0, 1, 0]

results = methods.score_texts(
    TEXTS, _model, _tok, k=20, device='cpu', show_progress=False
)

check('C1  len(results) == len(TEXTS)',
      len(results) == len(TEXTS), f'got {len(results)}')

required_keys = {'min_k_score', 'n_tokens', 'n_selected'}
for i, res in enumerate(results):
    missing = required_keys - set(res.keys())
    check(f'C2  result[{i}] has required keys', len(missing) == 0,
          f'missing={missing}')

for i, res in enumerate(results):
    expected_sel = max(1, math.ceil(res['n_tokens'] * 20 / 100))
    check(f'C3  result[{i}] n_selected == ceil(n_tokens*0.2)',
          res['n_selected'] == expected_sel,
          f'got {res["n_selected"]}  expected {expected_sel}')

check('C4  no NaN scores',
      all(not math.isnan(r['min_k_score']) for r in results))

check('C5  n_tokens > 0 for all texts',
      all(r['n_tokens'] > 0 for r in results))

print(f'\n  Score summary (untrained model — direction not guaranteed):')
for text, label, res in zip(TEXTS, LABELS, results):
    tag = 'MEMBER ' if label == 1 else 'NON-MBR'
    print(f'    [{tag}] score={res["min_k_score"]:8.4f}  '
          f'tok={res["n_tokens"]}  sel={res["n_selected"]}  '
          f'"{text[:55]}..."')


# ══════════════════════════════════════════════════════════════════════════════
# GROUP D — run_03_mink.py CLI via subprocess
# ══════════════════════════════════════════════════════════════════════════════

section('[D] run_03_mink.py CLI — end-to-end script test')

DATA_DIR = ROOT / 'test_data'
OUT_DIR  = ROOT / 'test_outputs'
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

FAKE_ROWS = [
    {'text_id': 0, 'text': TEXT_MEMBER,
     'label': 1, 'paraphrase_text': 'Normans fought Harold at Hastings 1066.'},
    {'text_id': 1, 'text': TEXT_NON_MEMBER,
     'label': 0, 'paraphrase_text': 'zxqvw blorple flumtrix snorkel.'},
    {'text_id': 2, 'text': "Paris is the capital of France.",
     'label': 1, 'paraphrase_text': "France's capital is Paris."},
    {'text_id': 3, 'text': "norp flizzle quork wazzle.",
     'label': 0, 'paraphrase_text': "norp flizzle quork wazzle blorb."},
]

fake_csv = DATA_DIR / 'wikimia_length64_processed.csv'
with open(fake_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['text_id', 'text', 'label', 'paraphrase_text'])
    w.writeheader()
    w.writerows(FAKE_ROWS)
print(f'  Fake dataset written: {fake_csv.name}  ({len(FAKE_ROWS)} rows)')

# ── Patch models.py with TinyModel so the script doesn't need HuggingFace ─────
PATCH_CODE = textwrap.dedent(f"""
import sys, torch
from unittest.mock import MagicMock
sys.path.insert(0, {str(SRC)!r})

class _TensorDict(dict):
    def to(self, device): return self

class _MockTok:
    vocab_size = 1000
    def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
        words = text.split()
        if max_length: words = words[:max_length]
        ids = [sum(ord(c) for c in w) % self.vocab_size for w in words] or [0]
        return _TensorDict({{'input_ids': torch.tensor([ids]),
                            'attention_mask': torch.ones(1, len(ids), dtype=torch.long)}})
    def decode(self, ids, **kw):
        return ' '.join(f'tok{{i}}' for i in ids)

class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._emb = torch.nn.Embedding(1000, 8)
        self._lm  = torch.nn.Linear(8, 1000, bias=False)
    def forward(self, input_ids=None, attention_mask=None, **kw):
        h = self._emb(input_ids)
        logits = self._lm(h)
        out = MagicMock(); out.logits = logits; return out
    def to(self, *a, **kw): return self

def load_model(model_id, device='cpu'):
    return _TinyModel(), _MockTok()

def get_token_logprobs(text, model, tokenizer, device='cpu'):
    from methods import compute_token_logprobs
    return compute_token_logprobs(text, model, tokenizer, device=device)
""")

_orig_models = (SRC / 'models.py').read_text()
(SRC / 'models.py').write_text(PATCH_CODE)
print('  Patched models.py for CLI test.')

cmd = [
    sys.executable, str(ROOT / 'scripts' / 'run_03_mink.py'),
    '--input_dir',   str(DATA_DIR),
    '--output_dir',  str(OUT_DIR),
    '--models',      'fake-model',
    '--model_keys',  'fakemodel',
    '--lengths',     '64',
    '--settings',    'original', 'paraphrase',
    '--k',           '20',
    '--smoke',
    '--cache_logprobs',
]
print(f'\n  Running script ...\n')
proc = subprocess.run(cmd, capture_output=True, text=True)

(SRC / 'models.py').write_text(_orig_models)   # always restore
print('  Restored original models.py.')

print('\n  ── script stdout ──')
for line in proc.stdout.strip().splitlines():
    print(f'    {line}')
if proc.stderr.strip():
    print('\n  ── script stderr ──')
    for line in proc.stderr.strip().splitlines()[:15]:
        print(f'    {line}')

check('D1  script exited with code 0',
      proc.returncode == 0, f'returncode={proc.returncode}')

orig_csv = OUT_DIR / 'mink_scores_fakemodel_len64_original.csv'
para_csv = OUT_DIR / 'mink_scores_fakemodel_len64_paraphrase.csv'
check('D2  original CSV created', orig_csv.exists())
check('D3  paraphrase CSV created', para_csv.exists())

if orig_csv.exists():
    import pandas as pd
    df_out = pd.read_csv(orig_csv)
    required_cols = {'text_id', 'text', 'label', 'min_k_score', 'n_tokens', 'n_selected'}
    missing_cols  = required_cols - set(df_out.columns)
    check('D4  output CSV has all required columns',
          len(missing_cols) == 0, f'missing={missing_cols}')
    check('D5  row count matches input (smoke caps at 4 rows)',
          len(df_out) == len(FAKE_ROWS), f'got {len(df_out)}')
    check('D6  no NaN in min_k_score',
          df_out['min_k_score'].isna().sum() == 0)
    check('D7  n_selected == ceil(n_tokens * 0.20) per row',
          all(df_out.loc[i, 'n_selected'] ==
              max(1, math.ceil(df_out.loc[i, 'n_tokens'] * 20 / 100))
              for i in df_out.index))
    check('D8  text_id values match input',
          set(df_out['text_id'].tolist()) == {r['text_id'] for r in FAKE_ROWS},
          f'got {df_out["text_id"].tolist()}')

    print(f'\n  Output CSV (original setting):')
    print(df_out[['text_id','label','min_k_score','n_tokens','n_selected']].to_string(index=False))

check('D9  run_log.txt created', (OUT_DIR / 'run_log.txt').exists())

# D10: logprobs pkl files created (--cache_logprobs flag)
pkl_path = OUT_DIR / 'logprobs' / 'logprobs_fakemodel_len64_original.pkl'
check('D10 logprobs .pkl cache created (--cache_logprobs)',
      pkl_path.exists())

# D11: --skip_existing skips already-done files on re-run
(SRC / 'models.py').write_text(PATCH_CODE)
cmd_skip  = cmd + ['--skip_existing']
proc_skip = subprocess.run(cmd_skip, capture_output=True, text=True)
(SRC / 'models.py').write_text(_orig_models)

check('D11 --skip_existing: script exits 0 on re-run',
      proc_skip.returncode == 0)
check('D12 --skip_existing: "SKIP" appears in output',
      'SKIP' in proc_skip.stdout,
      f'"SKIP" found={("SKIP" in proc_skip.stdout)}')


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print(f'\n{"═"*60}')
passed = sum(1 for _, ok in _results if ok)
failed = sum(1 for _, ok in _results if not ok)
total  = len(_results)
print(f'  RESULT: {passed}/{total} passed    {failed} failed')
print(f'{"═"*60}')

if failed > 0:
    print('\n  Failed checks:')
    for name, ok in _results:
        if not ok:
            print(f'    ❌  {name}')
    sys.exit(1)
else:
    print('\n  All checks passed.')
    print('  ✅  methods.py and run_03_mink.py are working correctly.')
    print('  Safe to proceed with the Colab notebook run.\n')