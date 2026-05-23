from src.baselines import ppl_score, zlib_score, lowercase_score
from src.metrics import compute_auc, tpr_at_fpr

# --- mock get_token_logprobs so we don't need the real model ---
def mock_logprob_fn(text):
    # returns fake log-probs proportional to text length
    return [-0.5] * len(text.split())

# test ppl_score
assert ppl_score([-1.0, -2.0, -3.0]) == -2.0
print("ppl_score OK")

# test zlib_score
s = zlib_score([-1.0, -2.0], "the cat sat on the mat")
assert isinstance(s, float)
print("zlib_score OK")

# test lowercase_score
s = lowercase_score("The Cat Sat", mock_logprob_fn)
assert isinstance(s, float)
print("lowercase_score OK")

# test metrics
assert compute_auc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0
assert compute_auc([0.1, 0.2, 0.9, 0.8], [1, 1, 0, 0]) == 0.0
print("compute_auc OK")

t = tpr_at_fpr([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0], fpr=0.05)
assert isinstance(t, float)
print("tpr_at_fpr OK")

print("\nAll Friday checks passed — ready for 9 PM sync with Hanna.")