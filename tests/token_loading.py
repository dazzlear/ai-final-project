# Run this from the repository root
from src.models import load_model, get_token_logprobs
from src.methods import min_k_prob

model, tokenizer = load_model("EleutherAI/pythia-410m")

text = "The 15th Miss Universe Thailand pageant was held at Royal Paragon Hall."
lp = get_token_logprobs(text, model, tokenizer)
print("Log-probs:", [round(x, 3) for x in lp])   # all negative floats

score = min_k_prob(lp, k=20)
print("Min-K% score:", round(score, 4))            # positive float (negated mean)
