import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, List


def load_model(name: str, device=None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:

    # Load a HuggingFace causal language model and its tokenizer.
    print(f"[load_model] Loading '{name}' ...")

    model = AutoModelForCausalLM.from_pretrained(
        name,
        return_dict=True,
        device_map="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(name)

    # Prevent tokenizer warnings when pad_token is not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[load_model] Done.")
    return model, tokenizer

 # Compute token-level log-probabilities for a text string.
def get_token_logprobs(
    text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> List[float]:
  
    # Safer device resolution when model may be split across multiple GPUs
    device = next(model.parameters()).device

    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids)

    # Use named attribute 
    logits    = outputs.logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Shift: for position i, the label is token i+1
    all_prob = []
    for i, token_id in enumerate(input_ids[0][1:]):
        log_p = log_probs[0, i, token_id].item()
        all_prob.append(log_p)

    return all_prob