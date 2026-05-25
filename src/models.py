import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, List


def load_model(
    name: str,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    
    print(f"[load_model] Loading '{name}' ...")

    # device_map='auto' lets HuggingFace decide GPU vs CPU automatically
    model = AutoModelForCausalLM.from_pretrained(
        name,
        return_dict=True,
        device_map='auto',
    )
    model.eval()  # disable dropout; we only need forward passes

    tokenizer = AutoTokenizer.from_pretrained(name)

    print(f"[load_model] Done.")
    return model, tokenizer


def get_token_logprobs(
    text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> List[float]:
   
    # Tokenise and move to the same device as the model
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    # outputs[1] = logits, shape (1, seq_len, vocab_size)
    logits = outputs[1]

    # Convert raw logits → log-probabilities (numerically stable)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # For each position i, extract the log-prob of the actual token at i+1.
    # input_ids[0][1:] gives the "label" tokens (shift by 1).
    all_prob = []
    for i, token_id in enumerate(input_ids[0][1:]):
        probability = log_probs[0, i, token_id].item()
        all_prob.append(probability)

    return all_prob