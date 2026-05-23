# -- scores with checkpointing every 50 texts --
checkpoint_path = SCORE_DIR / f"checkpoint_{short}.pkl"

# load existing checkpoint if run was interrupted
if checkpoint_path.exists():
    print("  Found checkpoint — resuming...")
    checkpoint = pickle.load(open(checkpoint_path, "rb"))
    scores = checkpoint["scores"]
    start_idx = checkpoint["last_idx"] + 1
    print(f"  Resuming from index {start_idx}")
else:
    scores = {
        "PPL": [], "Zlib": [], "Min-K%": [],
        "Lowercase": [], "Neighbor": [], "Smaller Ref": []
    }
    start_idx = 0

model, tok = load_model(target_name)
fn = lambda t: get_token_logprobs(model, tok, t)

for i, (lp, text) in enumerate(zip(target_lps[start_idx:], texts[start_idx:]), start=start_idx):
    scores["PPL"].append(ppl_score(lp))
    scores["Zlib"].append(zlib_score(lp, text))
    scores["Min-K%"].append(min_k_prob(lp, k=20))
    scores["Lowercase"].append(lowercase_score(text, fn))
    scores["Neighbor"].append(neighbor_score(text, fn, n_neighbors=5))

    # checkpoint every 50 texts
    if (i + 1) % 50 == 0:
        pickle.dump({"scores": scores, "last_idx": i}, open(checkpoint_path, "wb"))
        print(f"  Checkpoint saved at index {i+1}/{ len(texts)}")

del model; torch.cuda.empty_cache()

# smaller ref scores with same checkpointing
ref_model, ref_tok = load_model(REF_MODELS[target_name])
for i, (lp_t, text) in enumerate(zip(target_lps[start_idx:], texts[start_idx:]), start=start_idx):
    lp_r = get_token_logprobs(ref_model, ref_tok, text)
    scores["Smaller Ref"].append(smaller_ref_score(lp_t, lp_r))

    if (i + 1) % 50 == 0:
        pickle.dump({"scores": scores, "last_idx": i}, open(checkpoint_path, "wb"))
        print(f"  Ref checkpoint saved at index {i+1}/{len(texts)}")

del ref_model; torch.cuda.empty_cache()

# delete checkpoint once fully complete
checkpoint_path.unlink(missing_ok=True)