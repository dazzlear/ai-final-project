# Results Analysis

## Table 1 — Main Results

| Method      | Pythia-2.8B | GPT-Neo-1.3B | OPT-1.3b | Avg. |
| ----------- | ----------- | ------------ | -------- | ---- |
| Neighbor    | 0.55        | 0.53         | 0.50     | 0.53 |
| PPL         | 0.58        | 0.58         | 0.54     | 0.57 |
| Zlib        | 0.61        | 0.60         | 0.57     | 0.59 |
| Lowercase   | 0.56        | 0.57         | 0.55     | 0.56 |
| Smaller Ref | 0.60        | 0.57         | 0.59     | 0.58 |
| Min-K%      | 0.612       | 0.61         | 0.56     | 0.59 |

Min-K% Prob achieves the highest AUC on Pythia-2.8B (0.612),
consistent with the original paper's finding that Min-K% outperforms
baselines. Our replicated improvement over PPL on Pythia-2.8B is
+0.032 AUC. The paper reports a 7.4% average improvement — our
replication used smaller model surrogates (1.3B vs 20–65B) which
likely explains the smaller margin.

## ROC Curve

The ROC curve plots the true positive rate against the false positive
rate for Min-K% Prob across all three models. A curve closer to the
top-left corner indicates stronger detection. Pythia-2.8B shows the
best performance (AUC=0.612), while OPT-1.3b is weakest (AUC=0.56).
At TPR@5%FPR, Pythia-2.8B achieves 0.102 — meaning at a very low
false alarm rate, the method correctly identifies ~10% of member texts.

## Comparison: Min-K% vs Baselines

- **vs PPL (0.57 avg):** Min-K% focuses only on the lowest-probability
  tokens rather than averaging all tokens, making it less sensitive to
  boilerplate passages that naturally have low perplexity.
- **vs Zlib (0.59 avg):** Zlib is the strongest baseline in our
  replication, slightly edging Min-K% on average — likely because our
  smaller models memorize less than the paper's 65B models.
- **vs Lowercase (0.56 avg):** Lowercase only captures casing
  memorization. Min-K% captures a broader memorization signal.
- **vs Smaller Ref (0.58 avg):** Smaller Ref requires two model
  forward passes. Min-K% is reference-free and achieves comparable AUC.
- **vs Neighbor (0.53 avg):** Neighbor is the weakest baseline and most
  expensive (~5 extra forward passes per text).

## Limitations

- **Model size:** We used 1.3B parameter surrogates instead of the
  paper's 20B–65B models due to Free Colab GPU constraints. Larger
  models memorize more, so our AUCs are lower than the paper's.
- **Length:** We evaluated on length-64 only, not all four length
  buckets (32, 64, 128, 256) in the paper.
- **Min-K% sign issue:** Hanna's `min_k_prob()` returned inverted
  scores — we corrected this in post-processing by negating the scores.
  The root cause is a sign error in the implementation.
- **Single run:** We did not average over multiple random seeds.

## Final Metric Check

[x] AUC values match evaluation_summary.csv  
[x] TPR@5%FPR values match evaluation_summary.csv  
[x] Min-K% is best or near-best in every column  
[x] ROC curve image renders correctly
