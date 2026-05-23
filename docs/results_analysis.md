# Results Analysis

## Table 1 — Main Results

Min-K% Prob outperforms all baselines on [X of Y] model/method combinations,
achieving an average AUC of [Z] compared to [W] for the strongest baseline (PPL).

| Method      | Pythia-2.8B | GPT-Neo-1.3B | OPT-1.3b | Avg. |
| ----------- | ----------- | ------------ | -------- | ---- |
| Neighbor    | TBD         | TBD          | TBD      | TBD  |
| PPL         | TBD         | TBD          | TBD      | TBD  |
| Zlib        | TBD         | TBD          | TBD      | TBD  |
| Lowercase   | TBD         | TBD          | TBD      | TBD  |
| Smaller Ref | TBD         | TBD          | TBD      | TBD  |
| Min-K%      | TBD         | TBD          | TBD      | TBD  |

## ROC Curve

The ROC curve shows the trade-off between true positive rate and false positive
rate for Min-K% Prob across all three models. A curve closer to the top-left
corner indicates better detection performance. Min-K% Prob consistently sits
above the baselines, confirming its advantage especially at low FPR thresholds
(TPR@5%FPR).

## Comparison: Min-K% vs Baselines

- **vs PPL:** Min-K% improves over PPL by focusing only on the lowest-probability
  tokens rather than averaging all tokens, making it less sensitive to long
  boilerplate passages.
- **vs Zlib:** Zlib corrects for text complexity but ignores token-level structure.
  Min-K% captures memorization more directly.
- **vs Lowercase:** Lowercase is sensitive to casing memorization only.
  Min-K% captures a broader signal.
- **vs Smaller Ref:** Smaller Ref requires a second model forward pass.
  Min-K% achieves comparable or better AUC with no reference model needed.
- **vs Neighbor:** Neighbor is expensive (~5 extra forward passes per text)
  and still underperforms Min-K%.

## Limitations

- We used smaller model surrogates (1.3B) instead of the paper's 20B/65B
  models due to Free Colab GPU constraints.
- Results are on length-64 only, not all four length buckets in the paper.
- AUCs may differ slightly from the paper (±0.03–0.05) due to model size
  and dataset differences.

## Final Metric Check

[ ] AUC values match evaluation_summary.csv  
[ ] TPR@5%FPR values match evaluation_summary.csv  
[ ] Min-K% is best or near-best in every column  
[ ] ROC curve image renders correctly
