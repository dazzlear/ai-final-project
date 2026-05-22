# Min-K% Prob — Framework Implementation

An independent implementation of the **Min-K% Prob** pretraining data detection method from:

> *Detecting Pretraining Data from Large Language Models*
> Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi Chen, Luke Zettlemoyer
> Published at **ICLR 2024** · [arXiv:2310.16789](https://arxiv.org/abs/2310.16789)

---

## What This Project Does

This repository builds the core computational framework of Min-K% Prob — a reference-free membership inference attack (MIA) method for detecting whether a piece of text was included in an LLM's pretraining data.

Given a text sample and black-box access to an LLM, the method checks: **was this text seen during pretraining?**

The system produces:
- Token log-probability scores for each input text
- Min-K% Prob scores (default k=20)
- PPL and zlib baseline scores
- AUC and TPR@5% FPR evaluation metrics
- A result table (Table 1-style)
- ROC curve figure

---

## How Min-K% Prob Works

The method is based on a simple hypothesis:

- **Unseen (non-member) text** tends to contain a few outlier tokens with very low probabilities under the LLM.
- **Seen (member) text** is less likely to contain such low-probability tokens, because the model has already learned them.

Given a sequence of tokens $x = x_1, x_2, \ldots, x_N$, the score is computed as:

$$\text{Min-K\% Prob}(x) = \frac{1}{|E|} \sum_{x_i \in \text{Min-K\%}(x)} \log p(x_i \mid x_1, \ldots, x_{i-1})$$

where Min-K%(x) is the set of the k% tokens with the **lowest** token probabilities. A higher score means the text is more likely to be a member of the pretraining data.

**No reference model or access to pretraining data is required.**

---

## Dataset

We use **WikiMIA** — the benchmark introduced in the original paper — loaded from HuggingFace:

```
swj0419/WikiMIA
```

We use the `WikiMIA_length64` split. Labels are:
- `1` = **member** (text seen during pretraining — from pre-2017 Wikipedia)
- `0` = **non-member** (text not seen during pretraining — from post-2023 Wikipedia)

The temporal gap between member and non-member data ensures ground truth accuracy: events after an LLM's training cutoff are guaranteed to be unseen.

---

## Models

We use models from the **Pythia** family (EleutherAI), chosen for accessibility and open weights:

| Model | Use |
|---|---|
| `EleutherAI/pythia-410m` | Primary (default, fastest) |
| `EleutherAI/pythia-1b` | Fallback if 410m is insufficient |
| `EleutherAI/pythia-2.8b` | Larger run, closer to paper results |

Start with the smallest model to confirm the pipeline works before scaling up.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-org>/min-k-prob.git
cd min-k-prob

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt** includes:
```
torch
transformers
datasets
scikit-learn
pandas
numpy
matplotlib
zlib  # standard library, no install needed
```

---

## Quickstart

### Step 1 — Smoke test (5–10 samples)

Confirms that the pipeline can load data, load a model, compute scores, and save output.

```bash
python src/run.py --smoke_test
```

Expected output: `outputs/smoke_test_scores.csv`

### Step 2 — Full run

```bash
python src/run.py --dataset wikimia_length64 --model EleutherAI/pythia-410m --k 20
```

Expected outputs:
```
outputs/all_scores.csv
outputs/evaluation_summary.csv
outputs/table1_results.csv
figures/roc_curve_min_k.png
```

---

## Expected Results

Based on the original paper (Table 1, Pythia-2.8B, original setting):

| Method | AUC |
|---|---|
| PPL | 0.61 |
| Zlib | 0.65 |
| Neighbor | 0.61 |
| **Min-K% Prob** | **0.67** |

Our implementation targets results in the same range using `pythia-410m` on `WikiMIA_length64`. Results may differ slightly due to model size.

---

## Team

| Member | Role | Primary Deliverables |
|---|---|---|
| Dazel | Dataset and Data Preparation | WikiMIA loading, dataset script, dataset docs |
| Hanna | Implementation | Model loading, token log-prob pipeline, Min-K% scoring |
| Jianna | Baselines and Evaluation | PPL/zlib baselines, AUC, TPR, result table, ROC curve |
| Kurt | Documentation and Final Assembly | Methodology, pseudocode, limitations, README, final report |

---

## Reference

```bibtex
@inproceedings{shi2024detecting,
  title     = {Detecting Pretraining Data from Large Language Models},
  author    = {Shi, Weijia and Ajith, Anirudh and Xia, Mengzhou and Huang, Yangsibo
               and Liu, Daogao and Blevins, Terra and Chen, Danqi and Zettlemoyer, Luke},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024}
}
```

Official project page: [swj0419.github.io/detect-pretrain.github.io](https://swj0419.github.io/detect-pretrain.github.io)
