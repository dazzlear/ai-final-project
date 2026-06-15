# Min-K% Prob — Framework Implementation

An independent implementation and empirical evaluation of the **Min-K% Prob** pretraining data detection method from:

> *Detecting Pretraining Data from Large Language Models*
> Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi Chen, Luke Zettlemoyer
> Published at **ICLR 2024** · [arXiv:2310.16789](https://arxiv.org/abs/2310.16789)

This project was developed for **COSC 304 – Introduction to Artificial Intelligence**, PUP College of Computer and Information Sciences (BSCS 3-2, A.Y. 2025-2026).

---

## What This Project Does

This repository builds the core computational framework of Min-K% Prob — a reference-free membership inference attack (MIA) method for detecting whether a piece of text was included in an LLM's pretraining data.

Given a text sample and access to an LLM's token probabilities, the method checks: **was this text seen during pretraining?**

The system produces:
- Token log-probability scores for each input text
- Min-K% Prob scores (k=20)
- Five baseline scores: PPL, Zlib, Lowercase, Neighborhood Attack, and Smaller Reference Model
- ROC-AUC and TPR@5% FPR evaluation metrics across all methods
- Per-run CSV output with scores and ground truth labels
- ROC curve figures and summary result tables

---

## How Min-K% Prob Works

The method is based on a simple hypothesis:

- **Unseen (non-member) text** tends to contain a few outlier tokens with very low probabilities under the LLM.
- **Seen (member) text** is less likely to contain such low-probability tokens, because the model has already learned them.

Given a sequence of tokens $x = x_1, x_2, \ldots, x_N$, the score is computed as:

$$\text{Min-K\% Prob}(x) = \frac{1}{|\text{Min-K\%}(x)|} \sum_{x_i \in \text{Min-K\%}(x)} \log p(x_i \mid x_1, \ldots, x_{i-1})$$

where Min-K%(x) is the set of the bottom k% of tokens by log probability. A higher (less negative) score means the text is more likely to be a member of the pretraining data.

**No reference model or access to the original pretraining data is required** (the Smaller Reference Model baseline is the only exception, used for comparison purposes).

---

## Dataset

We use **WikiMIA** — the benchmark introduced in the original paper — loaded from HuggingFace:

```
swj0419/WikiMIA
```

All four token-length splits are evaluated:

| Split | Description |
|---|---|
| `WikiMIA_length32` | 32-token passages |
| `WikiMIA_length64` | 64-token passages (primary reference split, 542 examples) |
| `WikiMIA_length128` | 128-token passages |
| `WikiMIA_length256` | 256-token passages |

Labels:
- `1` = **member** (text seen during pretraining — Wikipedia events pre-2017)
- `0` = **non-member** (text not seen during pretraining — Wikipedia events post-January 2023)

The temporal gap between member and non-member data ensures ground truth accuracy: events after a model's training cutoff are guaranteed to be unseen.

Both **original** and **paraphrased** versions of all four splits are evaluated, extending the original paper's paraphrase analysis (which covered only the 64-token split) to all four lengths.

---

## Models

| Model | Parameters | Role |
|---|---|---|
| `EleutherAI/pythia-410m` | 410M | Smoke test only — verifies pipeline correctness |
| `EleutherAI/pythia-2.8b` | 2.8B | Primary evaluation target |
| `EleutherAI/gpt-neo-1.3B` | 1.3B | Primary evaluation target |
| `facebook/opt-1.3b` | 1.3B | Primary evaluation target |

Start with the smoke test model to confirm the pipeline works before running the primary evaluation models.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-org>/min-k-prob.git
cd min-k-prob

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt:**
```
torch
transformers
datasets
scikit-learn
pandas
numpy
matplotlib
tqdm
```

> Note: `zlib` is part of the Python standard library and does not need to be installed separately.

---

## Quickstart

### Step 1 — Smoke test (5–10 samples)

Confirms that the pipeline can load data, load a model, compute scores, and save output.

```bash
python src/run.py --smoke_test --model EleutherAI/pythia-410m
```

Expected output: `outputs/smoke_test_scores.csv`

### Step 2 — Full run

```bash
python src/run.py --dataset wikimia_length64 --model EleutherAI/pythia-2.8b --k 20
```

Repeat for each model (`pythia-2.8b`, `gpt-neo-1.3B`, `opt-1.3b`) and each split (`32`, `64`, `128`, `256`), for both original and paraphrased text.

Expected outputs:
```
outputs/all_scores.csv
outputs/evaluation_summary.csv
outputs/table_results.csv
figures/roc_curve_min_k.png
```

---

## Results Summary

Full results, tables, and discussion are presented in the accompanying paper (`docs/AI_FINAL_PROJECT.pdf`). At the 64-token reference split using original text, Min-K% Prob achieved the highest AUC among all six methods for Pythia-2.8B (0.6067) and GPT-Neo-1.3B (0.6114), while the Smaller Reference Model baseline slightly outperformed it on OPT-1.3B (0.5873 vs. 0.5667).

Detection performance generally improved with longer passage lengths across all models, and — notably — paraphrased text at the 256-token split produced the *strongest* detection signal observed in the study (AUC up to 0.75), reversing the expectation that paraphrasing always weakens detection.

> Note: because this study uses smaller models (1.3B–2.8B parameters) than those in the original paper (up to 66B), AUC values are expected to be lower in absolute terms. The goal of this implementation is to verify the algorithmic mechanism and relative ranking of methods, not to reproduce the original paper's exact numbers.

---

## Team

| Member | Role | Primary Deliverables |
|---|---|---|
| Argallon, Dazel | Dataset and Data Preparation | WikiMIA loading, dataset scripts, dataset documentation |
| Babasa, Maria Hanna | Implementation | Model loading, token log-prob pipeline, Min-K% Prob scoring |
| Castillo, Julianna Leila (Jianna) | Baselines and Evaluation | Baseline implementations, AUC/TPR evaluation, result tables, ROC curves |
| Borondia, Kurt Ashley | Documentation and Final Assembly | Methodology, pseudocode, limitations, README, final report |

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
