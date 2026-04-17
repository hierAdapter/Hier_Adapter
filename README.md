# HierAdapter: Input-Adaptive Hierarchical Laplacian Residual Fusion for Parameter-Efficient Fine-Tuning in Code Intelligence

<p align="center">
  <img src="figures/main.png" alt="HierAdapter Architecture" width="85%"/>
  <br/>
  <em>Fig. 2 — Working procedure of the HierAdapter framework: an input-adaptive Laplacian residual fusion approach across hierarchical transformer layers.</em>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#repository-structure">Repository Structure</a> •
  <a href="#datasets">Datasets</a> •
  <a href="#installation">Installation</a> •
  <a href="#experimental-configurations">Experiments</a> •
  <a href="#inference--analysis">Inference</a> •
  <a href="#ablation-study">Ablation</a> •
  <a href="#citation">Citation</a>
</p>

> **Submission Note (TSE — Double-Blind Review):** This repository is prepared in compliance with IEEE Transactions on Software Engineering (TSE) reproducibility and anonymity requirements. Author identities are fully anonymized. All code, data, and experimental configurations are made publicly accessible to support full replication of reported results. No personally identifiable information appears in any file or commit history.

---

## Overview

**HierAdapter** is a novel Parameter-Efficient Fine-Tuning (PEFT) framework designed for code intelligence tasks. Unlike conventional adapters that operate at a single transformer layer, HierAdapter introduces a **hierarchical, input-adaptive architecture** that:

- Groups transformer layers into structured pools (Pool₀–Pool₃) capturing representations at different semantic depths.
- Applies a **Laplacian Residual Kernel (Δ)** at each hierarchical level to compute layer-wise difference signals, preserving fine-grained syntactic and semantic variation in source code.
- Fuses multi-level representations through a learned **α-Fusion Gate** combining base class embedding with hierarchical adapter outputs.
- Maintains a **frozen backbone** (no gradient updates to pre-trained weights), updating only low-rank adapter matrices (LRA, Rank 6).

This design is validated primarily on **code vulnerability detection** — a safety-critical software engineering task — and extended to **code clone detection**, with evaluation across multiple Code Pre-trained Language Models (CodePTMs) and token length settings.

<p align="center">
  <img src="figures/fig3_stochastic.png" alt="Stochastic Variation Analysis" width="80%"/>
  <br/>
  <em>Fig. 3 — Multidimensional analysis of stochastic variations across identical training configurations. (A) Hierarchical representation of Laplacian signal strength and seed-wise Laplacian norm distributions on the test set; (B) Empirical seed-wise convergence trajectories.</em>
</p>

---

## Repository Structure

```
hieradapter/
├── Clone det with Codt5.ipynb        # Code clone detection — CodeT5+ 770M backbone
├── HierAdapter_all.ipynb             # HierAdapter: all model × token configurations (main experiments)
├── Inferences.ipynb                  # Inference & out-of-domain / adversarial analysis
├── Statistics of Dataset.ipynb       # Dataset statistical overview & preprocessing
├── Token length 512.ipynb            # All PEFT baselines — 512-token setting
├── Token length 1024.ipynb           # All PEFT baselines — 1024-token setting
└── figures/
    ├── fig2_architecture.png         # HierAdapter architecture diagram
    └── fig3_stochastic.png           # Stochastic variation analysis
```

> **Quick Navigation:** Use the table below to jump directly to any experiment of interest.

| Notebook | Task | What it covers |
|---|---|---|
| [`Clone det with Codt5.ipynb`](#code-clone-detection) | Code Clone Detection | GateRA, Prefix Tuning, Adapter (Clone Det), LoRA, BitFit, TS-PEFT |
| [`HierAdapter_all.ipynb`](#hireadapter-main-experiments) | Code Vulnerability Detection | HierAdapter across all CodePTMs × token lengths + Ablation |
| [`Inferences.ipynb`](#inference--analysis) | Out-of-Domain & Adversarial Inference | SOTA PEFT + HierAdapter on OOD and adversarial datasets |
| [`Statistics of Dataset.ipynb`](#dataset-statistics) | Dataset Analysis | Statistical overview, preprocessing of all datasets |
| [`Token length 512.ipynb`](#token-length-experiments) | Code Vulnerability Detection | All SOTA PEFT methods — 512 token setting |
| [`Token length 1024.ipynb`](#token-length-experiments) | Code Vulnerability Detection | All SOTA PEFT methods — 1024 token setting |

---

## Datasets

All datasets are publicly hosted on Hugging Face and can be accessed directly via the links below. Due to large file sizes, datasets are **not** included in this repository.

**🔗 Hugging Face Organization:** [https://huggingface.co/hieradapter](https://huggingface.co/hieradapter)

| Dataset | Description | Link | Access |
|---|---|---|---|
| `hieradapter/Processed_CodeXGLUE` | Preprocessed CodeXGLUE vulnerability dataset (in-domain training/eval split) — 16.2k samples | [🔗 View](https://huggingface.co/datasets/hieradapter/Processed_CodeXGLUE) | Public |
| `hieradapter/OOD_datasets` | Out-of-domain evaluation dataset for generalization analysis — 5.28k samples | [🔗 View](https://huggingface.co/datasets/hieradapter/OOD_datasets) | Public |
| `hieradapter/Adverserial_Dataset` | Adversarial vulnerability samples for robustness evaluation | [🔗 View](https://huggingface.co/datasets/hieradapter/Adverserial_Dataset) | Preview |
| `hieradapter/Code_clone` | Code clone detection dataset — 24k samples | [🔗 View](https://huggingface.co/datasets/hieradapter/Code_clone) | Public |

### Loading Datasets via Code

```python
from datasets import load_dataset

# In-domain vulnerability detection (CodeXGLUE)
dataset = load_dataset("hieradapter/Processed_CodeXGLUE")

# Out-of-domain evaluation
ood_dataset = load_dataset("hieradapter/OOD_datasets")

# Adversarial robustness evaluation
adv_dataset = load_dataset("hieradapter/Adverserial_Dataset")

# Code clone detection
clone_dataset = load_dataset("hieradapter/Code_clone")
```

---

## Installation

### Requirements

```bash
# Clone this repository
git clone https://github.com/hieradapter/hieradapter.git
cd hieradapter

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Core Dependencies

```bash
pip install torch transformers datasets peft \
            scikit-learn numpy pandas matplotlib \
            jupyter notebook accelerate
```

> **Environment:** All experiments were conducted using Python 3.9+, PyTorch ≥ 2.0, and Hugging Face Transformers ≥ 4.35. GPU with ≥ 16 GB VRAM is recommended for 770M and 3B parameter models.

---

## Experimental Configurations

### Code Clone Detection

**📓 Notebook:** [`Clone det with Codt5.ipynb`](./Clone%20det%20with%20Codt5.ipynb)

Evaluates the following PEFT strategies on the code clone detection task using **CodeT5+ 770M** as the backbone:

| Method | Description |
|---|---|
| GateRA | Gated residual adapter fine-tuning |
| Prefix Tuning (CodeT5+) | Soft prompt prefix injection |
| Adapter Clone Det | Standard bottleneck adapter |
| LoRA | Low-rank weight decomposition |
| BitFit | Bias-only fine-tuning |
| TS-PEFT | Task-specific PEFT variant |

```bash
# Launch the notebook
jupyter notebook "Clone det with Codt5.ipynb"
```

---

### HierAdapter Main Experiments

**📓 Notebook:** [`HierAdapter_all.ipynb`](./HierAdapter_all.ipynb)

Covers all HierAdapter configurations for the **code vulnerability detection** task across models and token lengths:

| Configuration | Model | Token Length |
|---|---|---|
| HierAdapter | CodeT5+ 220M | 512 |
| HierAdapter | CodeT5+ 220M | 1024 |
| HierAdapter | CodeT5+ 770M | 512 |
| HierAdapter | CodeT5+ 770M | 1024 |
| HierAdapter | Qwen2.5-Coder-3B | 512 |
| HierAdapter | Qwen2.5-Coder-3B | 1024 |
| HierAdapter | Qwen2.5-Coder-1.5B | 512 |
| HierAdapter | Qwen2.5-Coder-1.5B | 1024 |
| HierAdapter | UniXcoder (Frozen) | — |
| Ablation Study | Multiple | Multiple |

```bash
jupyter notebook HierAdapter_all.ipynb
```

---

### Token Length Experiments

**📓 Notebooks:** [`Token length 512.ipynb`](./Token%20length%20512.ipynb) · [`Token length 1024.ipynb`](./Token%20length%201024.ipynb)

Comprehensive PEFT baseline experiments across both 512- and 1024-token settings. All major SOTA PEFT methods are evaluated:

**PEFT Methods Covered:**

| Method | 512 Token | 1024 Token | Models |
|---|:---:|:---:|---|
| HierAdapter (Ours) | ✅ | ✅ | CodeT5+ 220M/770M, Qwen 1.5B/3B |
| LoRA | ✅ | ✅ | CodeT5+ 220M/770M, Qwen 1.5B/3B |
| BitFit | ✅ | ✅ | CodeT5+ 220M/770M, Qwen 1.5B/3B |
| Prefix Tuning | ✅ | ✅ | CodeT5+ 220M/770M, Qwen 2.5-Coder-3B |
| Adapter (Standard) | ✅ | ✅ | CodeT5+ 220M/770M, Qwen 1.5B/3B |

```bash
# 512-token setting
jupyter notebook "Token length 512.ipynb"

# 1024-token setting
jupyter notebook "Token length 1024.ipynb"
```

---

### Dataset Statistics

**📓 Notebook:** [`Statistics of Dataset.ipynb`](./Statistics%20of%20Dataset.ipynb)

Provides a complete statistical overview of all datasets used in this study, including:

- Sample distributions across train/validation/test splits
- Vulnerability label distributions (vulnerable vs. non-vulnerable)
- Token length distributions at both 512 and 1024 settings
- CodeXGLUE dataset statistics with simple preprocessing view

```bash
jupyter notebook "Statistics of Dataset.ipynb"
```

---

## Inference & Analysis

**📓 Notebook:** [`Inferences.ipynb`](./Inferences.ipynb)

Covers all inference and generalization analysis experiments:

| Experiment | Backbone | Dataset |
|---|---|---|
| Out-of-domain analysis — SOTA PEFT | CodeT5+ 770M (Frozen), fine-tuned on CodeXGLUE | OOD dataset |
| Out-of-domain analysis — SOTA PEFT | CodeT5+ 220M (Frozen), fine-tuned on CodeXGLUE | OOD dataset |
| Adversarial robustness — SOTA PEFT | CodeT5+ 220M (Frozen), fine-tuned on CodeXGLUE | Adversarial dataset |
| HierAdapter inference — adversarial | UniXcoder | `adversarial_vuln` dataset |
| HierAdapter inference — in-domain | UniXcoder | In-domain-distribution dataset |

```bash
jupyter notebook Inferences.ipynb
```

---

## Ablation Study

The ablation study is contained within [`HierAdapter_all.ipynb`](./HierAdapter_all.ipynb) as a dedicated section. It systematically evaluates the contribution of each core HierAdapter component:

- Effect of hierarchical pooling depth (Pool₀ through Pool₃)
- Impact of the Laplacian Residual Kernel (Δ)
- Role of the α-Fusion Gate
- Rank sensitivity of the Low-Rank Adapter (LRA)
- Frozen vs. partially unfrozen backbone configurations

---

## Reproducibility

All experiments are fully reproducible. To replicate any result reported in the paper:

1. Install dependencies as described in [Installation](#installation).
2. Download the required dataset from [Hugging Face](https://huggingface.co/hieradapter) using the provided code snippets.
3. Open the corresponding notebook (see [Repository Structure](#repository-structure)).
4. Follow the section headers within each notebook — each section maps directly to a configuration reported in the paper.
5. Random seeds are fixed inside each notebook cell to ensure deterministic results.

> **TSE Reproducibility Compliance:** This repository satisfies the IEEE TSE artifact availability requirement. All training configurations, dataset splits, and evaluation scripts are included. No proprietary data or closed-source tools are required.

---

## Citation

> *Author information is anonymized for double-blind peer review in compliance with IEEE TSE submission guidelines. This section will be updated upon paper acceptance.*



---

## License

This project is released under the [MIT License](./LICENSE). Datasets hosted on Hugging Face are released under their respective licenses as described on each dataset page.

---

<p align="center">
  <strong>🔗 Hugging Face Datasets:</strong>
  <a href="https://huggingface.co/datasets/hieradapter/Processed_CodeXGLUE">CodeXGLUE</a> •
  <a href="https://huggingface.co/datasets/hieradapter/OOD_datasets">OOD</a> •
  <a href="https://huggingface.co/datasets/hieradapter/Adverserial_Dataset">Adversarial</a> •
  <a href="https://huggingface.co/datasets/hieradapter/Code_clone">Code Clone</a>
</p>
