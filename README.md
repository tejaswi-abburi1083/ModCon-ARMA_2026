# ModCon-ARMA: A Modularity-Aware Contrastive Framework for Biomedical Graph Clustering

Official implementation of **ModCon-ARMA**, an unsupervised graph learning framework for biomedical graph clustering that jointly optimizes:

* Jensen–Shannon Divergence (JSD)-based contrastive consistency
* Modularity-based structural regularization
* Entropy regularization
* EMA-guided self-distillation
* Monte Carlo Dropout uncertainty estimation

The framework is evaluated on multiple biomedical datasets, including:

* ADNI (Alzheimer's Disease Neuroimaging Initiative)
* BreastMNIST
* PneumoniaMNIST

---

## Overview

ModCon-ARMA learns cluster-aware graph representations without using diagnostic labels.

Key features:

* Fully unsupervised graph clustering
* ARMA graph convolution backbone
* Distribution-level contrastive learning using JSD
* Community-preserving modularity optimization
* Uncertainty-aware cluster assignment via Monte Carlo Dropout
* Applicable across multiple biomedical imaging modalities

---

## Repository Structure

```text
ModCon-ARMA_2026/
│
├── ADNI/
│   ├── CN_MCI/
│   └── CN_AD/
│
├── BreastMNIST/
│
├── PneumoniaMNIST/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dependencies

Main packages:

```text
torch
torch-geometric
numpy
pandas
scikit-learn
scipy
matplotlib
networkx
```

---

## Datasets

### ADNI

The ADNI dataset is subject to data-use agreements and cannot be redistributed through this repository.

Access:

https://adni.loni.usc.edu

### BreastMNIST

Available through:

https://medmnist.com

### PneumoniaMNIST

Available through:

https://medmnist.com

---

## Running Experiments

Example:

```bash
python ModCon_ARMA_CN_MCI.py
```

Other datasets can be executed using their corresponding scripts.

---

## Method Summary

The proposed framework consists of:

1. Graph Construction
2. Graph Augmentation

   * Feature masking
   * Edge dropout
3. ARMA Graph Encoder
4. EMA Target Encoder
5. JSD-based Contrastive Learning
6. Modularity-based Structural Optimization
7. Entropy Regularization
8. Monte Carlo Dropout Uncertainty Estimation

---

## Results

Across ADNI, BreastMNIST, and PneumoniaMNIST datasets, ModCon-ARMA consistently achieves superior clustering performance compared with classical clustering methods, graph partitioning approaches, and recent graph contrastive learning baselines.

---

## Reproducibility

All experiments were conducted using:

* Fixed random seed: 42
* 10 independent runs
* AdamW optimizer
* Cosine annealing learning-rate scheduler
* NVIDIA A4000 GPU

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{abburi2026modconarma,
  title={ModCon-ARMA: A Modularity-Aware Contrastive Framework for Biomedical Graph Clustering},
  author={Abburi, V. S. S. Tejaswi and Shigwan, Saurabh J. and Kumar, Nitin},
  journal={Scientific Reports},
  year={2026},
  note={Under Review}
}
```

---

## License

This repository is released for academic and research purposes.

---

## Contact

V. S. S. Tejaswi Abburi
Department of Computer Science and Engineering
Shiv Nadar Institution of Eminence
Delhi NCR, India

For questions regarding the implementation, please open a GitHub issue.
