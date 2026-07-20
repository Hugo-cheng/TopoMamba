# TopoMamba: Topology-Aware Mamba for Weakly Supervised Medical Image Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-PDF-blue.svg)](#) <!-- 论文录用后可替换为 arXiv / Journal 链接 -->

Official PyTorch implementation of **TopoMamba**, a topology-aware Mamba-based framework designed for weakly supervised  medical image segmentation.

---

## 💡 Highlights

- **Linear SSM Meets Topological Persistence**: First weakly supervised framework to integrate topological persistence constraints ($\mathcal{L}_{\text{PH}}$) directly into linear-time ($\mathcal{O}(N)$) State Space Models (Mamba), solving the pseudo-label noise propagation issue during selective scanning.
- **Dual-stream Spatial-Topological Mechanism (DSTM)**: Replaces classic skip connections with dynamic gating ($\mathcal{G}_{\text{topo}}$) driven by high-level semantic gradients ($\nabla\mathcal{L}_{\text{PH}}$) to eliminate boundary artifacts and preserve organ connectivity.
- **Active Topo-Consistency Training Loop**: Employs a real-time `topoPseudoFilter` pruning loop to filter out architecturally invalid teacher pseudo-labels, combined with a differential boundary consistency loss ($\mathcal{L}_{\text{bound}}$) to stabilize edges under sparse scribble signals.
- **State-of-the-Art Cross-Modality Performance**: Evaluated across 4 distinct clinical modalities (**Cardiac MRI**, **Abdominal CT**, **Abdominal MRI**, and **Microscopy images**), achieving superior segmentation and minimal Betti Errors.

---

## 🛠️ Installation

### 1. Environment Setup
Clone the repository and create a Python virtual environment:

```bash
git clone [https://github.com/Hugo-cheng/TopoMamba.git](https://github.com/Hugo-cheng/TopoMamba.git)
cd TopoMamba

# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate topomamba
