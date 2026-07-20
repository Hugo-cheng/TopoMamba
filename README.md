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
## 📁 Data Preparation

### Benchmark Datasets

Our framework **TopoMamba** is evaluated across four distinct clinical modalities. You can download the raw datasets from their official challenges/sources below:

| Dataset | Modality | Target Organs | Link |
| :--- | :---: | :---: | :---: |
| **ACDC** | Cardiac MRI | Right/Left Ventricle, Myocardium | [ACDC Challenge]([https://www.creatis.insa-lyon.fr/Challenge/acdc/](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb)) |
| **FLARE 2022** / **Synapse** | Abdominal CT | Multi-organ (Liver, Kidney, Spleen, etc.) | [Synapse Portal]([https://www.synapse.org/#!Synapse:syn3193805](https://flare22.grand-challenge.org/Dataset/)) |
| **AMOS** | Abdominal MRI | Liver, Kidneys, Spleen | [AMOS Challenge]([https://chaos.grand-challenge.org/](https://amos22.grand-challenge.org/Instructions/)) |
| **MoNuSeg** | Microscopy | Cell Nuclei | [MoNuSeg Challenge]([https://monuseg.grand-challenge.org/](https://neurips22-cellseg.grand-challenge.org/dataset/)) |

> **Scribble Annotations**: The weak scribble supervision signals used in our experiments are synthesized/adopted following the standard protocols in weakly supervised segmentation benchmarks.

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
