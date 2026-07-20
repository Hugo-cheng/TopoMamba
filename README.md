# TopoMamba: Topology-Aware Mamba for Weakly Supervised Medical Image Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-PDF-blue.svg)](#) <!-- 论文录用后可替换为 arXiv / Journal 链接 -->

Official PyTorch implementation of **TopoMamba**, a topology-aware Mamba-based framework designed for weakly supervised 2D and 3D medical image segmentation.

---

## 💡 Highlights

- **Mamba Backbone Integration**: Harnesses state space models (SSMs) to achieve long-range contextual modeling with linear computational complexity.
- **Topology-Aware Constraints**: Incorporates topological losses/guidance to preserve critical structural features under weak supervision.
- **Multi-Modality Support**: Capable of handling both 2D slices and 3D volumetric medical images (e.g., MRI/CT).

---

## 🛠️ Installation

### 1. Environment Setup
Clone the repository and create a Python virtual environment:

```bash
git clone [https://github.com/Hugo-cheng/TopoMamba.git](https://github.com/Hugo-cheng/TopoMamba.git)
cd TopoMamba

conda create -n topomamba python=3.10 -y
conda activate topomamba
