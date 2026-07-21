# TopoMamba: Topology-Aware Mamba for Weakly Supervised Medical Image Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-PDF-blue.svg)](#) <!-- 论文录用后可替换为 arXiv / Journal 链接 -->

Official PyTorch implementation of **TopoMamba**, a topology-aware Mamba-based framework designed for weakly supervised  medical image segmentation.



https://github.com/user-attachments/assets/366b8128-50f1-42f8-9299-124300a58887



---





## 💡 Highlights

* **Linear SSM Meets Topological Persistence:** First weakly supervised framework to integrate topological persistence constraints (L<sub>PH</sub>) directly into linear-time (*O*(*N*)) State Space Models (Mamba), solving the pseudo-label noise propagation issue during selective scanning.
* **Dual-stream Spatial-Topological Mechanism (DSTM):** Replaces classic skip connections with dynamic gating (*G*<sub>topo</sub>) driven by high-level semantic gradients (∇L<sub>PH</sub>) to eliminate boundary artifacts and preserve organ connectivity.
* **Active Topo-Consistency Training Loop:** Employs a real-time `topoPseudoFilter` pruning loop to filter out architecturally invalid teacher pseudo-labels, combined with a differential boundary consistency loss (L<sub>bound</sub>) to stabilize edges under sparse scribble signals.
* **State-of-the-Art Cross-Modality Performance:** Evaluated across 4 distinct clinical modalities (Cardiac MRI, Abdominal CT, Abdominal MRI, and Microscopy images), achieving superior segmentation and minimal Betti Errors.

---
## 📁 Data Preparation

### Benchmark Datasets

Our framework **TopoMamba** is evaluated across four distinct clinical/biological modalities under sparse scribble supervision. You can download the raw datasets and their official challenges from the links below:

| Dataset | Modality | Target Structures / Organs | Official Website (Copy & Paste) |
| :--- | :---: | :--- | :--- |
| **ACDC** | Cine-MRI | Right Ventricle (RV), Left Ventricle (LV), Myocardium (MYO) | `https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb` |
| **Abdomen CT (FLARE)** | Abdominal CT | 13 Abdominal Organs (Liver, Spleen, Pancreas, Kidneys, Stomach, etc.) | `https://flare22.grand-challenge.org/` |
| **Abdomen MRI (AMOS)** | Abdominal MRI | 13 Abdominal Organs (Cross-modality evaluation) | `https://amos22.grand-challenge.org/` |
| **Cell Segmentation** | Microscopy | Multi-modal Cell Nuclei & Instances | `https://neurips22-cellseg.grand-challenge.org/` |

> **Scribble Annotations**: The weak scribble supervision signals used in our experiments are synthesized/adopted following the standard  pipeline from [WSL4MIS ](https://github.com/HiLab-git/WSL4MIS/blob/main/code/scribbles_generator.py).

## 🛠️ Usage

### Environment Setup
Clone the repository and create a Python virtual environment:

```bash
git clone [https://github.com/Hugo-cheng/TopoMamba.git](https://github.com/Hugo-cheng/TopoMamba.git)
cd TopoMamba

# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate topomamba
```
## 🚀 Quick Start Training
Run the standard training command with pre-trained weights on the ACDC dataset:
```
python train_acdc.py \
    --pretrain_weights "best_model.pth" \
    --batch_size 8 \
    --max_epochs 200 \
    --root_path "./data/acdc/train"
```

## 📊 Evaluate TopoMamba
Run testing on the validation/test set using the trained TopoMamba checkpoint:
```
python test_acdc.py \
    --pth_path "best_model.pth" \
    --root_path "./data/acdc/val"
```

## 🔍 TopoMamba Feature Map Visualization
Generate Class Activation Maps (CAM) for the proposed TopoMamba architecture:
```
python visual_cam.py \
    --pth_path "best_model.pth" \
    --root_path "./data/acdc/val" \
    --model_type "BiS_Mamba"
```
