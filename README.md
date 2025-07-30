# WISE-MAE

## Wavelet-Informed Sampling for Encoding  
**Learning from the Right Patches for Histopathology Representation Learning**

[[`Paper`](https://doi.org/xx.xxxx/wise-mae)] 


<p align="center">
  <img src="./images/banner_placeholder.png" width="800"/>
</p>

---



---

## 🧠 Abstract

Self-supervised learning (SSL) holds promise for scalable histopathology modeling, yet struggles to efficiently learn from highly redundant WSIs. We introduce **WISE-MAE**, a two-stage learning framework that improves representation learning from histopathology by identifying the *right patches* using a wavelet-informed sampling scheme. Stage one uses wavelet-based energy maps to guide the patch selection process, focusing on visually and texturally rich regions at 40× magnification. Stage two employs a masked autoencoder (MAE) to pretrain a ViT encoder, which is later frozen and used for CLAM-based classification. WISE-MAE demonstrates improved accuracy and generalization on three datasets (TCGA-NSCLC, TCGA-RCC, and CAMELYON16), surpassing standard MAE and GCMAE, especially under transfer settings and frozen evaluation.

---

## 📦 Installation

This repo builds on the [MAE](https://github.com/facebookresearch/mae) and [CLAM](https://github.com/mahmoodlab/CLAM) pipelines. Use the following steps:

```bash
git clone https://github.com/your-repo/wise-mae.git
cd wise-mae
pip install -r requirements.txt
```

---

## 🛠️ Usage

### 1. Pretrain WISE-MAE (Masked Autoencoding)

```bash
python pretrain_mae.py --config configs/mae_pretrain.yaml
```

### 2. Extract Features with Frozen Encoder

```bash
python extract_features.py --encoder_ckpt checkpoints/mae_encoder.pth
```

### 3. Run CLAM Classification

```bash
python train_clam.py --config configs/clam_config.yaml
```

---

## 🧪 Results Summary

All results use frozen ViT encoders passed to CLAM. Evaluation is performed on TCGA-NSCLC (lung), TCGA-RCC (renal), and CAMELYON16 (metastasis detection).

| Model                | Acc (NSCLC) | AUC (NSCLC) | F1 (NSCLC) | Acc (RCC) | AUC (RCC) | F1 (RCC) | Acc (CAM16) | AUC (CAM16) | F1 (CAM16) |
|----------------------|-------------|-------------|------------|-----------|-----------|----------|--------------|--------------|-------------|
| MAE                  | 0.867       | 0.941       | 0.859      | 0.899     | 0.971     | 0.879    | 0.874        | 0.912        | 0.842       |
| GCMAE                | 0.862       | 0.937       | 0.850      | 0.891     | 0.969     | 0.870    | 0.882        | 0.922        | 0.854       |
| WISE-MAE             | 0.868       | 0.944       | 0.860      | 0.903     | 0.973     | 0.885    | 0.894        | 0.935        | 0.866       |
| WISE-MAE + Contrast  | **0.869**   | **0.945**   | **0.862**  | **0.906** | **0.974** | **0.887**| **0.901**    | **0.943**    | **0.873**   |

---

## 🌐 Dataset Preparation

- **TCGA-NSCLC**: LUAD vs LUSC  
- **TCGA-RCC**: KIRC vs KIRP vs KICH  
- **CAMELYON16**: Tumor vs Normal  

Patch extraction uses 224×224 crops at 40× magnification. Coordinates are selected using 5× wavelet maps (db4) filtered by energy thresholds.

---

## 🧬 Method Overview

<p align="center">
  <img src="./images/workflow_placeholder.png" width="800"/>
</p>

1. **Wavelet-based Selection**: Select top-N patches using energy from db4 decomposition.
2. **Patch Extraction**: Extract 224×224 tiles at 40× magnification.
3. **MAE Pretraining**: Masked autoencoder with 75% masking and ViT-base encoder.
4. **Frozen Feature Encoding**: Encoder is frozen; features are stored for downstream tasks.
5. **CLAM Evaluation**: Weakly-supervised attention MIL model is trained for slide-level classification.

---

## 📁 Folder Structure

```bash
wise-mae/
├── mae/                # MAE model and pretraining code
├── patch_selection/    # Wavelet-guided patch sampling
├── clam_eval/          # CLAM classification framework
├── pt_files/           # Pre-extracted features (.pt)
├── h5_files/           # HDF5 coordinate files
├── configs/            # All YAML configs for training
├── results/            # Saved evaluation metrics
├── figures/            # Paper figures and visualizations
└── README.md
```

---


---

## 📬 Contact

For questions, issues, or collaboration proposals, please open an issue on GitHub.
