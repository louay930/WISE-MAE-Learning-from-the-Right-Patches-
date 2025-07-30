# WISE-MAE-Learning-from-the-Right-Patches-

# Wavelet-MAE: Learning from the Right Patches for Histopathology Representation Learning

![Banner](images/banner_placeholder.png)
<!-- Replace with your general image: images/banner.png -->

This repository provides the official implementation of **Wavelet-MAE**, a two-stage self-supervised learning framework for Whole Slide Image (WSI) representation learning. It integrates a wavelet-driven patch selection strategy and a ViT-based masked autoencoder to learn efficient slide-level features, which are evaluated using the CLAM weakly supervised classification pipeline.

---

## 🚀 Highlights

- **Wavelet Patch Selection**: Uses 5× resolution to locate tissue-rich regions, extracts 40× patches.
- **MAE Pretraining**: Vision Transformer trained to reconstruct masked patches.
- **Feature Reuse**: The encoder is frozen after pretraining; features are passed to a CLAM classifier.
- **Strong Results**: Outperforms standard MAE and GCMAE on three datasets under frozen evaluation.

---

## 🗂 Repository Overview

.
├── mae/ # Masked autoencoder code
├── patch_selection/ # Wavelet-based patch sampling
├── clam_eval/ # CLAM evaluation code
├── pt_files/ # Extracted features
├── h5_files/ # Patch coordinate files
├── configs/ # Config files for training/eval
├── figures/ # Visual assets
├── results/ # Evaluation results
└── README.md

yaml
Copy
Edit

---

## 📊 Results Summary

| Model               | Acc (NSCLC) | AUC (NSCLC) | F1 (NSCLC) | Acc (RCC) | AUC (RCC) | F1 (RCC) | Acc (CAM16) | AUC (CAM16) | F1 (CAM16) |
|---------------------|-------------|-------------|------------|-----------|-----------|----------|--------------|--------------|-------------|
| MAE                 | 0.867       | 0.941       | 0.859      | 0.899     | 0.971     | 0.879    | 0.874        | 0.912        | 0.842       |
| GCMAE               | 0.862       | 0.937       | 0.850      | 0.891     | 0.969     | 0.870    | 0.882        | 0.922        | 0.854       |
| WISE-MAE            | 0.868       | 0.944       | 0.860      | 0.903     | 0.973     | 0.885    | 0.894        | 0.935        | 0.866       |
| WISE-MAE + Contrast | 0.869       | 0.945       | 0.862      | 0.906     | 0.974     | 0.887    | **0.901**    | **0.943**    | **0.873**   |

> All results use CLAM with frozen encoder. Full experiments available in the `results/` folder.

---

## 🔬 Method Overview

![Workflow](images/workflow_placeholder.png)
<!-- Replace with your actual workflow image -->

1. **Wavelet-based Selection** (5×): Choose top-N tissue-rich coordinates.
2. **Patch Extraction** (40×): Extract 224×224 patches.
3. **Masked Autoencoder Pretraining**: 75% masking, ViT-base.
4. **Feature Extraction**: Encoder frozen, used to embed patches.
5. **CLAM Evaluation**: MIL-based slide classification.

---

## 🛠️ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
2. Pretrain MAE
bash
Copy
Edit
python pretrain_mae.py --config configs/mae_pretrain.yaml
3. Extract Features
bash
Copy
Edit
python extract_features.py --encoder_ckpt checkpoints/mae_encoder.pth
4. Run CLAM Classification
bash
Copy
Edit
python train_clam.py --config configs/clam_config.yaml
📁 Datasets
TCGA-NSCLC: LUAD vs LUSC

TCGA-RCC: KIRC vs KIRP vs KICH

CAMELYON16: Tumor vs Normal

All slides were patchified using 224×224 tiles at 40× resolution. Coordinates extracted using 5× wavelet maps.

📚 Citation
If you use this work, please cite:

bibtex
Copy
Edit
@inproceedings{frikha2025waveletmae,
  title     = {Learning from the Right Patches: A Two-Stage Wavelet-Driven Masked Autoencoder for Histopathology},
  author    = {Frikha, Firas and Others},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2025}
}
📬 Contact
For questions or collaborations:
