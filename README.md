# Low-Dose CT Image Reconstruction (AI4MIA Assignment 3)

PyTorch & MONAI implementation for **Low-Dose CT (Quarter-Dose to Full-Dose) Image Restoration** using original $512 \times 512$ Raw DICOM (`.IMA`, 3mm slice thickness, Sharp reconstruction kernel) data.

---

## 1. Overview

This repository implements an end-to-end training, validation, test evaluation, and automated report generation pipeline for low-dose CT denoising:

- **Original Raw DICOM (`.IMA`) Pipeline (`src/data.py`)**:
  - Reads raw $512 \times 512$ `.IMA` slices via MONAI `ITKReader` and applies physical attenuation windowing of **$[-100, 300]\text{ HU} \rightarrow [0, 1]$** (`ScaleIntensityRanged`).
  - Strictly separates subjects **`L192` (240 slices)** and **`L506` (211 slices)** as a held-out test set (451 slices total), splitting the remaining 8 subjects into training (1,541 slices) and validation (386 slices) sets (`seed=42`).
- **Resolution-Preserving U-Net Variants (`src/UNet/Models.py`)**:
  - Preserves $512 \times 512$ spatial resolution across all encoder/decoder stages (`kernel_size=3, stride=1, padding=1` with bilinear/nearest `Upsample + Conv2d`).
  - Supports **6 architectures** for residual learning and architectural comparison:
    - `U_Net`: Standard U-Net with global input-to-output residual connection (`out = Conv(d2) + x`).
    - `U_Net_NoRes`: Ablation baseline without global residual connection (`out = Conv(d2)`).
    - `AttU_Net`: Attention U-Net with Attention Gates (`Attention_block`) on skip connections.
    - `NestedUNet`: UNet++ with nested multi-scale skip pathways.
    - `R2U_Net`: Recurrent Residual U-Net with intra-block recurrent residual units (`RRCNN_block`, $t=2$).
    - `R2AttU_Net`: Integration of Recurrent Residual blocks and Attention Gates.
- **Composite Loss & Custom Metrics (`src/loss.py`, `src/main.py`, `src/trainer.py`)**:
  - Optimizes an FP32-stabilized composite loss: $\mathcal{L}(\hat{y}, y) = \|\hat{y} - y\|_1 + 0.1 \cdot \bigl(1 - \mathrm{SSIM}(\hat{y}, y)\bigr)$.
  - Implements custom PyTorch Ignite `PSNR` and Gaussian-window `SSIM` metrics, mixed-precision (`bfloat16`/`float16` AMP) training with unscaled gradient clipping, early stopping, and automated held-out test evaluation.

---

## 2. Repository Structure

```text
├── data/                           # Raw DICOM (.IMA) dataset (Full Dose & Quarter Dose)
├── src/
│   ├── main.py                     # CLI entry point, model/loss/optimizer setup
│   ├── data.py                     # Raw .IMA loader, [-100, 300] HU windowing & L192/L506 split
│   ├── loss.py                     # Custom Ignite PSNR & Gaussian-window SSIM metrics
│   ├── trainer.py                  # MONAI SupervisedTrainer/Evaluator & TensorBoard logging
│   ├── utils.py                    # Paired data augmentation registry (Rot90_Flip_Combo, etc.)
│   └── UNet/
│       ├── Models.py               # U_Net, U_Net_NoRes, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet
│       └── network.py              # Base network utilities & weight initialization
├── script/
│   └── run_all.sh                  # Automated Docker GPU grid search script
├── tests/
│   └── test_refactored_pipeline.py # Unit tests for models, metrics, augmentations, and trainer
├── report/
│   ├── build_report.sh             # 1-click non-intrusive evaluation & LaTeX PDF report builder
│   ├── main.tex                    # Korean academic report source (XeLaTeX)
│   ├── extract_logs_and_eval.py    # Read-only checkpoint evaluator & TensorBoard log parser
│   ├── generate_figures_and_tables.py # Generates [-100, 300] HU figures & LaTeX tables
│   └── render_code_snapshots.py    # Renders syntax-highlighted code figures
└── 2026710517_심순용_Report.pdf     # Compiled final report PDF
```

---

## 3. Usage Guide

### 3.1 Train a Single Model
Run training and automatic test evaluation (`L192`, `L506`) directly via `src.main`:

```bash
# Standard U-Net with global residual connection and 90-deg rotation + flip augmentation
python -m src.main \
  --model U_Net \
  --augment Rot90_Flip_Combo \
  --batch_size 4 \
  --epochs 100 \
  --lr 2e-5 \
  --grad_norm 1.0 \
  --patience 2 \
  --seed 42

# Attention U-Net (best performing configuration: Test PSNR 27.37 dB / SSIM 0.8379)
python -m src.main \
  --model AttU_Net \
  --augment Rot90_Flip_Combo \
  --batch_size 4 \
  --epochs 100 \
  --lr 2e-5 \
  --grad_norm 1.0 \
  --patience 2 \
  --seed 42

# Non-residual baseline U-Net (without global +x shortcut)
python -m src.main \
  --model U_Net_NoRes \
  --batch_size 4 \
  --epochs 100 \
  --lr 2e-5 \
  --grad_norm 1.0 \
  --patience 2 \
  --seed 42
```

**Supported `--model` options**: `U_Net`, `U_Net_NoRes`, `AttU_Net`, `NestedUNet`, `R2U_Net`, `R2AttU_Net`  
**Supported `--augment` options**: omitted (no augmentation), `Rot90_Flip_Combo`, `RandRot90d`, `RandAffined`, `Affine_Noise_Combo`, `Rot90_Scale_Combo`

### 3.2 Run Full Grid Search (Docker GPU)
To run the automated grid search across models and augmentations inside the `ai4mia` Docker container (automatically skips already converged runs in `log/` unless `SKIP_COMPLETED=0` is set):

```bash
bash script/run_all.sh
```

### 3.3 Run Unit Tests
Verify forward/backward gradient flow, FP32 loss stability under AMP, augmentation registry keys, and TensorBoard handlers:

```bash
.venv/bin/pytest tests/test_refactored_pipeline.py -q
```

### 3.4 Build the PDF Report
Evaluate checkpoints in read-only mode (cached in `report/cache/eval_cache.json`), generate all $[-100, 300]\text{ HU}$ comparison figures and LaTeX tables, and compile the PDF report via `xelatex`:

```bash
bash report/build_report.sh --student-id "2026710517" --name "심순용" --device cpu
```

---

## 4. Acknowledgments & Citations

### 4.1 U-Net Variants Implementation (`Unet-Segmentation-Pytorch-Nest-of-Unets`)
Adapted from [bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets](https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets) and [LeeJunHyun/Image_Segmentation](https://github.com/LeeJunHyun/Image_Segmentation), modified to preserve full spatial resolution (`padding=1`) and support global residual learning (`+ x`) for CT image restoration:

1. **U-Net**: O. Ronneberger, P. Fischer, and T. Brox, *"U-Net: Convolutional Networks for Biomedical Image Segmentation,"* MICCAI 2015. [<https://arxiv.org/abs/1505.04597>]
2. **R2U-Net**: M. Z. Alom et al., *"Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation,"* 2018. [<https://arxiv.org/abs/1802.06955>]
3. **Attention U-Net**: O. Oktay et al., *"Attention U-Net: Learning Where to Look for the Pancreas,"* MIDL 2018. [<https://arxiv.org/abs/1804.03999>]
4. **Attention R2U-Net**: Integration of R2U-Net and Attention U-Net.
5. **UNet++ (Nested U-Net)**: Z. Zhou et al., *"UNet++: A Nested U-Net Architecture for Medical Image Segmentation,"* DLMIA / ML-CDS 2018. [<https://arxiv.org/abs/1807.10165>]

```bibtex
@article{DBLP:journals/corr/abs-1906-07160,
  author    = {Malav Bateriwala and
               Pierrick Bourgeat},
  title     = {Enforcing temporal consistency in Deep Learning segmentation of brain
               {MR} images},
  journal   = {CoRR},
  volume    = {abs/1906.07160},
  year      = {2019},
  url       = {http://arxiv.org/abs/1906.07160},
  archivePrefix = {arXiv},
  eprint    = {1906.07160},
  timestamp = {Mon, 24 Jun 2019 17:28:45 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1906.07160},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

### 4.2 Low-Dose CT Reconstruction & Dataset References
- **AAPM Low-Dose CT Grand Challenge Dataset**: C. H. McCollough et al., *"Low-dose CT for the detection and classification of metastatic liver lesions: Results of the 2016 Low Dose CT Grand Challenge,"* Medical Physics, vol. 44, no. 10, pp. e339–e352, 2017.
- **Residual Encoder-Decoder for LDCT (RED-CNN)**: H. Chen et al., *"Low-Dose CT With a Residual Encoder-Decoder Convolutional Neural Network,"* IEEE Transactions on Medical Imaging, vol. 36, no. 12, pp. 2524–2535, 2017. [<https://arxiv.org/abs/1702.00288>]
- **SSIM & L1+SSIM Restoration Loss**: Z. Wang et al., *"Image quality assessment: from error visibility to structural similarity,"* IEEE TIP, 2004; H. Zhao et al., *"Loss Functions for Image Restoration With Neural Networks,"* IEEE TCI, 2017.
- **MONAI Framework**: M. J. Cardoso et al., *"MONAI: An open-source framework for deep learning in healthcare,"* arXiv:2211.02701, 2022. [<https://monai.io/>]
