# 🖐️ Fake Palm Image Generation using Transformers (Phase 1)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPOSITORY/blob/main/Fake_Palm_Generation_Pipeline.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Colab T4](https://img.shields.io/badge/Google%20Colab-T4%20GPU%20Ready-orange.svg)](https://colab.research.google.com/)

An end-to-end transformer-based generative deep learning pipeline for synthesizing photorealistic fake contactless palmprint images to evaluate biometric presentation attack detection (PAD) systems and privacy preservation.

---

## 📌 Overview

This repository hosts the **self-contained research pipeline** for **Phase 1** of synthetic palm biometric research:
* **Input**: Genuine contactless palm image + corresponding palm segmentation mask (e.g., from IIT Delhi Touchless Palmprint Database).
* **Generator**: Dual-conditioned **Swin Transformer** with Biometric Texture Modulation (BTM) and multiscale skip decoding.
* **Discriminator**: Multiscale **PatchGAN** discriminator enforcing high-frequency dermal ridge realism.
* **Loss Functions**: Multi-objective loss combining Adversarial loss, L1 pixel reconstruction, VGG perceptual loss, and Sobel/Laplacian ridge minutiae loss.
* **Output**: Photorealistic synthetic fake palm images with plausible line/ridge modifications that preserve global hand anatomy and illumination tone.

```
[ Genuine Palm Image ] ──┐
                         ├──> [ Dual-Conditioned Swin Generator ] ──> [ Synthetic Fake Palm ]
[ Palm Mask / Contour ] ─┘                  │                                     │
                                            ▼                                     ▼
                                [ Multi-Objective Loss ] <───────────── [ PatchGAN Discriminator ]
                                 (L1 + Perceptual + Ridge)
```

---

## 🚀 Quick Start (Run on Google Colab)

The pipeline is completely self-contained in a single master notebook: [`Fake_Palm_Generation_Pipeline.ipynb`](Fake_Palm_Generation_Pipeline.ipynb).

1. Click the **Open In Colab** badge at the top (or upload `Fake_Palm_Generation_Pipeline.ipynb` to [Google Colab](https://colab.research.google.com/)).
2. Under **Runtime** > **Change runtime type**, select **T4 GPU** (available on Colab free tier).
3. Prepare the IITD Touchless Palmprint dataset:
   * Upload `IIT_Delhi_Palmprint_V1.zip` (or `.7z`) to your Google Drive under `/PalmProject/` (or upload directly to the Colab session storage).
4. Run all cells step-by-step:
   * **Step 1–3**: Environment verification, dependency installations (`timm`, `lpips`, `torcheval`), and Google Drive mounting.
   * **Step 4–6**: Automatic archive extraction, path resolution, and synchronized paired dataset loading.
   * **Step 7–9**: Architecture instantiation (Swin Transformer generator + PatchGAN discriminator) and optimizer setup.
   * **Step 10**: Training loop with automatic checkpointing to Google Drive.
   * **Step 11–13**: Single-image inference, quantitative evaluation (SSIM, PSNR, LPIPS), and multi-column visual diagnostics.
   * **Step 14**: Exporting paired Real/Fake palm datasets for Phase-2 classifier training.

---

## 🔬 Key Architectural Highlights

* **Pretrained Swin Transformer Backbone**: Captures long-range hierarchical spatial dependencies across principal palmar crease lines (heart, head, life lines).
* **Biometric Texture Modulation (BTM)**: Injects stochastic latent representations into intermediate transformer features to alter ridge minutiae without distorting global hand geometry.
* **Multiscale PatchGAN Discriminator**: Inspects localized $70 \times 70$ pixel patches to penalize blurriness and encourage realistic dermal ridge textures.
* **Drive Persistence & Session Resiliency**: Periodically checkpoints model weights and sample generation grids to Google Drive to protect against Colab runtime disconnections.

---

## 📊 Evaluation & Metrics

The notebook evaluates generated synthetic palms against real genuine palms across multiple standards:

| Metric | Target / Evaluation Criterion |
| :--- | :--- |
| **SSIM** (Structural Similarity) | Measures macro hand shape, finger contours, and illumination consistency |
| **PSNR** (Peak Signal-to-Noise Ratio) | Assesses pixel-level fidelity |
| **LPIPS** (Perceptual Loss) | Evaluates deep feature similarity against human visual perception |
| **Visual Diagnostics** | Multi-column comparison grid: Real Palm, Segmented Mask, Generated Fake Palm, and Error/Minutiae Diff |

---

## 📄 License & Citations

This project is intended for academic research, biometric security benchmarking, and presentation attack detection studies. If you use this work or the IIT Delhi Touchless Palmprint database, please acknowledge the original dataset authors and cite this repository.

