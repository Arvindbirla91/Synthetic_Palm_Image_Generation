# 🖐️ Synthetic Palm Image Generation using Transformers

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/Arvindbirla91/Synthetic_Palm_Image_Generation/blob/main/Fake_Palm_Generation_Pipeline.ipynb)

A Transformer-based generative deep learning pipeline for synthesizing contactless palm images and their corresponding segmented representations from genuine palm captures.

## 📌 Overview
This repository contains a notebook-based pipeline for generating synthetic palm biometric images. The model uses a genuine full-color palm image together with its corresponding segmentation mask as input. The generation workflow is designed to produce a synthetic full-color palm image and its corresponding segmented representation.
The main objective is to preserve the overall hand structure, palm shape, illumination, and principal creases while generating realistic, continuous palm-ridge textures.

## 🔄 Generation Pipeline
**Inputs:** Genuine full-color palm image and corresponding segmentation mask.
**Generator:** Transformer-based architecture with dual-stream conditioning and feature fusion.
**Outputs:** Synthetic full-color palm image and corresponding segmented palm image.

## 🧠 Model Architecture
* Swin Transformer Backbone
* Dual-Stream Conditioning
* Feature Fusion
* Multiscale Decoder
* PatchGAN Discriminator
* Combined Training Loss

## 📊 Evaluation Metrics
* SSIM
* PSNR
* LPIPS

## 🚀 Quick Start
The complete workflow is available in `Fake_Palm_Generation_Pipeline.ipynb`.
1. Open the notebook in Google Colab.
2. Select a GPU runtime.
3. Prepare the required palm-image dataset.
4. Run the notebook cells in sequence.
5. Inspect generated images and evaluation results.

## 🛠️ Technologies
Python, PyTorch, Hugging Face Transformers, torchvision, timm, LPIPS, Google Colab and Jupyter Notebook.

## 📄 License and Attribution
Please acknowledge the original dataset authors and follow the dataset provider's access and citation requirements.
