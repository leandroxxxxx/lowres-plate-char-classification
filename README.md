# Low-Resolution License Plate Character Classification (`lowres-plate-char-classification`)

Single-character optical character recognition (OCR) classifier for single-line vehicle license plates captured in heavily degraded, low-resolution video footage (H.264 compression artifacts from dashcams and CCTV cameras).

## Overview

Each input sample is a single character crop spanning from the top edge to the bottom edge of the license plate at a fixed **9:16 aspect ratio**. Because the vertical bounds are preserved, vertical pixel positioning carries strong discriminative spatial cues (e.g., `Y` concentrates near the center, while `T` and `P` concentrate near the top).

This repository documents the iterative development of the classifier, starting with an exploratory baseline pipeline in Google Colab.

## Dataset & Active Classes

- **Input format:** Grayscale crops (1 channel), resized to `36×64` or `45×80`.
- **Total classes defined:** 36 indices (`0–25` for `A–Z`, `26–35` for `0–9`) to preserve output head compatibility for future data additions.
- **Active classes (30):** 10 digits (`0–9`) and 20 letters.
- **Excluded classes (6):** `M` (zero samples in dataset) and `Z`, `K`, `X`, `W`, `L` (excluded due to severe sample scarcity; masked out with $-\infty$ logits during training and evaluation).
- **Random guess baselines:** `1/30` (3.33%) overall, `1/10` (10.0%) for digits, and `1/20` (5.0%) for letters.

## Current Status: Version 1 (Baseline)

The `v1` baseline establishes an end-to-end training and diagnostic pipeline designed to benchmark performance and quantify data leakage before introducing domain-specific constraints:

- **Architecture:** ImageNet-pretrained `ResNet18`, with `conv1` adapted to 1-channel grayscale (channel weights summed) and a 36-class linear head.
- **Data Splitting:** Stratified random split (`70/15/15`) at the crop level. *Note: Because consecutive video frames from the same vehicle (`group_id`) appear across splits, `v1` accuracy represents an optimistic upper bound rather than real-world generalization.*
- **Training Setup:** Cross-entropy loss over active classes, AdamW optimizer, mixed precision (AMP), no data augmentation, and optional class-frequency weighting (disabled by default in Run 1).
- **Diagnostics:**
  - Overall and type-isolated (letters vs. digits) Top-1 and Top-3 accuracy.
  - Cross-type confusion analysis (e.g., `O/0`, `I/1`, `B/8`, `S/5`).
  - Performance breakdown across degradation quantiles (`orig_h`).
  - Pixel-space kNN / Logistic Regression sanity baseline to measure leakage inflation.

## Project Structure

```text
lowres-plate-char-classification/
├── notebooks/
│   └── v1_baseline_colab.ipynb    # End-to-end Colab training & diagnostic pipeline
├── configs/                       # Exported hyperparameter and split configurations
├── artifacts/                     # Saved checkpoints, normalization stats, and class masks (gitignored)
└── README.md
