#  Ship Detection in SAR Images using U-Net

##  Project Overview

This repository contains **six structured deep learning experiments** for ship detection and segmentation in Synthetic Aperture Radar (SAR) imagery using the **U-Net architecture and its extensions**.

The experiments systematically investigate:
- the impact of data augmentation strategies
- architectural improvements (including dual-head learning)
- skeleton-based structural supervision
- performance optimization for SAR-based segmentation

---

##  Experiments Summary

| Experiment | Description | Key Features |
|------------|-------------|---------------|
| Exp 01 | Baseline U-Net | Standard U-Net without augmentation |
| Exp 02 | Strong Augmentation | U-Net with aggressive augmentations |
| Exp 03 | Augmented Training | Improved training pipeline |
| Exp 04 | Tuned Augmentation | Optimized SAR augmentation settings |
| Exp 05 | Optimized Augmentation | Refined augmentation strategy |
| Exp 06 | Dual-Head U-Net | Multi-task learning with segmentation + skeleton prediction |

---

##  Results & Evaluation

All experimental results are summarized in:
metrics/metrics_summary_all_experiments.jpg


### Evaluated Metrics:
- Training & Validation Loss
- Dice Coefficient
- Intersection over Union (IoU)
- Skeleton prediction performance (Exp 06)

---

##  Technologies Used

- **Deep Learning Framework:** PyTorch
- **Model Architecture:** U-Net and Dual-Head U-Net
- **Data Type:** SAR (Synthetic Aperture Radar) imagery
- **Data Augmentation:** Albumentations
- **Visualization:** Matplotlib
- **Numerical Computing:** NumPy

---
##  Repository Structure


ship-detection-U-Net-experiments/

![Image](https://github.com/user-attachments/assets/096234b6-ca43-4a54-b443-fa2858f4aaf8)



##  Getting Started

### 1️ Clone Repository

```bash
git clone https://github.com/Temesghen-Eyassu/ship-detection-U-Net-experiments.git
cd ship-detection-U-Net-experiments
### 2️ Install Dependencies
pip install torch torchvision numpy matplotlib opencv-python albumentations rasterio scipy

### 3️ Run Experiments
Open any notebook inside src/
Run cells sequentially
Compare results across experiments

## Applications
Maritime surveillance
Automatic ship detection in SAR imagery
Coastal monitoring systems
Defense and naval intelligence
Commercial shipping analytics

## Future Work
Real-time inference pipeline
Multi-class ship classification
Transformer-based segmentation models
Transfer learning from pre-trained SAR models

## Course Information
Course: GEO-OMA24
Topic: Ship Detection in SAR Images
Institution: EAGLE Program

## Author
Name: Temesghen Eyassu
GitHub: Temesghen-Eyassu

Project: ship-detection-U-Net-experiments
