# Ship Detection in SAR Images using U-Net

## Project Overview
This repository contains **6 comprehensive experiments** for ship segmentation in Synthetic Aperture Radar (SAR) images using the U-Net architecture. The experiments systematically explore various augmentation techniques and architectural modifications to optimize ship detection performance.

## Experiments Summary

| Experiment | Description | Key Features |
|------------|-------------|---------------|
| Exp 01 | Baseline U-Net | Standard U-Net architecture without augmentation |
| Exp 02 | Strong Augmentation | U-Net with aggressive augmentation techniques |
| Exp 03 | Augmented Training | Enhanced training pipeline with data augmentation |
| Exp 04 | Tuned Augmentation | Optimized augmentation parameters for SAR data |
| Exp 05 | Optimized Augmentation | Fine-tuned augmentation strategy |
| Exp 06 | Skeleton + Dual Head | Advanced architecture with skeleton detection and dual-head output |

## Key Results
- Systematic evaluation of augmentation strategies for SAR ship detection
- Comparison of baseline vs. augmented approaches
- Novel dual-head architecture with skeleton prediction
- Optimized pipeline for maritime surveillance applications

## Technologies Used
- **Framework:** PyTorch / TensorFlow
- **Architecture:** U-Net and Variants
- **Data:** SAR Ship Detection Dataset
- **Augmentation:** Albumentations library
- **Visualization:** Matplotlib, Seaborn

##  Repository Structure
── train_ship_detection_experiment_01_baseline_unet.ipynb
── train_ship_detection_experiment_02_unet_strong_augmentation.ipynb
── train_ship_detection_experiment_03_unet_augmented_training.ipynb
── train_ship_detection_experiment_04_unet_tuned_augmentation.ipynb
── train_ship_detection_experiment_05_unet_optimized_augementation.ipynb
── train_ship_detection_experiment_06_ with_skelton_and_dual_head_Unet.ipynb


## Getting Started
1. Clone the repository
2. Install dependencies: `pip install torch torchvision matplotlib opencv-python albumentations`
3. Run experiments in Jupyter Notebook
4. Compare results across experiments

## Applications
- Maritime surveillance
- Automatic ship identification
- Coast guard monitoring
- Naval defense systems
- Commercial shipping tracking

## Future Work
- Integration with YOLO for detection
- Real-time inference pipeline
- Multi-class ship classification
- Transfer learning from pre-trained models

##  Contact
**Author:** Temesghen Eyassu  
**GitHub:** [Temesghen-Eyassu](https://github.com/Temesghen-Eyassu)  
**Repository:** [ship-detection-U-Net-experiments](https://github.com/Temesghen-Eyassu/ship-detection-U-Net-experiments)
