# Vehicle Detection Methods: A Comparative Study

> Comparing Thresholding, GMM, Tiny-YOLOv4, and YOLOv4 CSP-Darknet53 on traffic video — implemented in MATLAB

![MATLAB](https://img.shields.io/badge/MATLAB-R2023a%2B-orange?logo=mathworks)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Object%20Detection-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Overview

This project benchmarks **four car detection methods** on a real traffic video against annotated ground-truth bounding boxes. It spans the full spectrum from classical signal processing to modern deep learning:

| # | Method | Approach |
|---|--------|----------|
| 1 | **Thresholding** | Background subtraction + pixel-wise threshold |
| 2 | **GMM** | Gaussian Mixture Model foreground detection |
| 3 | **Tiny-YOLOv4** | Lightweight CNN (tiny-yolov4-coco) |
| 4 | **YOLOv4 CSP-Darknet53** | Full YOLOv4 with CSP backbone |

Each method is evaluated with **Precision**, **Recall**, and **F1-Score** at IoU = 0.5.

---

## Results

| Method | Precision | Recall | F1-Score |
|--------|:---------:|:------:|:--------:|
| Thresholding | 0.7681 | 0.7097 | 0.7377 |
| GMM | 0.8361 | 0.6931 | 0.7579 |
| Tiny-YOLOv4 | **1.0000** | 0.5056 | 0.6716 |
| **YOLOv4 CSP-Darknet53** | 0.9660 | **0.8465** | **0.9023** |

**Key findings:**
- YOLOv4 CSP-Darknet53 achieves the best overall F1-score (0.9023), balancing high precision and recall.
- Tiny-YOLOv4 attains perfect precision but misses nearly half of all cars (low recall), making it unsuitable where recall matters.
- GMM outperforms simple thresholding by adapting to dynamic backgrounds (waving branches, shadows, stopped vehicles).
- Thresholding is the fastest but is fragile to illumination changes and noise.

---

## Methods

### 1. Thresholding (Background Subtraction)

The simplest approach: subtract a static reference frame from each incoming frame, then threshold the absolute pixel-wise difference. Connected components above a minimum area are extracted as bounding boxes.

```
diffImg = |currentFrame - referenceFrame|
binaryMask = diffImg > threshold
```

**Pros:** Extremely fast, zero training required.
**Cons:** Breaks under lighting changes, shadows, or camera motion.

---

### 2. Gaussian Mixture Model (GMM)

Each pixel's intensity is modelled as a weighted mixture of Gaussians that updates online. Pixels that cannot be explained by any Gaussian component are classified as foreground (moving objects). This gives robustness to dynamic backgrounds.

MATLAB's `vision.ForegroundDetector` is used with `NumGaussians=3` and `NumTrainingFrames=5`.

**Pros:** Adaptive to gradual background changes.
**Cons:** Can produce multiple bounding boxes per vehicle; slower than thresholding.

---

### 3. Tiny-YOLOv4

A compressed variant of YOLOv4 with fewer convolutional layers, designed for real-time performance on constrained hardware. Detections are filtered to the `car` class with a confidence threshold of 0.3.

MATLAB model: `yolov4ObjectDetector("tiny-yolov4-coco")`

**Pros:** Very fast inference, high precision (no false positives in this test).
**Cons:** Reduced recall — misses roughly half of vehicles.

---

### 4. YOLOv4 CSP-Darknet53

The full YOLOv4 architecture using CSP-Darknet53 as the backbone. Cross-Stage Partial (CSP) connections improve gradient flow and reduce computation while maintaining accuracy.

MATLAB model: `yolov4ObjectDetector("csp-darknet53-coco")`

**Pros:** Best overall accuracy; generalises across diverse scenes.
**Cons:** High computational cost; requires a capable GPU for real-time use.

---

## Evaluation Metrics

All methods are evaluated frame-by-frame using:

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × (Precision × Recall) / (Precision + Recall)
IoU       = Area of Overlap / Area of Union   [threshold = 0.5]
```

A detection is counted as a **True Positive (TP)** only if its IoU with a ground-truth box exceeds 0.5. Per-frame metrics are averaged across all 120 evaluation frames.

---

## Repository Structure

```
.
├── vehicle_detection_comparison.m  # Main MATLAB script (all detectors + evaluation)
├── vehicle_detection_report.pdf    # Full project report with figures and references
├── README.md
└── LICENSE
```

> **Note:** The traffic video (`traffic.mj2`) and ground-truth annotation file (`export_ground_truth.mat`) are not included due to file size and licensing. See *Setup* below.

---

## Setup & Requirements

### Prerequisites

- MATLAB R2023a or later
- [Computer Vision Toolbox](https://www.mathworks.com/products/computer-vision.html)
- [Deep Learning Toolbox](https://www.mathworks.com/products/deep-learning.html)
- [Image Processing Toolbox](https://www.mathworks.com/products/image-processing.html)

### Required Data Files

Place the following files in the same directory as the `.m` script:

| File | Description |
|------|-------------|
| `traffic.mj2` | Traffic video (MJ2/Motion JPEG 2000 format) |
| `export_ground_truth.mat` | Ground-truth bounding boxes (MATLAB `gTruth` object) |

### Running

```matlab
% Open MATLAB and navigate to the project directory
cd path/to/img-proc

% Run the main script
Image_Processing_Code
```

The script will:
1. Load the video and ground-truth annotations
2. Run all four detectors sequentially (with live visualisation)
3. Print per-method metrics to the console
4. Display a grouped bar chart comparing Precision, Recall, and F1-Score

---

## Background & Motivation

Vehicle detection is a core task in intelligent transportation systems, traffic monitoring, and autonomous driving. This project was developed as part of an Image Processing module to compare the progression from hand-crafted signal processing methods to modern deep learning detectors — and to quantify the trade-offs between speed, simplicity, and accuracy.

---

## References

- Bochkovskiy, A., Wang, C.-Y., & Liao, H.-Y. M. (2020). *YOLOv4: Optimal Speed and Accuracy of Object Detection.* https://arxiv.org/abs/2004.10934
- Fradi, H., & Dugelay, J.-L. (2012). *Robust foreground segmentation using improved Gaussian mixture model and optical flow.* ICIEV 2012. https://doi.org/10.1109/ICIEV.2012.6317376
- Jiang, Y. et al. (2023). *YOLOv4-dense: A smaller and faster YOLOv4 for real-time edge-device based object detection in traffic scene.* IET Image Processing. https://doi.org/10.1049/ipr2.12656
- Medina, A. et al. (2024). *Learning manufacturing computer vision systems using Tiny YOLOv4.* Frontiers in Robotics and AI. https://doi.org/10.3389/frobt.2024.1331249
- Padilla, R. et al. (2021). *A comparative analysis of object detection metrics with a companion open-source toolkit.* Electronics. https://doi.org/10.3390/electronics10030279
- Wang, C.-Y., Bochkovskiy, A., & Liao, H.-Y. M. (2021). *Scaled-YOLOv4: Scaling Cross Stage Partial Network.* https://arxiv.org/abs/2011.08036

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
