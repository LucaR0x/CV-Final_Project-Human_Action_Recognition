# Human Action Recognition for Human-Robot Interaction (HRI)

This repository contains a C++ implementation of a Human Action Recognition (HAR) system evaluated on the **KTH Action Recognition Dataset**. 
The system compares two tracking and localization approaches: a **Classical Computer Vision** pipeline (median background subtraction and morphological filtering) and a **Deep Learning Fallback** pipeline (YOLOv8-Pose ONNX via OpenCV DNN).

---

## Directory & Dataset Structure

Before running the application, ensure the following directory structure is set up:

```
computer-vision-final-project/
├── CMakeLists.txt
├── README.md
├── report.tex / report.pdf
├── data/                      <-- DATASET DIRECTORY
│   ├── boxing/                │   Contains 6 subfolders (one per action class).
│   ├── handclapping/          │   Each folder contains 40frame .png image 
│   ├── handwaving/            │   sequences and groundtruth .txt bounding box
│   ├── jogging/               │   annotation files.
│   ├── running/               │
│   └── walking/               │
├── models/                    <-- DL MODELS DIRECTORY
│   └── yolov8n-pose.onnx       │   Pretrained YOLOv8-Pose ONNX model file
├── include/                   <-- C++ Header Files (.hpp)
├── src/                       <-- C++ Source Files (.cpp)
└── output/                    <-- Auto-generated output directory
    ├── cv/                    │   Classical CV output visualizations
    └── yolo/                  │   YOLO Deep Learning output visualizations
```

---

## System Architecture & Classes Overview

The codebase follows a modular Object Oriented C++ design:

1. **`DatasetLoader`** ([include/DatasetLoader.hpp](include/DatasetLoader.hpp))
   * Scans the `data/` directory, loads 40-frame image sequences, and parses ground-truth bounding box text annotations for the median frame (frame 20).

2. **`Tracker`** ([include/tracker.hpp](include/tracker.hpp)) — Classical Computer Vision Mode
   * Computes a sequence-wide median background image $\mathbf{B}(x,y)$, performs background subtraction, adaptive thresholding, and applies vertical morphological closing kernels to localize human actors.

3. **`YoloTracker`** ([include/YoloTracker.hpp](include/YoloTracker.hpp)) — Deep Learning Fallback Mode
   * Loads `models/yolov8n-pose.onnx` via the OpenCV DNN module. Detects 17 human body keypoints, computes their convex hull, encloses extremities, and applies Exponential Moving Average (EMA) filtering to reduce jitter.

4. **`FeatureExtractor`** & **`YoloFeatureExtractor`** ([include/FeatureExtractor.hpp](include/FeatureExtractor.hpp))
   * Extract spatiotemporal motion features using dense Farnebäck optical flow, body sub region flow energies (torso vs legs), percentile translation speeds, and bounding box relative scale ratios, constructing a 27-element feature vector $\mathbf{x} \in \mathbb{R}^{27}$.

5. **`Classifier`** ([include/Classifier.hpp](include/Classifier.hpp))
   * Implements a Radial Basis Function (RBF) kernel Support Vector Machine (SVM). Handles Zscore feature standardization, hyperparameter tuning grid search, and Stratified 6-Fold Cross Validation.

6. **`main.cpp`** ([src/main.cpp](src/main.cpp))
   * Application entry point. Coordinates dataset loading, executes pipeline tracking/classification, computes median Intersection over Union (mIoU) tracking scores, and prints cross-validation metrics.

---

## Build & Execution Guide

### Prerequisites
* C++17
* CMake (version >= 3.10)
* OpenCV (version 4.x with DNN module support)

### Compilation
From the project root directory, run:

```bash
mkdir -p build
cd build
cmake ..
make
```

---

### Running the Executable

Inside the `build/` directory, run `./main` using one of the supported command-line flags:

#### 1. Classical Computer Vision Pipeline
Runs median background subtraction tracking:
```bash
./main
```

#### 2. Deep Learning Pipeline (YOLOv8-Pose)
Runs keypoint-based pose estimation tracking:
```bash
./main --use-yolo
```

#### 3. Complete Comparative Execution
Runs **both pipelines sequentially** and prints a side-by-side performance comparison summary table:
```bash
./main --use-all
```

---

## Quantitative Results Summary

| Operational Mode | Tracking mIoU | 6-Fold CV Accuracy | Overall F1-Score |
| :--- | :---: | :---: | :---: |
| **Classical Computer Vision** | 0.7567 | 70.83% | 0.71 |
| **Deep Learning (YOLOv8-Pose)** | **0.8544** | **93.06%** | **0.93** |

Predicted bounding box visualizations (green) overlaid with Ground Truth annotations (blue) for frame 20 will be saved automatically in `output/cv/visualizations/` and `output/yolo/visualizations/`.
