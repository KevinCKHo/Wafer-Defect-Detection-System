# YOLO Training & Prediction Usage Manual

Complete YOLO Object Detection Training & Prediction Tools - Supports YOLOv8, YOLOv10, YOLOv11, YOLOv12

---

## 📋 Table of Contents

- [Features](#features)
- [Environment Setup](#environment-setup)
- [Downloading Pre-trained Weights](#downloading-pre-trained-weights)
- [Quick Start](#quick-start)
  - [Train Model](#train-model)
  - [Run Predictions](#run-predictions)
  - [Real-time Camera Detection](#real-time-camera-detection)
- [Training Parameters](#training-parameters)
  - [Model Config](#model-config)
  - [Training Config](#training-config)
  - [Optimizer Config](#optimizer-config)
  - [Data Augmentation](#data-augmentation)
  - [WandB Settings](#wandb-settings)
- [Prediction Parameters](#prediction-parameters)
- [Camera Parameters](#camera-parameters)
- [Common Examples](#common-examples)
- [Parameter Tuning Advice](#parameter-tuning-advice)
- [Troubleshooting](#troubleshooting)

---

## 🌟 Features

### Training Script (`train.py`)
- ✅ **Multi-version Support**: YOLOv8, YOLOv10, YOLOv11, YOLOv12
- ✅ **Multiple Sizes**: n (nano), s (small), m (medium), l (large), x (xlarge)
- ✅ **Custom Weights**: Load pre-trained weights seamlessly
- ✅ **WandB Integration**: Log training metrics automatically
- ✅ **Validation Visualizations**: Saves randomly sampled validation images every epoch
- ✅ **Full Argument Control**: Command-line control over all vital parameters
- ✅ **Auto-naming**: Generates meaningful project names based on datasets
- ✅ **Resume Training**: Recover and continue from an interrupted run seamlessly
- ✅ **Cleaner Layouts**: Now automatically separates outputs into `runs/train` and `runs/results`.

### Prediction Script (`predict.py`)
- ✅ **Multi-format Support**: Process images, videos, and batch folders
- ✅ **Confidence Control**: Adjust detection thresholds
- ✅ **Output Saving**: Automatically saves annotated results
- ✅ **Detailed Stats**: Outputs object counts, categories, and inference times

### Camera Script (`camera.py`)
- ✅ **Real-time Detection**: Process live objects directly from webcams
- ✅ **Auto-saving**: Automatically saves frames when objects are detected
- ✅ **Object Cropping**: Saves cropped instances of bounding boxes
- ✅ **Video Recording**: Save the entire detection stream as a video
- ✅ **Live Feedback**: Real-time stats on object counts and statuses

---

## 🔧 Environment Setup

### 1. Basic Dependencies

```bash
# Install PyTorch (CPU)
pip install torch torchvision

# PyTorch CUDA 11.8 (GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# PyTorch CUDA 12.1 or 12.4 (GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 2. Install YOLO & Python Modules

```bash
pip install ultralytics wandb opencv-python pillow
```

### 3. WandB Setup (Optional)

```bash
# Login to WandB
wandb login
```

---

## 📥 Downloading Pre-trained Weights

Pre-trained weights accelerate training and boost precision. 

### Method 1: Automatic (Recommended for Beginners)
The script will auto-download from Ultralytics when needed:
```bash
python train.py --version 11 --size n
```

### Method 2: Manual Download
You can download them offline to your `models/` directory for zero-delay startup.

```cmd
mkdir models
cd models
curl -L -o yolov11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
```
*(See the README.md for full curl commands covering versions v8 through v12).*

| Model Size | Parameters | Speed | Precision | Usage |
|------|----------|-----------|------------|-----------|
| **n** (nano) | Fewest | Fastest | Lower | Edge devices / Real-time |
| **s** (small) | Few | Fast | Medium | General use |
| **m** (medium) | Moderate | Moderate | Mid-High | Balanced |
| **l** (large) | Many | Slow | High | High accuracy requirement |
| **x** (xlarge) | Most | Slowest | Highest | Maximum accuracy |

---

## 🚀 Quick Start

### Train Model
```bash
# Basic YOLOv11n
python train.py --version 11 --size n

# YOLOv8s with modified epochs
python train.py --version 8 --size s --epochs 100 --batch 16
```

### Run Predictions
```bash
# Predict image
python predict.py --source image.jpg --weights runs/train/yolov11n_.../weights/best.pt
```

### Real-time Camera
```bash
# Open camera tracker with auto-saving activated
python camera.py --weights runs/train/yolov11n_.../weights/best.pt --auto-save
```

---

## 📚 Training Parameters

### Model Config

| Argument | Description | Default | Options |
|------|------|--------|------|
| `--version` | YOLO Version | `11` | `8`, `10`, `11`, `12` |
| `--size` | Model Size | `n` | `n`, `s`, `m`, `l`, `x` |
| `--weights` | Custom weights | Auto | Any `.pt` file |
| `--data` | Dataset YAML | `dataset/Etching_M1/data.yaml` | Any `.yaml` |

### Training Config

| Argument | Description | Default | Range |
|------|------|--------|----------|
| `--epochs` | Total Epochs | `100` | 50-300 |
| `--batch` | Batch Size | `16` | 4-64 |
| `--imgsz` | Image Size | `640` | 320, 640, 1280 |
| `--device` | GPU Device | `0` | `0`, `1`, `cpu` |
| `--workers` | Dataloader Threads | `8` | 4-16 |

### Optimizer Config

| Argument | Description | Default | Options |
|------|------|--------|-----------|
| `--optimizer` | Optimizer | `auto` | `SGD`, `AdamW`, `Adam`, `auto` |
| `--lr0` | Initial LR | `0.01` | 0.0001-0.1 |
| `--weight-decay` | Weight Decay | `0.0005` | 0.0-0.001 |

---

### Data Augmentation

Augmentations boost robustness and prevent overfitting. Note: YOLO dynamically adjusts data geometries to mimic new unseen data!

| Argument | Description | Default | Range |
|------|------|--------|------|
| `--degrees` | Random Rotation | `0.0` | 0.0-45.0 |
| `--translate` | Random Translate | `0.1` | 0.0-0.5 |
| `--scale` | Random Scale | `0.5` | 0.0-0.9 |
| `--flipud` | Vertical Flip | `0.0` | 0.0-1.0 |
| `--fliplr` | Horizontal Flip | `0.5` | 0.0-1.0 |
| `--mosaic` | Mosaic Prob | `1.0` | 0.0-1.0 |
| `--mixup` | MixUp Prob | `0.0` | 0.0-1.0 |

**Example (Heavy Augmentation for small datasets):**
```bash
python train.py --version 11 --size n \
    --degrees 15 --translate 0.2 --scale 0.7 \
    --flipud 0.2 --fliplr 0.5 --mosaic 1.0 --mixup 0.15
```

---

## 🎯 Prediction Parameters 

| Argument | Description | Required | Default |
|------|------|------|--------|
| `--source` | Media input (img/vid/dir) | ✅ | - |
| `--weights` | Trained `.pt` path | ✅ | - |
| `--conf` | Confidence Threshold | ❌ | `0.25` |
| `--save-txt` | Save Label TXT | ❌ | False |

---

## 🎥 Camera Parameters

| Argument | Description | Default |
|------|------|--------|
| `--camera` | Camera Index | `0` |
| `--auto-save` | Save frames upon detection | False |
| `--save-crops` | Save cropped boxes | False |
| `--save` | Record complete video | False |

**Example (Quality Control Logic):**
```bash
python camera.py --weights runs/train/.../weights/best.pt \
    --conf 0.7 --auto-save --save-crops --output-dir quality_control
```

---

## 🔍 Exploring Training Outputs

Upon completion, all metrics are automatically sorted into:

```text
runs/
├── train/
│   └── yolov11n_timestamp/
│       ├── weights/
│       │   ├── best.pt          # Use this for prediction!
│       │   └── last.pt          # Use this to resume training
│       ├── validation_samples/  # Raw image overlays
│       └── results.png          # Master curve graph
│
└── results/
    └── res_yolov11n_timestamp/
        └── (Post-training validation stats and matrices)
```

---

## ⚠️ Troubleshooting

#### 1. FileNotFoundError: data.yaml not found
**Fix:** Explicitly state your data file using absolute or correct relative paths: `--data dataset/70_15_15_C3/data.yaml`

#### 2. CUDA Out of Memory
**Fix:** Try reducing the batch size `--batch 8`, reducing image size `--imgsz 416`, or scaling down your model size `--size n`.

#### 3. Low mAP (Accuracy)
**Fix:** Train for more epochs (`--epochs 300`), use a larger model (`--size m`), and apply heavier augmentations (`--mixup 0.15 --degrees 10`).

#### 4. Invalid CUDA 'device=0' requested
**Fix:** You are missing the Nvidia CUDA version of PyTorch. Uninstall the CPU version and reinstall with the cu118/cu121/cu124 tags:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`

*(End of Manual)*
