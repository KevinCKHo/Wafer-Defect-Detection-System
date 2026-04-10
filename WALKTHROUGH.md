# 🔍 System Training & Validation Walkthrough

Welcome to the Wafer Defect Detection System! This document provides a quick overview of how the training and validation pipelines operate under the hood, and how directories are structured during an active session.

## Environment & Setup

When initializing a new training cycle, ensure your environment is fully set up and activated.

For example, using our recently configured environment:
- **Environment Name**: `Defect_training_1`
- **Activation Script**: `.\Defect_training_1\Scripts\activate`

## Executing a Training Run

To launch training, pass your desired parameters to the heavily customized `train.py` script. 

Example Command:
```cmd
python train.py --version 11 --size n --epochs 100 --batch 16 --data dataset/70_15_15_C3/data.yaml
```

Once started, the system creates two completely independent output sequences:

### 1. Training Outputs (`runs/train`)
During execution, all fundamental weights, graphs, logs, and loss curves are saved directly to:
> `runs/train/yolov11n_[timestamp]/`

This ensures that active training cycles are organized linearly from newest to oldest.

### 2. Validation Processing (`runs/results`)
Upon completion of the training cycle, the script handles testing by running YOLO's validation protocols. These logs and test outputs securely bypass YOLO's native structure and are automatically saved into:
> `runs/results/res_yolov11n_[timestamp]/`

This structure prevents clashing directories and makes debugging your newest results streamlined and intuitive.
