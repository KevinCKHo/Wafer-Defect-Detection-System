# 🔍 Wafer Defect Detection System

Welcome to the Wafer Defect Detection System! This repository contains a fully streamlined, GPU-accelerated pipeline for training, validating, and deploying YOLO models (v8 through v12) specifically tuned for semiconductor wafer defect analysis.

Whether you are training new models from scratch, running bulk video predictions, or setting up a live quality-control webcam station, everything you need is right here.

---

## 🏗️ Repository File Structure

When you clone this project, your directory will be structured like this:

```text
Wafer Defect detection system/
│
├── dataset/                     # 📂 Place your dataset folders and data.yaml files here
│   └── Video/                   # 📂 Place prerecorded testing videos here
│
├── models/                      # 📂 Downloaded pre-trained base models (.pt files)
│
├── runs/                        # 📂 Central hub for all generated outputs
│   ├── train/                   # 📈 Training weights, loss curves, and epoch logs
│   └── results/
│       ├── summary/             # 📊 Post-training validation matrix and stats (sum_yolo...)
│       ├── live detection/      # 📸 Live webcam auto-saved frames and video caps
│       └── video/               # 🎥 Output videos from predict.py (video_yolo...)
│
├── train.py                     # 🚀 Main script for training new models
├── predict.py                   # 📼 Script for analyzing prerecorded images/videos
├── camera.py                    # 🎥 Script for live webcam detection and auto-cropping
│
├── README.md                    # 📖 This quick-start guide
├── USAGE_MANUAL.md              # 📖 Exhaustive documentation (all flags & params)
└── requirements.txt             # 📦 Dependency map
```

---

## 🛠️ Step 1: Getting Started (Installation)

To get things running, you need to create a virtual environment and install the required dependencies with GPU acceleration.

**1. Create & Activate your Virtual Environment:**
```cmd
python -m venv Defect_training_1
.\Defect_training_1\Scripts\activate
```

**2. Install GPU-Accelerated PyTorch First:**
Because object detection relies heavily on your NVIDIA graphics card, you *must* install the CUDA toolkit version of PyTorch before anything else:
```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**3. Install Remaining Dependencies:**
```cmd
pip install -r requirements.txt
```

---

## 📥 Step 2: Download the Dataset (Roboflow)

This project utilizes the custom `etching-defect-class-3` wafer dataset. To process detections or reproduce the training loops here, you must download the dataset and place it in the `dataset/` directory.

> 🔗 **[View & Download the Dataset on Roboflow](https://universe.roboflow.com/search?q=etching-defect-class-3)**
> Download the sample video (video4) from YouTube via (https://youtu.be/LenLCRIkb0A)
> *(Note: Ensure you export the dataset in YOLO format. After downloading, extract the ZIP contents directly into `Wafer Defect detection system/dataset/`)*

---

## 🚀 Step 3: How to Train a Model

Use `train.py` to train new models. The system automatically routes all loss curves, graphs, and new model weights into `runs/train/[run_name]/weights/best.pt`. Once completed, it natively routes the validation summaries to `runs/results/summary/`.

**Basic Training Command (YOLOv11 Nano):**
```cmd
python train.py --version 11 --size n --epochs 100 --batch 16 --data dataset/Your_Dataset/data.yaml
```

**Advanced Training (Larger model + Augmentation):**
```cmd
python train.py --version 11 --size m --epochs 200 --batch 8 --mixup 0.15 --degrees 10
```

---

## 📼 Step 4: Run Predictions on Prerecorded Video

To analyze an existing video (like `dataset/Video/video4.mp4`), use `predict.py` and point it to the `best.pt` file generated during your training. The script will automatically intercept the output and save your newly drawn bounding-box video nicely into `runs/results/video/`.

**Run Prediction & Show Live Preview:**
```cmd
python predict.py --source dataset/Video/video4.mp4 --weights runs/train/yolov11n_.../weights/best.pt --show --conf 0.5
```

---

## 🎥 Step 5: Run Live Camera Detection

Use `camera.py` for real-time webcam inference. You can configure it to automatically capture screenshots whenever it spots a defect or record a continuous video. Files are smartly routed to `runs/results/live detection/`.

**Basic Real-Time View (No saving):**
```cmd
python camera.py --weights runs/train/yolov11n_.../weights/best.pt
```

**Auto-Save Defect Frames & Record Video:**
```cmd
python camera.py --weights runs/train/yolov11n_.../weights/best.pt --auto-save --save
```

*(Tip: If your computer pulls from the wrong camera, add `--camera 1` or `--camera 2` to switch lenses).*

---

### 📚 Need More Details?
Check out **`USAGE_MANUAL.md`** for extreme deep-dives into all tuning parameters, data augmentation strategies, and troubleshooting guides!
