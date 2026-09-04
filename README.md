# PPE Helmet & Head Detection (YOLOv8)

Real-time Personal Protective Equipment (PPE) detection system for safety helmets and bare heads. Supports **RTSP IP cameras**, **live webcams**, **local video files**, and **images**.

---

## Features
- **Real-Time Detection**: Trained with YOLOv8 on 5,000 workplace safety images (`helmet` vs `head`).
- **Pretrained Weights Included**: Ready-to-use 6.2 MB model inside `Final Model/best.pt`.
- **RTSP IP Camera Support**: Threaded zero-lag frame grabber with TCP transport and auto-reconnection.
- **Head Suppression Logic**: Automatically suppresses head detections if an overlapping helmet is detected.
- **Dynamic Threshold Tuning**: Interactive GUI sliders to tune confidence thresholds and overlap ratios in real time.
- **Alert System**: Automatically highlights workers without helmets with red `NO HELMET` bounding boxes.

---

## Quick Start Guide

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/sahil-sahu24/PPE-Detection.git
cd PPE-Detection
pip install -r requirements.txt
```

### 2. Run Real-Time Detection

#### A. Live Webcam
```bash
python detect_helmet.py --source 0
```

#### B. RTSP IP Camera
```bash
python detect_helmet.py --source "rtsp://admin:password@192.168.1.100:554/stream1"
```

#### C. Test on Image or Video
```bash
python detect_helmet.py --source test.jpg
```

*Controls: Adjust the live sliders for Helmet Confidence, Head Confidence, and Overlap Suppression. Press `Q` to quit, `S` to save a screenshot.*

---

## Model Training

### Dataset Setup
To train the model, ensure the `dataset_helmet` folder is placed in the project root:
```text
PPE-Detection/
+-- data.yaml
+-- dataset_helmet/
¦   +-- images/
¦   ¦   +-- train/
¦   ¦   +-- val/
¦   +-- labels/
¦       +-- train/
¦       +-- val/
```

### Start Training
```bash
python train.py
```
Trained weights will be saved to `runs/detect/yolov8_helmet/weights/best.pt`.

---

## Project Structure
```text
PPE-Detection/
+-- Final Model/
¦   +-- best.pt             # Pretrained YOLOv8 model weights (6.2 MB)
+-- data.yaml               # Dataset configuration (portable paths)
+-- detect_helmet.py        # RTSP & webcam detection with live threshold sliders
+-- train.py                # Model training script with GPU telemetry
+-- build_dataset.py        # Pascal VOC XML to YOLO dataset converter
+-- requirements.txt        # Python package dependencies
+-- .gitignore              # Git ignore rules
+-- README.md               # Project documentation
```
