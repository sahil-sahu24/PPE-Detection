from ultralytics import YOLO
import pynvml

def print_gpu_stats(trainer):
    """Callback to print live NVIDIA GPU utilization percentage and VRAM at each epoch."""
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        print(f"\n[NVIDIA GPU Status] Compute: {util.gpu}% | VRAM: {mem.used / 1024**2:.0f}/{mem.total / 1024**2:.0f} MB | Temp: {temp}°C\n")
    except Exception:
        pass

from pathlib import Path
import os
import torch

ROOT_DIR = Path(__file__).resolve().parent
DATA_YAML = ROOT_DIR / "data.yaml"
DATASET_DIR = ROOT_DIR / "dataset_helmet"

def main():
    # Verify dataset exists before training
    if not (DATASET_DIR / "images" / "train").exists():
        print("=" * 65)
        print("ERROR: Training dataset not found locally!")
        print(f"Expected folder: {DATASET_DIR / 'images' / 'train'}")
        print("Please place or download 'dataset_helmet' into this project directory.")
        print("=" * 65)
        return

    # Select compute device automatically
    device = 0 if torch.cuda.is_available() else "cpu"
    if device == 0:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected. Training on CPU.")

    # 1. Load YOLOv8 pretrained model
    model = YOLO("yolov8n.pt")  

    # 2. Add callback to log real hardware GPU % at the end of each epoch
    model.add_callback("on_train_epoch_end", print_gpu_stats)

    # 3. Train with optimized settings
    results = model.train(
        data=str(DATA_YAML),
        epochs=30,             # 30 epochs is plenty for fine-tuning
        imgsz=416,             # Native resolution of dataset
        batch=32,              # 125 batches per epoch (saturates GPU cores)
        device=device,         # CUDA GPU or CPU
        workers=0,             # Stable across Windows and Linux
        cache=True,            # Caches images in RAM for maximum throughput
        name="yolov8_helmet"   # Run name
    )

if __name__ == "__main__":
    main()
