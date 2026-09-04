from ultralytics import YOLO
import cv2

# ============================================================
# CONFIGURATION - CHANGE THESE VALUES MANUALLY
# ============================================================

# Trained model
MODEL_PATH = r"D:\PPE-Detection\runs\detect\yolov8_helmet-4\weights\best.pt"

# Webcam
CAMERA_ID = 0

# Confidence threshold
CONFIDENCE = 0.25

# Image size used for inference
IMAGE_SIZE = 640

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")
print("Starting webcam...")
print("Press Q to quit.")

# ============================================================
# OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    print("Try changing CAMERA_ID from 0 to 1.")
    exit()

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # Read frame from webcam
    success, frame = cap.read()

    if not success:
        print("ERROR: Could not read frame.")
        break

    # --------------------------------------------------------
    # RUN YOLO
    # --------------------------------------------------------

    results = model.predict(
    source=frame,
    conf=CONFIDENCE,
    imgsz=IMAGE_SIZE,
    device=0,
    verbose=False
)

    # --------------------------------------------------------
    # DRAW YOLO RESULTS
    # --------------------------------------------------------

    annotated_frame = results[0].plot()

    # Show result
    cv2.imshow("PPE Detection - YOLO", annotated_frame)

    # --------------------------------------------------------
    # PRESS Q TO EXIT
    # --------------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("Webcam stopped.")