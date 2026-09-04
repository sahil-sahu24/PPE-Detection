import os
import time
import threading
import argparse
import cv2
from ultralytics import YOLO

from pathlib import Path

# ==============================================================================
# FORCE TCP FOR RTSP STREAMS (Prevents UDP frame corruption and smearing)
# ==============================================================================
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Base directory (Portable for any computer)
ROOT_DIR = Path(__file__).resolve().parent

# ==============================================================================
# CONFIGURATION & THRESHOLDS (TUNABLE)
# ==============================================================================
MODEL_PATH = str(ROOT_DIR / "Final Model" / "best.pt")

# Default confidence thresholds (0.0 to 1.0)
DEFAULT_HELMET_CONF = 0.50     # Below this, helmet is NOT shown
DEFAULT_HEAD_CONF = 0.45       # Below this, head is NOT shown (if above, labeled as NO HELMET)
DEFAULT_OVERLAP_THRESH = 0.25  # If helmet overlaps head by >= 25%, head is hidden

# Input source:
# - Camera ID (e.g. "0")
# - RTSP URL (e.g. "rtsp://admin:password@192.168.1.100:554/stream1")
# - Video file path ("video.mp4")
# - Image file path ("test.jpg")
DEFAULT_SOURCE = 0


# ==============================================================================
# THREADED RTSP / VIDEO STREAM READER (ZERO-LAG BUFFER PURGING)
# ==============================================================================
class RTSPVideoStream:
    """
    Threaded video reader dedicated for RTSP IP cameras and live webcams.
    Continuously discards old frames in a background thread so the inference
    loop always receives the latest, zero-lag frame. Includes auto-reconnection.
    """
    def __init__(self, src, reconnect_interval=3.0):
        self.src = src
        self.is_rtsp = isinstance(src, str) and (src.startswith("rtsp://") or src.startswith("rtsps://") or src.startswith("http://"))
        self.reconnect_interval = reconnect_interval

        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG if self.is_rtsp else cv2.CAP_ANY)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.connected = False

        if self.cap.isOpened():
            self.ret, self.frame = self.cap.read()
            self.connected = self.ret

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                self.connected = False
                if self.is_rtsp:
                    print(f"[RTSP] Stream disconnected. Reconnecting in {self.reconnect_interval}s...")
                    time.sleep(self.reconnect_interval)
                    self._reconnect()
                else:
                    time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = True
                    self.frame = frame
                    self.connected = True
            else:
                if self.is_rtsp:
                    print(f"[RTSP] Lost frame from {self.src}. Attempting reconnect...")
                    self.connected = False
                    self._reconnect()
                else:
                    self.ret = False
                    time.sleep(0.01)

    def _reconnect(self):
        with self.lock:
            if self.cap is not None:
                self.cap.release()
            time.sleep(self.reconnect_interval)
            print(f"[RTSP] Connecting to {self.src}...")
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG if self.is_rtsp else cv2.CAP_ANY)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.ret = True
                    self.frame = frame
                    self.connected = True
                    print("[RTSP] Reconnected successfully.")

    def read(self):
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def is_alive(self):
        return self.connected

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()


# ==============================================================================
# GEOMETRY & DETECTION LOGIC
# ==============================================================================
def compute_intersection_over_head(head_box, helmet_box):
    """
    Computes overlap ratio: Area(head ∩ helmet) / Area(head).
    If helmet covers a significant portion of head, head is wearing helmet.
    """
    x1 = max(head_box[0], helmet_box[0])
    y1 = max(head_box[1], helmet_box[1])
    x2 = min(head_box[2], helmet_box[2])
    y2 = min(head_box[3], helmet_box[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    head_area = (head_box[2] - head_box[0]) * (head_box[3] - head_box[1])
    if head_area <= 0:
        return 0.0

    return inter_area / head_area


def draw_box(img, box, label, color, thickness=2):
    """Draws rounded-style bounding box with solid label tag."""
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w, _ = img.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    # Box outline
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # Label badge
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    badge_y1 = max(0, y1 - text_h - 6)
    badge_y2 = y1
    badge_x2 = min(w, x1 + text_w + 6)

    cv2.rectangle(img, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 4), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)


def process_frame(frame, model, helmet_conf, head_conf, overlap_thresh, imgsz=416, fps=0.0):
    """
    Runs YOLO and applies threshold tuning and helmet/head suppression logic:
    1. Detects objects.
    2. Keeps helmet if conf >= helmet_conf.
    3. Keeps head only if conf >= head_conf.
    4. If head overlaps with any visible helmet >= overlap_thresh -> SUPPRESS HEAD.
    5. Remaining visible heads are labeled as 'NO HELMET'.
    """
    # Predict with lower conf floor so trackbar adjustments work dynamically
    results = model.predict(source=frame, conf=0.15, imgsz=imgsz, device=0, verbose=False)
    boxes = results[0].boxes

    raw_helmets = []
    raw_heads = []

    if boxes is not None and len(boxes) > 0:
        for b in boxes:
            cls_id = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            xyxy = b.xyxy[0].tolist()

            # Class 0: helmet, Class 1: head
            if cls_id == 0 and conf >= helmet_conf:
                raw_helmets.append({"box": xyxy, "conf": conf})
            elif cls_id == 1 and conf >= head_conf:
                raw_heads.append({"box": xyxy, "conf": conf})

    # Suppression logic: If showing helmet, DON'T show head
    valid_no_helmets = []
    for head in raw_heads:
        has_helmet = False
        for helmet in raw_helmets:
            overlap = compute_intersection_over_head(head["box"], helmet["box"])
            if overlap >= overlap_thresh:
                has_helmet = True
                break  # Suppress this head detection

        if not has_helmet:
            valid_no_helmets.append(head)

    # Draw detections
    annotated = frame.copy()

    # Draw Helmets (Green)
    for h in raw_helmets:
        label = f"HELMET {h['conf']:.0%}"
        draw_box(annotated, h["box"], label, color=(0, 200, 0), thickness=2)

    # Draw Bare Heads as "NO HELMET" (Red)
    for nh in valid_no_helmets:
        label = f"NO HELMET {nh['conf']:.0%}"
        draw_box(annotated, nh["box"], label, color=(0, 0, 230), thickness=2)

    # Top HUD Banner (clean two-line display if width is narrow)
    img_w = annotated.shape[1]
    helmet_count = len(raw_helmets)
    no_helmet_count = len(valid_no_helmets)

    fps_str = f" | {fps:.1f} FPS" if fps > 0 else ""

    if img_w < 650:
        hud_h = 56
        cv2.rectangle(annotated, (0, 0), (img_w, hud_h), (20, 20, 20), -1)
        status_text = f"HELMETS: {helmet_count}  |  NO HELMET: {no_helmet_count}{fps_str}"
        status_color = (0, 220, 0) if no_helmet_count == 0 else (0, 0, 255)
        cv2.putText(annotated, status_text, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)
        info_text = f"Thresh: Helm>={int(helmet_conf*100)}% Head>={int(head_conf*100)}% Overlap>={int(overlap_thresh*100)}%"
        cv2.putText(annotated, info_text, (12, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
    else:
        hud_h = 42
        cv2.rectangle(annotated, (0, 0), (img_w, hud_h), (20, 20, 20), -1)
        status_text = f"HELMETS: {helmet_count}   |   NO HELMET: {no_helmet_count}{fps_str}"
        status_color = (0, 220, 0) if no_helmet_count == 0 else (0, 0, 255)
        cv2.putText(annotated, status_text, (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)
        info_text = f"Thresh: Helmet>={int(helmet_conf*100)}% Head>={int(head_conf*100)}% Overlap>={int(overlap_thresh*100)}%"
        (tw, th), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(annotated, info_text, (img_w - tw - 15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)

    return annotated, helmet_count, no_helmet_count


def nothing(x):
    pass


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Helmet & Head Detection (RTSP / Webcam / Video / Image)")
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE,
                        help="RTSP URL (rtsp://...), Webcam ID (0), video file (.mp4), or image (.jpg)")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to trained best.pt weights")
    parser.add_argument("--helmet-conf", type=float, default=DEFAULT_HELMET_CONF, help="Helmet confidence threshold (0.0 to 1.0)")
    parser.add_argument("--head-conf", type=float, default=DEFAULT_HEAD_CONF, help="Head confidence threshold (0.0 to 1.0)")
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP_THRESH, help="Overlap threshold to suppress head under helmet")
    parser.add_argument("--imgsz", type=int, default=416, help="Inference image size")
    args = parser.parse_args()

    # Verify model weights
    if not os.path.isfile(args.model):
        alt_paths = [
            r"D:\PPE-Detection\runs\detect\yolov8_helmet-4\weights\best.pt",
            r"D:\PPE-Detection\runs\detect\yolov8_helmet-3\weights\best.pt",
            r"D:\PPE-Detection\best.pt"
        ]
        found = False
        for p in alt_paths:
            if os.path.isfile(p):
                args.model = p
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Trained model not found at: {args.model}")

    print(f"Loading YOLO model from: {args.model}")
    model = YOLO(args.model)

    window_name = "PPE Helmet Detection - RTSP / Stream"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 640)

    # Interactive Sliders
    cv2.createTrackbar("Helmet Conf %", window_name, int(args.helmet_conf * 100), 100, nothing)
    cv2.createTrackbar("Head Conf %", window_name, int(args.head_conf * 100), 100, nothing)
    cv2.createTrackbar("Overlap Suppr %", window_name, int(args.overlap * 100), 100, nothing)

    # Check source type
    src_str = str(args.source).strip()
    is_cam = src_str.isdigit()
    is_rtsp = src_str.startswith("rtsp://") or src_str.startswith("rtsps://") or src_str.startswith("http://")
    is_image = False

    if not is_cam and not is_rtsp and os.path.isfile(src_str):
        ext = os.path.splitext(src_str)[1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            is_image = True

    # --------------------------------------------------------------------------
    # CASE 1: STATIC IMAGE
    # --------------------------------------------------------------------------
    if is_image:
        print(f"Opening static image: {src_str}")
        frame = cv2.imread(src_str)
        if frame is None:
            print(f"Error: Could not read image at {src_str}")
            return

        print("Controls: Adjust sliders to tune thresholds. Press 'Q' or ESC to exit, 'S' to save image.")
        while True:
            h_conf = max(1, cv2.getTrackbarPos("Helmet Conf %", window_name)) / 100.0
            hd_conf = max(1, cv2.getTrackbarPos("Head Conf %", window_name)) / 100.0
            ov_thresh = cv2.getTrackbarPos("Overlap Suppr %", window_name) / 100.0

            annotated, n_helmets, n_no_helmets = process_frame(
                frame, model, h_conf, hd_conf, ov_thresh, imgsz=args.imgsz
            )

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("s"):
                out_name = "tuned_result.jpg"
                cv2.imwrite(out_name, annotated)
                print(f"Saved tuned screenshot to: {out_name}")

    # --------------------------------------------------------------------------
    # CASE 2: RTSP STREAM OR LIVE WEBCAM (THREADED FOR ZERO LAG)
    # --------------------------------------------------------------------------
    elif is_rtsp or is_cam:
        source_val = int(src_str) if is_cam else src_str
        stream_desc = f"RTSP IP Camera ({src_str})" if is_rtsp else f"Webcam Device ({source_val})"
        print(f"Connecting to {stream_desc} using threaded zero-lag capture...")

        stream = RTSPVideoStream(source_val)

        # Wait up to 5s for first frame
        wait_start = time.time()
        while not stream.is_alive() and (time.time() - wait_start) < 5.0:
            time.sleep(0.1)

        if not stream.is_alive():
            print(f"Error: Could not open {stream_desc}. Check connection and URL.")
            stream.stop()
            return

        print(f"Stream started! Real-time display active.")
        print("Controls: Adjust sliders in real time. Press 'Q' or ESC to quit, 'S' to save screenshot.")

        fps = 0.0
        frame_times = []

        while True:
            t0 = time.time()
            ret, frame = stream.read()
            if not ret or frame is None:
                # Brief sleep while waiting for fresh network packet
                time.sleep(0.01)
                continue

            # Compute rolling FPS
            frame_times.append(time.time())
            if len(frame_times) > 30:
                frame_times.pop(0)
            if len(frame_times) > 1:
                fps = len(frame_times) / (frame_times[-1] - frame_times[0])

            # Read current slider values
            h_conf = max(1, cv2.getTrackbarPos("Helmet Conf %", window_name)) / 100.0
            hd_conf = max(1, cv2.getTrackbarPos("Head Conf %", window_name)) / 100.0
            ov_thresh = cv2.getTrackbarPos("Overlap Suppr %", window_name) / 100.0

            annotated, n_helmets, n_no_helmets = process_frame(
                frame, model, h_conf, hd_conf, ov_thresh, imgsz=args.imgsz, fps=fps
            )

            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("s"):
                out_name = f"rtsp_capture_{int(time.time())}.jpg"
                cv2.imwrite(out_name, annotated)
                print(f"Saved capture to: {out_name}")

        stream.stop()

    # --------------------------------------------------------------------------
    # CASE 3: LOCAL VIDEO FILE
    # --------------------------------------------------------------------------
    else:
        print(f"Opening local video file: {src_str}")
        cap = cv2.VideoCapture(src_str)
        if not cap.isOpened():
            print(f"Error: Could not open video file {src_str}")
            return

        fps = 0.0
        frame_times = []

        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video file reached.")
                break

            frame_times.append(time.time())
            if len(frame_times) > 30:
                frame_times.pop(0)
            if len(frame_times) > 1:
                fps = len(frame_times) / (frame_times[-1] - frame_times[0])

            h_conf = max(1, cv2.getTrackbarPos("Helmet Conf %", window_name)) / 100.0
            hd_conf = max(1, cv2.getTrackbarPos("Head Conf %", window_name)) / 100.0
            ov_thresh = cv2.getTrackbarPos("Overlap Suppr %", window_name) / 100.0

            annotated, n_helmets, n_no_helmets = process_frame(
                frame, model, h_conf, hd_conf, ov_thresh, imgsz=args.imgsz, fps=fps
            )

            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("s"):
                out_name = f"video_capture_{int(time.time())}.jpg"
                cv2.imwrite(out_name, annotated)
                print(f"Saved capture to: {out_name}")

        cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
