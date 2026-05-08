import cv2
import pygame
import time
import threading
import queue
import os
import sys
import platform
import glob

# ==========================================
# CROSS-PLATFORM AUDIO LOADER
# ==========================================

ASSETS_DIR = "Assets"
SUPPORTED_AUDIO = ["*.mp3", "*.wav", "*.ogg", "*.flac", "*.aac", "*.m4a"]

def find_audio_file():
    """Auto-detect audio file from Assets folder on any OS."""
    if not os.path.exists(ASSETS_DIR):
        return None

    found = []
    for pattern in SUPPORTED_AUDIO:
        found += glob.glob(os.path.join(ASSETS_DIR, pattern))
        found += glob.glob(os.path.join(ASSETS_DIR, pattern.upper()))

    # Deduplicate while preserving order
    found = list(dict.fromkeys(found))

    if found:
        return os.path.normpath(found[0])
    return None

# Try config.py first (written by setup.py), fallback to auto-detect
try:
    from config import SOUND_FILE
    if not os.path.isfile(SOUND_FILE):
        raise FileNotFoundError
    SOUND_FILE = os.path.normpath(SOUND_FILE)
    print(f"[Audio] Loaded from config: {SOUND_FILE}")
except (ImportError, FileNotFoundError):
    SOUND_FILE = find_audio_file()
    if SOUND_FILE:
        print(f"[Audio] Auto-detected: {SOUND_FILE}")
    else:
        print(f"[WARNING] No audio file found in '{ASSETS_DIR}/' — alarm disabled")

# ==========================================
# CROSS-PLATFORM CAMERA INDEX
# ==========================================

def get_camera_index():
    """
    Windows : 0 works for most webcams
    Linux   : /dev/video0 = index 0
    macOS   : 0 for built-in, 1 for external
    """
    OS = platform.system()
    if OS == "Windows":
        # CAP_DSHOW is faster on Windows (DirectShow backend)
        return 0, cv2.CAP_DSHOW
    elif OS == "Darwin":
        return 0, cv2.CAP_AVFOUNDATION
    else:
        # Linux — try v4l2 backend
        return 0, cv2.CAP_V4L2

# ==========================================
# SETTINGS
# ==========================================

CONFIDENCE       = 0.5
DETECT_EVERY_N   = 3
INFER_SIZE       = 320
PHONE_CLASS_ID   = 77        # COCO: cell phone

MODEL_WEIGHTS = os.path.normpath(
    "ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb"
)
MODEL_CONFIG = os.path.normpath(
    "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
)

DISAPPEAR_DELAY = 2

# ==========================================
# VALIDATE MODEL FILES
# ==========================================

if not os.path.isfile(MODEL_WEIGHTS):
    print(f"[ERROR] Model weights not found: {MODEL_WEIGHTS}")
    print("        Run setup.py first to get download instructions.")
    sys.exit(1)

if not os.path.isfile(MODEL_CONFIG):
    print(f"[ERROR] Model config not found: {MODEL_CONFIG}")
    print("        Run setup.py first to get download instructions.")
    sys.exit(1)

# ==========================================
# LOAD MODEL
# ==========================================

print("[Model] Loading SSD MobileNet v3...")
net = cv2.dnn_DetectionModel(MODEL_WEIGHTS, MODEL_CONFIG)
net.setInputSize(INFER_SIZE, INFER_SIZE)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Smart backend — tries OpenCL, falls back to CPU on error
if platform.system() == "Windows":
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("[Backend] CPU (Windows)")
else:
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
        # Test kar lo ek dummy forward pass se
        test = cv2.dnn.blobFromImage(
            cv2.resize(cv2.imread("ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt") 
                      if False else 
                      cv2.Mat() if False else
                      __import__('numpy').zeros((320, 320, 3), dtype='uint8'),
            (320, 320)), 1.0/127.5, (320, 320), (127.5,)*3, swapRB=True)
        net.setInput(test)
        net.forward()
        print("[Backend] OpenCL ✔")
    except Exception:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("[Backend] OpenCL failed — fallback to CPU")

print("[Model] Ready")

# ==========================================
# PYGAME SOUND INIT (cross-platform)
# ==========================================

sound_enabled = False

if SOUND_FILE and os.path.isfile(SOUND_FILE):
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(SOUND_FILE)
        sound_enabled = True
        print(f"[Sound] Ready — {os.path.basename(SOUND_FILE)}")
    except Exception as e:
        print(f"[WARNING] Sound init failed: {e}")
        print("          Continuing without audio alarm.")
else:
    print("[WARNING] Sound disabled — no audio file found.")

# ==========================================
# CAMERA INIT
# ==========================================

cam_index, cam_backend = get_camera_index()
cap = cv2.VideoCapture(cam_index, cam_backend)

if not cap.isOpened():
    # Fallback: try default backend
    print(f"[Camera] Backend failed — trying default...")
    cap = cv2.VideoCapture(cam_index)

if not cap.isOpened():
    print("[ERROR] Could not open camera. Check if it is connected.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimize camera buffer lag

actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print(f"[Camera] {int(actual_w)}x{int(actual_h)} | OS: {platform.system()}")
print("Press ESC to quit\n")

# ==========================================
# SHARED STATE
# ==========================================

frame_queue     = queue.Queue(maxsize=1)
result_lock     = threading.Lock()
latest_boxes    = []
last_phone_seen = 0.0
sound_playing   = False
stop_event      = threading.Event()

# ==========================================
# DETECTION THREAD
# ==========================================

def detection_loop():
    global last_phone_seen

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        try:
            classes, confs, boxes = net.detect(frame, confThreshold=CONFIDENCE)
        except Exception as e:
            print(f"[Detection Error] {e}")
            continue

        found = []

        if len(classes) > 0:
            for cls, conf, box in zip(classes.flatten(), confs.flatten(), boxes):
                if int(cls) == PHONE_CLASS_ID:
                    found.append((*box, float(conf)))
                    last_phone_seen = time.time()

        with result_lock:
            latest_boxes.clear()
            latest_boxes.extend(found)


det_thread = threading.Thread(target=detection_loop, daemon=True)
det_thread.start()

# ==========================================
# MAIN LOOP
# ==========================================

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("[ERROR] Frame read failed — camera disconnected?")
        break

    # Fix mirror effect (all platforms)
    frame = cv2.flip(frame, 1)

    frame_count += 1

    # Feed detection thread every N frames (drop if busy)
    if frame_count % DETECT_EVERY_N == 0:
        small = cv2.resize(frame, (INFER_SIZE, INFER_SIZE))
        if not frame_queue.full():
            frame_queue.put(small)

    # ---- Draw boxes ----
    h, w = frame.shape[:2]
    scale_x = w / INFER_SIZE
    scale_y = h / INFER_SIZE

    with result_lock:
        boxes_copy = list(latest_boxes)

    phone_visible = len(boxes_copy) > 0

    for (bx, by, bw, bh, conf) in boxes_copy:
        x1 = int(bx * scale_x)
        y1 = int(by * scale_y)
        x2 = int((bx + bw) * scale_x)
        y2 = int((by + bh) * scale_y)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 80), 2)
        cv2.putText(
            frame,
            f"Phone {conf:.0%}",
            (x1, max(y1 - 8, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 255, 80), 2
        )

    # ---- Sound control ----
    now = time.time()

    if sound_enabled:
        if phone_visible and not sound_playing:
            print("[!] Phone detected — alarm ON")
            pygame.mixer.music.play(-1)
            sound_playing = True

        if sound_playing and (now - last_phone_seen > DISAPPEAR_DELAY):
            print("[!] Phone gone — alarm OFF")
            pygame.mixer.music.stop()
            sound_playing = False

    # ---- Status overlay ----
    if phone_visible:
        status = "PHONE DETECTED"
        color  = (0, 255, 80)
    elif not sound_enabled:
        status = "Monitoring (no audio)"
        color  = (100, 100, 255)
    else:
        status = "Monitoring..."
        color  = (180, 180, 180)

    cv2.putText(frame, status, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Platform info (top-right corner)
    os_label = platform.system()
    cv2.putText(frame, os_label,
                (w - 90, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    cv2.imshow("Phone Detector", frame)

    if cv2.waitKey(1) == 27:
        print("Exiting...")
        break

# ==========================================
# CLEANUP
# ==========================================

stop_event.set()
det_thread.join(timeout=2)

if sound_enabled:
    pygame.mixer.music.stop()
    pygame.mixer.quit()

cap.release()
cv2.destroyAllWindows()