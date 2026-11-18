import cv2
import numpy as np
import serial
import time
from ultralytics import YOLO
from pathlib import Path

# Repository root: this file lives in <repo>/<project>/scripts/
REPO_ROOT = Path(__file__).resolve().parents[2]


# ===== SETTINGS =====

MODEL_PATH = str(REPO_ROOT / "Smart_Manufacturing_Project" / "runs" / "detect" / "runs" / "parts_detection" / "weights" / "best.pt")
PORT = "COM3"

MAPPING_FILE = str(REPO_ROOT / "Smart_Manufacturing_Project" / "mapping_model.npy")


arduino = serial.Serial(PORT, 115200)
time.sleep(2)


# ===== LOAD CALIBRATION MODEL =====

mx, bx, my, by = np.load(MAPPING_FILE)


# ===== LOAD YOLO MODEL =====

model = YOLO(MODEL_PATH)


# ===== CAMERA START =====

cap = cv2.VideoCapture(1)


def move(x, y):
    arduino.write(f"{int(x)},{int(y)}\n".encode())


print("Opening camera preview...")

# ===== STEP 1: LIVE PREVIEW BEFORE DETECTION =====

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    cv2.imshow("Camera Monitor", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):   # press S to start sequence
        break

    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()


print("\nDetecting parts...\n")


# ===== STEP 2: DETECT PARTS =====

results = model(frame)[0]


detections = []


for box in results.boxes:

    cls = int(box.cls)

    label = model.names[cls]

    x1, y1, x2, y2 = box.xyxy[0]

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    detections.append((label, cx, cy, x1, y1, x2, y2))


# ===== SORT p1 → p17 =====

detections.sort(key=lambda x: int(x[0][1:]))


# ===== DRAW DETECTIONS ON SCREEN =====

for label, cx, cy, x1, y1, x2, y2 in detections:

    cv2.rectangle(frame,
                  (int(x1), int(y1)),
                  (int(x2), int(y2)),
                  (0, 255, 0),
                  2)

    cv2.putText(frame,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2)


cv2.imshow("Detected Parts", frame)
cv2.waitKey(1500)


print("Starting laser sequence...\n")


# ===== STEP 3: MOVE LASER SEQUENCE =====

for label, cx, cy, _, _, _, _ in detections:

    servoX = mx * cx + bx
    servoY = my * cy + by

    print("Moving to:", label)

    move(servoX, servoY)

    start_time = time.time()

    # keep camera visible during wait

    while time.time() - start_time < 3:

        ret, frame = cap.read()

        if not ret:
            continue

        cv2.putText(frame,
                    f"Pointing: {label}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2)

        cv2.imshow("Camera Monitor", frame)

        if cv2.waitKey(1) == ord('q'):
            break


print("\nSequence complete ✅")


# ===== STEP 4: KEEP CAMERA RUNNING AFTER SEQUENCE =====

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    cv2.imshow("Camera Monitor", frame)

    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()