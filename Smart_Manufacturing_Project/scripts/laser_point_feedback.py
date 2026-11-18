import cv2
import numpy as np
import serial
import time
from ultralytics import YOLO
from pathlib import Path

# Repository root: this file lives in <repo>/<project>/scripts/
REPO_ROOT = Path(__file__).resolve().parents[2]


# ================= SETTINGS =================

MODEL_PATH = str(REPO_ROOT / "Smart_Manufacturing_Project" / "runs" / "detect" / "runs" / "parts_detection" / "weights" / "best.pt")
PORT = "COM3"
MAPPING_FILE = str(REPO_ROOT / "Smart_Manufacturing_Project" / "mapping_model.npy")

CAMERA_INDEX = 1
CONF_THRESHOLD = 0.4


# ================= INITIALIZATION =================

arduino = serial.Serial(PORT, 115200)
time.sleep(2)

mx, bx, my, by = np.load(MAPPING_FILE)

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_INDEX)


def move(x, y):
    arduino.write(f"{int(x)},{int(y)}\n".encode())


print("Press S to start pick-confirmation guidance")


# ================= WAIT FOR START =================

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    cv2.putText(frame,
                "Press S to start",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("Camera Monitor", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        break

    if key == ord('q'):
        exit()


# ================= PART ORDER =================

part_sequence = [f"p{i}" for i in range(1, 18)]

print("\nStarting closed-loop picking sequence\n")


# ================= MAIN LOOP =================

for target_part in part_sequence:

    print(f"\nProcessing {target_part}")

    # ---------- FIND PART POSITION ----------

    while True:

        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame)[0]

        detected_boxes = []

        for box in results.boxes:

            conf = float(box.conf)
            if conf < CONF_THRESHOLD:
                continue

            cls = int(box.cls)
            label = model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # draw all detections
            cv2.rectangle(frame,
                          (x1, y1),
                          (x2, y2),
                          (0, 255, 0),
                          2)

            cv2.putText(frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

            if label == target_part:
                detected_boxes.append((x1, y1, x2, y2))


        cv2.imshow("Camera Monitor", frame)


        if len(detected_boxes) > 0:

            # combine multiple detections into one box

            x_min = min(b[0] for b in detected_boxes)
            y_min = min(b[1] for b in detected_boxes)
            x_max = max(b[2] for b in detected_boxes)
            y_max = max(b[3] for b in detected_boxes)

            center_x = int((x_min + x_max) / 2)
            center_y = int((y_min + y_max) / 2)


            servoX = mx * center_x + bx
            servoY = my * center_y + by


            move(servoX, servoY)

            print("Laser moved to", target_part)

            break


        if cv2.waitKey(1) == ord('q'):
            exit()


    # ---------- WAIT UNTIL PART REMOVED ----------

    print("Waiting for operator to pick", target_part)

    while True:

        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame)[0]

        still_present = False

        for box in results.boxes:

            conf = float(box.conf)
            if conf < CONF_THRESHOLD:
                continue

            cls = int(box.cls)
            label = model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame,
                          (x1, y1),
                          (x2, y2),
                          (0, 255, 0),
                          2)

            if label == target_part:

                still_present = True

                cv2.rectangle(frame,
                              (x1, y1),
                              (x2, y2),
                              (0, 0, 255),
                              3)

                cv2.putText(frame,
                            f"Pick {target_part}",
                            (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2)


        cv2.imshow("Camera Monitor", frame)


        if not still_present:

            print(target_part, "picked ✔")

            break


        if cv2.waitKey(1) == ord('q'):
            exit()


print("\nAll parts picked successfully ✅")


# ================= KEEP CAMERA OPEN =================

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    cv2.imshow("Camera Monitor", frame)

    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()