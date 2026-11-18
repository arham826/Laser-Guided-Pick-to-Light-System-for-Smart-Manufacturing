from ultralytics import YOLO
import cv2
import serial
import time
import numpy as np
from pathlib import Path

# Repository root: this file lives in <repo>/<project>/scripts/
REPO_ROOT = Path(__file__).resolve().parents[2]


# ================= SETTINGS =================

MODEL_PATH = str(REPO_ROOT / "Smart_Manufacturing_Project2" / "runs" / "detect" / "runs" / "parts_detection" / "weights" / "best.pt")
MAPPING_FILE = str(REPO_ROOT / "Smart_Manufacturing_Project" / "mapping_model.npy")
PORT = "COM3"

CONF_THRESHOLD = 0.20
STABLE_FRAMES_REQUIRED = 3


# ================= LOAD MODEL =================

model = YOLO(MODEL_PATH)

mx, bx, my, by = np.load(MAPPING_FILE)


# ================= SERIAL =================

arduino = serial.Serial(PORT,115200)
time.sleep(2)


# ================= CLASS MAP =================

correct_part_order = {
  0:'hand',
  1:'p1',
  2:'p10',
  3:'p11',
  4:'p12',
  5:'p13',
  6:'p14',
  7:'p15',
  8:'p16',
  9:'p17',
  10:'p2',
  11:'p3',
  12:'p4',
  13:'p5',
  14:'p6',
  15:'p7',
  16:'p8',
  17:'p9'
}


# ================= CAMERA =================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


# ================= MEMORY =================

frame_memory = {}
locked_part = None


# ================= LASER MOVE FUNCTION =================

def move_laser(cx, cy):

    servoX = mx * cx + bx
    servoY = my * cy + by

    command = f"{round(servoX)},{round(servoY)}\n"

    arduino.write(command.encode())

    print("Laser moved to:", command.strip())


# ================= MAIN LOOP =================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    results = model(frame, imgsz=640, conf=CONF_THRESHOLD)

    detected_parts = []


    for box in results[0].boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        part_name = correct_part_order[cls]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2


        detected_parts.append(part_name)


        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)

        cv2.putText(frame,
                    f"{part_name} ({conf:.2f})",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,(255,0,0),2)


        # ===== STABILITY CHECK =====

        if part_name not in frame_memory:
            frame_memory[part_name] = []

        frame_memory[part_name].append((cx,cy))


        if len(frame_memory[part_name]) > STABLE_FRAMES_REQUIRED:
            frame_memory[part_name].pop(0)


        # ===== LOCK DETECTION =====

        if len(frame_memory[part_name]) == STABLE_FRAMES_REQUIRED:

            if locked_part != part_name:

                avg_x = sum(p[0] for p in frame_memory[part_name]) // STABLE_FRAMES_REQUIRED
                avg_y = sum(p[1] for p in frame_memory[part_name]) // STABLE_FRAMES_REQUIRED

                move_laser(avg_x, avg_y)

                locked_part = part_name

                print("Locked on:", part_name)


    # ===== UNLOCK IF PART DISAPPEARS =====

    if locked_part is not None:

        if locked_part not in detected_parts:

            print("Unlocked:", locked_part)

            locked_part = None


    cv2.imshow("Dynamic Detection (Stable Laser)", frame)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()