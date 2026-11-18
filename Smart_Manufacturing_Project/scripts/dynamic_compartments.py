import cv2
import numpy as np
import serial
import time
from ultralytics import YOLO


# ================= SETTINGS =================

MODEL_PATH = "C:\\Smart_Manufacturing_Project\\runs\\detect\\runs\\parts_detection\\weights\\best.pt"
PORT = "COM3"
MAPPING_FILE = "C:\\Smart_Manufacturing_Project\\mapping_model.npy"

CAMERA_INDEX = 1
DWELL_TIME = 3

# distance threshold for grouping nearby detections
DIST_THRESHOLD = 120


# ================= INITIALIZATION =================

arduino = serial.Serial(PORT, 115200)
time.sleep(2)

mx, bx, my, by = np.load(MAPPING_FILE)

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_INDEX)


def move(x, y):
    arduino.write(f"{int(x)},{int(y)}\n".encode())


# ================= WAIT FOR START =================

print("Press S to start sequence")

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


# ================= DETECT ALL PARTS ONCE =================

print("Detecting compartments...")

ret, frame = cap.read()

results = model(frame)[0]


compartments = {}


for box in results.boxes:

    cls = int(box.cls)
    label = model.names[cls]

    x1, y1, x2, y2 = box.xyxy[0]

    if label not in compartments:
        compartments[label] = []

    compartments[label].append(
        (int(x1), int(y1), int(x2), int(y2))
    )


# ================= PART ORDER =================

part_sequence = [f"p{i}" for i in range(1, 18)]


print("\nStarting guidance sequence\n")


# ================= MAIN LOOP =================

for part in part_sequence:

    if part not in compartments:
        print(part, "not detected")
        continue


    boxes = compartments[part]


    # ===== CLUSTER NEARBY DETECTIONS ONLY =====

    centers = []

    for b in boxes:

        cx = (b[0] + b[2]) // 2
        cy = (b[1] + b[3]) // 2

        centers.append((cx, cy, b))


    ref_cx, ref_cy, _ = centers[0]

    cluster = []

    for cx, cy, b in centers:

        distance = ((cx - ref_cx)**2 + (cy - ref_cy)**2)**0.5

        if distance < DIST_THRESHOLD:
            cluster.append(b)


    # ===== COMBINE CLUSTERED BOXES =====

    x_min = min(b[0] for b in cluster)
    y_min = min(b[1] for b in cluster)

    x_max = max(b[2] for b in cluster)
    y_max = max(b[3] for b in cluster)


    center_x = int((x_min + x_max) / 2)
    center_y = int((y_min + y_max) / 2)


    servoX = mx * center_x + bx
    servoY = my * center_y + by


    print("Pointing:", part)


    move(servoX, servoY)


    hold_start = time.time()


    # ===== DISPLAY COMPARTMENT BOX DURING HOLD =====

    while time.time() - hold_start < DWELL_TIME:

        ret, live = cap.read()

        if not ret:
            continue


        cv2.rectangle(live,
                      (x_min, y_min),
                      (x_max, y_max),
                      (0, 0, 255),
                      3)


        cv2.circle(live,
                   (center_x, center_y),
                   6,
                   (0, 255, 255),
                   -1)


        cv2.putText(live,
                    f"{part} compartment",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2)


        cv2.imshow("Camera Monitor", live)


        if cv2.waitKey(1) == ord('q'):
            break


print("\nSequence complete ✅")


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