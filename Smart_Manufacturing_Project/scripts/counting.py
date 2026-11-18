import cv2
import time
from ultralytics import YOLO


# ================= SETTINGS =================

MODEL_PATH = "C:\\Smart_Manufacturing_Project\\runs\\detect\\runs\\parts_detection\\weights\\best.pt"

CAMERA_INDEX = 1
CONF_THRESHOLD = 0.4
COUNT_REFRESH_INTERVAL = 1   # seconds


# ================= LOAD MODEL =================

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_INDEX)


# ================= COUNT FUNCTION =================

def count_parts(results):

    counts = {f"p{i}": 0 for i in range(1, 18)}

    for box in results.boxes:

        conf = float(box.conf)

        if conf < CONF_THRESHOLD:
            continue

        cls = int(box.cls)

        label = model.names[cls]

        if label in counts:
            counts[label] += 1

    return counts


# ================= DRAW COUNT PANEL =================

def draw_count_panel(frame, counts):

    x_pos = 20
    y_pos = 60

    cv2.putText(frame,
                "Detected Part Counts",
                (x_pos, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2)

    for part in counts:

        text = f"{part}: {counts[part]}"

        cv2.putText(frame,
                    text,
                    (x_pos, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2)

        y_pos += 25


print("Stable live counting started (updates every 1 second). Press Q to exit.")


# ================= INITIAL COUNT STATE =================

last_update_time = 0
counts = {f"p{i}": 0 for i in range(1, 18)}


# ================= MAIN LOOP =================

while True:

    ret, frame = cap.read()

    if not ret:
        continue


    current_time = time.time()


    # ===== UPDATE COUNTS EVERY 1 SECOND =====

    if current_time - last_update_time >= COUNT_REFRESH_INTERVAL:

        results = model(frame)[0]

        counts = count_parts(results)

        last_update_time = current_time


    # ===== DRAW BOUNDING BOXES CONTINUOUSLY =====

    results_live = model(frame)[0]

    for box in results_live.boxes:

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


        cv2.putText(frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2)


    # ===== DRAW COUNT PANEL =====

    draw_count_panel(frame, counts)


    cv2.imshow("Stable Live Part Counter", frame)


    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()