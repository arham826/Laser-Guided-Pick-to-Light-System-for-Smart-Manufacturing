from ultralytics import YOLO
import cv2
from pathlib import Path

# Repository root: this file lives in <repo>/<project>/scripts/
REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = str(REPO_ROOT / "Smart_Manufacturing_Project2" / "runs" / "detect" / "runs" / "parts_detection" / "weights" / "best.pt")

model = YOLO(MODEL_PATH)

correct_part_order = {
  0: 'hand',
  1: 'p1',
  2: 'p10',
  3: 'p11',
  4: 'p12',
  5: 'p13',
  6: 'p14',
  7: 'p15',
  8: 'p16',
  9: 'p17',
  10: 'p2',
  11: 'p3',
  12: 'p4',
  13: 'p5',
  14: 'p6',
  15: 'p7',
  16: 'p8',
  17: 'p9'
}

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, imgsz=640, conf=0.20)

    for box in results[0].boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        part_name = correct_part_order[cls]

        label = f"{part_name} ({conf:.2f})"

        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)

        cv2.putText(frame,label,(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,(255,0,0),2)

    cv2.imshow("Corrected Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()