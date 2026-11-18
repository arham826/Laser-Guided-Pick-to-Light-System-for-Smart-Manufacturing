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
SEARCH_INTERVAL = 1


# ================= INITIALIZATION =================

arduino = serial.Serial(PORT,115200)
time.sleep(2)

mx,bx,my,by = np.load(MAPPING_FILE)

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_INDEX)


def move(x,y):

    command=f"{round(x)},{round(y)}\n"
    print("Moving:",command.strip())

    arduino.write(command.encode())


print("Waiting for container parts...")


# ================= WAIT FOR PARTS =================

while True:

    ret,frame=cap.read()

    if not ret:
        continue

    results=model(frame)[0]

    detected_parts=[]

    for box in results.boxes:

        conf=float(box.conf)

        if conf<CONF_THRESHOLD:
            continue

        cls=int(box.cls)
        label=model.names[cls]

        if label.startswith("p"):

            detected_parts.append(label)

            x1,y1,x2,y2=map(int,box.xyxy[0])

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            cv2.putText(frame,label,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,(0,255,0),2)


    cv2.putText(frame,
                "Waiting for parts...",
                (30,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,0),
                2)

    cv2.imshow("Camera Monitor",frame)

    if len(detected_parts)>0:

        print("Parts detected:",detected_parts)

        break


    if cv2.waitKey(1)==ord('q'):
        exit()


# ================= PART ORDER =================

part_sequence=[f"p{i}" for i in range(1,18)]


print("Starting guided picking sequence")


# ================= MAIN PICK LOOP =================

for target_part in part_sequence:

    print("Processing:",target_part)


    # ---------- FIND PART CENTER ----------

    while True:

        ret,frame=cap.read()

        if not ret:
            continue

        results=model(frame)[0]

        detected_boxes=[]


        for box in results.boxes:

            conf=float(box.conf)

            if conf<CONF_THRESHOLD:
                continue

            cls=int(box.cls)
            label=model.names[cls]

            x1,y1,x2,y2=map(int,box.xyxy[0])


            # DRAW ALL DETECTED PARTS CONTINUOUSLY

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            cv2.putText(frame,label,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,(0,255,0),2)


            if label==target_part:

                detected_boxes.append((x1,y1,x2,y2))


        if len(detected_boxes)>0:

            x_min=min(b[0] for b in detected_boxes)
            y_min=min(b[1] for b in detected_boxes)
            x_max=max(b[2] for b in detected_boxes)
            y_max=max(b[3] for b in detected_boxes)

            cx=(x_min+x_max)//2
            cy=(y_min+y_max)//2


            servoX=mx*cx+bx
            servoY=my*cy+by


            move(servoX,servoY)

            print("Pointing:",target_part)

            break


        cv2.putText(frame,
                    f"Searching {target_part}",
                    (30,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,255),
                    2)

        cv2.imshow("Camera Monitor",frame)

        if cv2.waitKey(1)==ord('q'):
            exit()

        time.sleep(SEARCH_INTERVAL)


    # ---------- WAIT UNTIL PICKED ----------

    while True:

        time.sleep(SEARCH_INTERVAL)

        ret,frame=cap.read()

        if not ret:
            continue

        results=model(frame)[0]

        still_present=False


        for box in results.boxes:

            conf=float(box.conf)

            if conf<CONF_THRESHOLD:
                continue

            cls=int(box.cls)
            label=model.names[cls]

            x1,y1,x2,y2=map(int,box.xyxy[0])


            # DRAW ALL DETECTIONS CONTINUOUSLY

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            cv2.putText(frame,label,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,(0,255,0),2)


            if label==target_part:

                still_present=True


                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)

                cv2.putText(frame,
                            f"Pick {target_part}",
                            (30,40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,0,255),
                            2)


        cv2.imshow("Camera Monitor",frame)


        if not still_present:

            print(target_part,"picked ✔")

            break


        if cv2.waitKey(1)==ord('q'):
            exit()


print("All parts completed")


# ================= KEEP CAMERA OPEN =================

while True:

    ret,frame=cap.read()

    if not ret:
        continue

    cv2.imshow("Camera Monitor",frame)

    if cv2.waitKey(1)==ord('q'):
        break


cap.release()
cv2.destroyAllWindows()