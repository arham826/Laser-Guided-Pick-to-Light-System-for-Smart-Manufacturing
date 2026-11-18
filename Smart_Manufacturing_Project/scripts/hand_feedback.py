import cv2
import numpy as np
import serial
import time
from ultralytics import YOLO


# ================= SETTINGS =================

MODEL_PATH = "C:\\Smart_Manufacturing_Project\\runs\\detect\\runs\\best.pt"
PORT = "COM3"
MAPPING_FILE = "C:\\Smart_Manufacturing_Project\\mapping_model.npy"

CAMERA_INDEX = 1

PART_CONF_THRESHOLD = 0.25
HAND_CONF_THRESHOLD = 0.6

HAND_REQUIRED_FRAMES = 2
HAND_COOLDOWN = 2

SEARCH_TIMEOUT = 3


# ================= INITIALIZATION =================

arduino = serial.Serial(PORT,115200)
time.sleep(2)

mx,bx,my,by = np.load(MAPPING_FILE)

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_INDEX)


def move(x,y):

    command=f"{round(x)},{round(y)}\n"
    arduino.write(command.encode())

    print("Moving:",command.strip())


print("System ready")


# ================= PART SEQUENCE =================

part_sequence=[f"p{i}" for i in range(1,18)]

last_hand_time=0


# ================= MAIN LOOP =================

for part in part_sequence:

    print("Searching:",part)

    search_start=time.time()

    found=False


    # ===== SEARCH PART IN REAL TIME =====

    while True:

        ret,frame=cap.read()

        if not ret:
            continue

        results=model(frame)[0]

        boxes=[]


        for box in results.boxes:

            conf=float(box.conf)
            label=model.names[int(box.cls)]

            if label==part and conf>PART_CONF_THRESHOLD:

                x1,y1,x2,y2=map(int,box.xyxy[0])

                boxes.append((x1,y1,x2,y2))


            if label.startswith("p") and conf>PART_CONF_THRESHOLD:

                x1,y1,x2,y2=map(int,box.xyxy[0])

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                cv2.putText(frame,label,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,(0,255,0),2)


        if len(boxes)>0:

            x_min=min(b[0] for b in boxes)
            y_min=min(b[1] for b in boxes)
            x_max=max(b[2] for b in boxes)
            y_max=max(b[3] for b in boxes)

            cx=(x_min+x_max)//2
            cy=(y_min+y_max)//2

            servoX=mx*cx+bx
            servoY=my*cy+by

            move(servoX,servoY)

            found=True

            break


        if time.time()-search_start>SEARCH_TIMEOUT:

            print(part,"not found, skipping")

            break


        cv2.putText(frame,
                    f"Searching {part}",
                    (30,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,255),
                    3)

        cv2.imshow("Camera Monitor",frame)

        if cv2.waitKey(1)==ord('q'):
            exit()


    if not found:
        continue


    # ===== WAIT HAND DETECTION =====

    hand_counter=0


    while True:

        ret,frame=cap.read()

        if not ret:
            continue

        results=model(frame)[0]


        cv2.putText(frame,
                    f"PICK: {part}",
                    (30,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,255),
                    3)


        for box in results.boxes:

            conf=float(box.conf)
            label=model.names[int(box.cls)]

            x1,y1,x2,y2=map(int,box.xyxy[0])


            if label=="hand" and conf>HAND_CONF_THRESHOLD:

                hand_counter+=1

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)

                cv2.putText(frame,"HAND",
                            (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,(0,0,255),2)


        if hand_counter>=HAND_REQUIRED_FRAMES:

            if time.time()-last_hand_time>HAND_COOLDOWN:

                last_hand_time=time.time()

                print("Hand detected → next part")

                break


        cv2.imshow("Camera Monitor",frame)

        if cv2.waitKey(1)==ord('q'):
            exit()


print("PROCESS COMPLETE")