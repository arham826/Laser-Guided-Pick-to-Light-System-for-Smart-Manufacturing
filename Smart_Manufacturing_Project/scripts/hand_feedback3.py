import cv2
import numpy as np
import serial
import time
from ultralytics import YOLO
from pathlib import Path

# Repository root: this file lives in <repo>/<project>/scripts/
REPO_ROOT = Path(__file__).resolve().parents[2]


# ================= SETTINGS =================

MODEL_PATH = str(REPO_ROOT / "Smart_Manufacturing_Project2" / "runs" / "detect" / "runs" / "parts_detection" / "weights" / "best.pt")
PORT = "COM3"
MAPPING_FILE = str(REPO_ROOT / "Smart_Manufacturing_Project" / "mapping_model.npy")

CAMERA_INDEX = 1

PART_CONF_THRESHOLD = 0.20
HAND_CONF_THRESHOLD = 0.40

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


# ================= CUSTOM PART SEQUENCE =================

part_sequence = [
    "p1","p14","p9","p2","p12","p15","p4","p6","p10","p8",
    "p15","p6","p4","p13","p7","p11","p2","p14","p17","p16",
    "p15","p3","p5","p9","p1","p12","p14","p5","p15"
]


last_hand_time = 0


# ================= MEMORY BUFFER =================

last_seen_positions = {}


# ================= MAIN LOOP =================

for part in part_sequence:

    print("Searching:",part)

    search_start=time.time()

    found=False


    while True:

        ret,frame=cap.read()

        if not ret:
            continue


        results=model(frame)[0]


        for box in results.boxes:

            conf=float(box.conf)
            label=model.names[int(box.cls)]

            if label.startswith("p") and conf>PART_CONF_THRESHOLD:

                x1,y1,x2,y2=map(int,box.xyxy[0])

                cx=(x1+x2)//2
                cy=(y1+y2)//2

                last_seen_positions[label]=(cx,cy)


                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                cv2.putText(frame,label,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,(0,255,0),2)


        # ===== MOVE USING MEMORY BUFFER =====

        if part in last_seen_positions:

            cx,cy=last_seen_positions[part]

            servoX=mx*cx+bx
            servoY=my*cy+by

            move(servoX,servoY)

            found=True
            break


        # retry detection instead of skipping

        if time.time()-search_start>SEARCH_TIMEOUT:

            print("Retrying detection:",part)

            search_start=time.time()


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


    # ================= WAIT FOR HAND + PART INSIDE HAND =================

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


        hand_box=None
        part_inside_hand=False


        for box in results.boxes:

            conf=float(box.conf)
            label=model.names[int(box.cls)]


            if label=="hand" and conf>HAND_CONF_THRESHOLD:

                hx1,hy1,hx2,hy2=map(int,box.xyxy[0])

                hand_box=(hx1,hy1,hx2,hy2)

                cv2.rectangle(frame,(hx1,hy1),(hx2,hy2),(0,0,255),3)

                cv2.putText(frame,"HAND",
                            (hx1,hy1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,(0,0,255),2)


        if hand_box is not None:

            hx1,hy1,hx2,hy2=hand_box


            for box in results.boxes:

                conf=float(box.conf)
                label=model.names[int(box.cls)]

                if label.startswith("p") and conf>PART_CONF_THRESHOLD:

                    px1,py1,px2,py2=map(int,box.xyxy[0])

                    pcx=(px1+px2)//2
                    pcy=(py1+py2)//2


                    if hx1<pcx<hx2 and hy1<pcy<hy2:

                        part_inside_hand=True
                        break


        if part_inside_hand:

            hand_counter+=1

            cv2.putText(frame,
                        "PART PICK CONFIRMED",
                        (30,80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0,0,255),
                        3)

        else:

            hand_counter=max(0,hand_counter-1)


        # ===== ENFORCE 2-SECOND GAP BETWEEN HAND CONFIRMATIONS =====

        if hand_counter>=HAND_REQUIRED_FRAMES:

            current_time=time.time()

            if current_time-last_hand_time>=2:

                last_hand_time=current_time

                print("Confirmed pick → next part")

                time.sleep(2)  # enforce delay before next confirmation allowed

                break


        cv2.imshow("Camera Monitor",frame)

        if cv2.waitKey(1)==ord('q'):
            exit()


print("PROCESS COMPLETE")