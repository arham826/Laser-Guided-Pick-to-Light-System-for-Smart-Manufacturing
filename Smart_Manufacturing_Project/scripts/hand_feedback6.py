from ultralytics import YOLO
import cv2
import numpy as np
import serial
import time
from pathlib import Path

# Repository root: this file lives in <repo>/<project>/scripts/
REPO_ROOT = Path(__file__).resolve().parents[2]

##########################################
# LOAD YOLO MODEL
##########################################

MODEL_PATH = str(REPO_ROOT / "Smart_Manufacturing_Project2" / "runs" / "detect" / "runs" / "parts_detection" / "weights" / "best.pt")
model = YOLO(MODEL_PATH)

##########################################
# LOAD 1D CALIBRATION FILE
##########################################

mapping = np.load(str(REPO_ROOT / "Smart_Manufacturing_Project" / "mapping_model.npy"))

ax = mapping[0]
bx = mapping[1]
ay = mapping[2]
by = mapping[3]

##########################################
# SERIAL CONNECTION
##########################################

arduino = serial.Serial('COM3',115200)
time.sleep(2)

##########################################
# CLASS LABEL MAP
##########################################

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

##########################################
# PICK SEQUENCE
##########################################

sequence = [

    "p1","p14","p9","p2","p12","p15","p4","p6","p10","p8",
    "p15","p6","p4","p13","p7","p11","p2","p14","p17","p16",
    "p15","p3","p5","p9","p1","p12","p14","p15"

]

current_target_index = 0
waiting_for_hand = False

##########################################
# HAND COOLDOWN VARIABLES
##########################################

hand_cooldown_time = 2
last_hand_time = 0


##########################################
# PIXEL → LASER FUNCTION
##########################################

def pixel_to_laser(px, py):

    lx = int(ax * px + bx)
    ly = int(ay * py + by)

    return lx, ly


##########################################
# BOX OVERLAP FUNCTION
##########################################

def boxes_overlap(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    return xA < xB and yA < yB


##########################################
# CAMERA START
##########################################

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


##########################################
# MAIN LOOP
##########################################

while True:

    ret, frame = cap.read()

    if not ret:
        break


    results = model(frame, imgsz=640, conf=0.25)

    hand_box = None
    current_time = time.time()


    ##########################################
    # DISPLAY CURRENT TARGET PART
    ##########################################

    if current_target_index < len(sequence):

        target_part = sequence[current_target_index]

        cv2.putText(frame,
                    f"TARGET PART: {target_part}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,255),
                    3)

    else:

        cv2.putText(frame,
                    "ALL PARTS COMPLETED",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    3)

        break


    ##########################################
    # PROCESS DETECTIONS
    ##########################################

    detected_part_boxes = []

    for box in results[0].boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        x1,y1,x2,y2 = map(int,box.xyxy[0])

        part_name = correct_part_order[cls]


        ##########################################
        # HAND DETECTION
        ##########################################

        if part_name == "hand" and (current_time - last_hand_time > hand_cooldown_time):

            hand_box = (x1,y1,x2,y2)

            cv2.putText(frame,
                        "HAND DETECTED",
                        (20,80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        3)


        ##########################################
        # STORE PART BOXES
        ##########################################

        if part_name != "hand":

            detected_part_boxes.append((x1,y1,x2,y2))


        ##########################################
        # TARGET PART HIGHLIGHT
        ##########################################

        if part_name == target_part:

            box_color = (0,255,0)

        else:

            box_color = (255,0,0)


        ##########################################
        # DRAW DETECTION BOX
        ##########################################

        label = f"{part_name} ({conf:.2f})"

        cv2.rectangle(frame,(x1,y1),(x2,y2),box_color,2)

        cv2.putText(frame,
                    label,
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2)


        ##########################################
        # MOVE LASER TO TARGET PART
        ##########################################

        if part_name == target_part and not waiting_for_hand:

            cx = (x1+x2)//2
            cy = (y1+y2)//2

            lx, ly = pixel_to_laser(cx, cy)

            command = f"{lx},{ly}\n"

            arduino.write(command.encode())

            print(f"Laser moved to {target_part} → ({lx},{ly})")

            waiting_for_hand = True


    ##########################################
    # VERIFY PICK USING HAND OVERLAP WITH ANY PART
    ##########################################

    if waiting_for_hand and hand_box:

        overlap_found = False

        for part_box in detected_part_boxes:

            if boxes_overlap(hand_box, part_box):

                overlap_found = True
                break


        if overlap_found:

            cv2.putText(frame,
                        "PART PICKED",
                        (20,120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        3)

            print("Part picked → moving next")

            current_target_index += 1
            waiting_for_hand = False
            last_hand_time = time.time()

        else:

            cv2.putText(frame,
                        "HAND DETECTED (NO PART)",
                        (20,120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        3)


    ##########################################
    # SHOW WINDOW
    ##########################################

    cv2.imshow("Laser Guided Pick-To-Light Monitor",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


##########################################
# CLEAN EXIT
##########################################

cap.release()
cv2.destroyAllWindows()
arduino.close()