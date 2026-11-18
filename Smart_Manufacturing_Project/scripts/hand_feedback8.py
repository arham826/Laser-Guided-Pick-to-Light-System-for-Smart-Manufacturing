from ultralytics import YOLO
import cv2
import numpy as np
import serial
import time
from pathlib import Path

# Repository root: this file lives in <repo>/<project>/scripts/
REPO_ROOT = Path(__file__).resolve().parents[2]

##########################################
# SETTINGS
##########################################

MODEL_PATH = str(REPO_ROOT / "Smart_Manufacturing_Project2" / "runs" / "detect" / "runs" / "parts_detection" / "weights" / "best.pt")
MAPPING_FILE = str(REPO_ROOT / "Smart_Manufacturing_Project" / "mapping_model.npy")

PART_CONF_THRESHOLD = 0.25
HAND_CONF_THRESHOLD = 0.50

HAND_REQUIRED_FRAMES = 2
HAND_COOLDOWN = 3

##########################################
# LOAD MODEL + CALIBRATION
##########################################

model = YOLO(MODEL_PATH)

mx, bx, my, by = np.load(MAPPING_FILE)

##########################################
# SERIAL CONNECTION
##########################################

arduino = serial.Serial("COM3",115200)
time.sleep(2)

##########################################
# PART LABEL MAP
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
last_hand_time = 0
hand_counter = 0

##########################################
# PIXEL → LASER FUNCTION
##########################################

def pixel_to_laser(px, py):

    lx = int(mx * px + bx)
    ly = int(my * py + by)

    return lx, ly


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
        continue


    results = model(frame)[0]

    detected_part_boxes = []
    hand_box = None

    ##########################################
    # TARGET DISPLAY
    ##########################################

    if current_target_index >= len(sequence):

        cv2.putText(frame,
                    "ALL PARTS COMPLETED",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    3)

        break


    target_part = sequence[current_target_index]

    cv2.putText(frame,
                f"TARGET PART: {target_part}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,255),
                3)


    ##########################################
    # DETECTION LOOP
    ##########################################

    for box in results.boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        part_name = correct_part_order[cls]

        x1,y1,x2,y2 = map(int,box.xyxy[0])


        ##########################################
        # HAND DETECTION WITH THRESHOLD
        ##########################################

        if part_name == "hand" and conf > HAND_CONF_THRESHOLD:

            hand_box = (x1,y1,x2,y2)

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)

            cv2.putText(frame,
                        "HAND DETECTED",
                        (20,80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        3)


        ##########################################
        # STORE PART BOXES WITH THRESHOLD
        ##########################################

        if part_name != "hand" and conf > PART_CONF_THRESHOLD:

            detected_part_boxes.append((x1,y1,x2,y2))


        ##########################################
        # TARGET PART HIGHLIGHT
        ##########################################

        if part_name == target_part and conf > PART_CONF_THRESHOLD:

            box_color = (0,255,0)

        else:

            box_color = (255,0,0)


        cv2.rectangle(frame,(x1,y1),(x2,y2),box_color,2)

        cv2.putText(frame,
                    f"{part_name} ({conf:.2f})",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2)


        ##########################################
        # MOVE LASER TO TARGET PART
        ##########################################

        if part_name == target_part and conf > PART_CONF_THRESHOLD and not waiting_for_hand:

            cx = (x1+x2)//2
            cy = (y1+y2)//2

            lx, ly = pixel_to_laser(cx, cy)

            command = f"{lx},{ly}\n"

            arduino.write(command.encode())

            print("Laser moved to", target_part)

            waiting_for_hand = True


    ##########################################
    # INDUSTRIAL FEEDBACK LOOP
    ##########################################

    if waiting_for_hand and hand_box:

        hx1, hy1, hx2, hy2 = hand_box

        part_inside_hand = False


        for px1,py1,px2,py2 in detected_part_boxes:

            pcx = (px1 + px2)//2
            pcy = (py1 + py2)//2

            if hx1 < pcx < hx2 and hy1 < pcy < hy2:

                part_inside_hand = True
                break


        ##########################################
        # TEMPORAL CONFIRMATION BUFFER
        ##########################################

        if part_inside_hand:

            hand_counter += 1

            cv2.putText(frame,
                        "PART PICK CONFIRMED",
                        (20,120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        3)

        else:

            hand_counter = max(0, hand_counter - 1)


        ##########################################
        # MULTI-FRAME VALIDATION + COOLDOWN
        ##########################################

        if hand_counter >= HAND_REQUIRED_FRAMES:

            current_time = time.time()

            if current_time - last_hand_time >= HAND_COOLDOWN:

                print("Confirmed pick → next part")

                current_target_index += 1
                waiting_for_hand = False
                last_hand_time = current_time
                hand_counter = 0


    ##########################################
    # DISPLAY WINDOW
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