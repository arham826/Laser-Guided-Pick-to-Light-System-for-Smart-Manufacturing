import cv2
import numpy as np
import serial
import time
from ultralytics import YOLO


# ===== SETTINGS =====

MODEL_PATH = "C:\\Smart_Manufacturing_Project\\runs\\detect\\runs\\parts_detection\\weights\\best.pt"
PORT = "COM3"
MAPPING_FILE = "C:\\Smart_Manufacturing_Project\\mapping_model.npy"

CAMERA_INDEX = 1


# ===== INITIALIZE =====

arduino = serial.Serial(PORT, 115200)
time.sleep(2)

mx, bx, my, by = np.load(MAPPING_FILE)

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_INDEX)


def move(x, y):
    arduino.write(f"{int(x)},{int(y)}\n".encode())


print("System ready")
print("Press S to start real-time guided sequence")


# ===== WAIT FOR START =====

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    cv2.putText(frame,
                "Press S to start sequence",
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
        cap.release()
        cv2.destroyAllWindows()
        exit()


print("\nStarting adaptive laser guidance...\n")


# ===== PART SEQUENCE =====

part_sequence = [f"p{i}" for i in range(1, 18)]


# ===== MAIN GUIDANCE LOOP =====

for target_part in part_sequence:

    print("Searching:", target_part)

    part_found = False


    while not part_found:

        ret, frame = cap.read()

        if not ret:
            continue


        results = model(frame)[0]


        for box in results.boxes:

            cls = int(box.cls)

            label = model.names[cls]


            x1, y1, x2, y2 = box.xyxy[0]

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)


            # draw bounding boxes

            cv2.rectangle(frame,
                          (int(x1), int(y1)),
                          (int(x2), int(y2)),
                          (0, 255, 0),
                          2)


            cv2.putText(frame,
                        label,
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2)


            if label == target_part:

                servoX = mx * cx + bx
                servoY = my * cy + by

                move(servoX, servoY)

                print("Pointing at:", label)

                part_found = True


                start_time = time.time()


                # wait 3 seconds while still tracking movement

                while time.time() - start_time < 3:

                    ret, frame = cap.read()

                    if not ret:
                        continue


                    results = model(frame)[0]


                    for box in results.boxes:

                        cls = int(box.cls)
                        label_now = model.names[cls]


                        if label_now == target_part:

                            x1, y1, x2, y2 = box.xyxy[0]

                            cx = int((x1 + x2) / 2)
                            cy = int((y1 + y2) / 2)


                            servoX = mx * cx + bx
                            servoY = my * cy + by

                            move(servoX, servoY)


                            cv2.rectangle(frame,
                                          (int(x1), int(y1)),
                                          (int(x2), int(y2)),
                                          (0, 0, 255),
                                          3)


                            cv2.putText(frame,
                                        f"ACTIVE: {target_part}",
                                        (30, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1,
                                        (0, 0, 255),
                                        2)


                    cv2.imshow("Camera Monitor", frame)

                    if cv2.waitKey(1) == ord('q'):
                        break


        cv2.imshow("Camera Monitor", frame)

        if cv2.waitKey(1) == ord('q'):
            break


print("\nSequence complete ✅")


while True:

    ret, frame = cap.read()

    if not ret:
        continue

    cv2.imshow("Camera Monitor", frame)

    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()