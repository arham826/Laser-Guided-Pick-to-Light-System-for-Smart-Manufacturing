import cv2
import serial
import time


# ===== SETTINGS =====

PORT = "COM3"
CAMERA_INDEX = 1

HOME_X = 90
HOME_Y = 90

STEP = 1   # degree step


# ===== INITIALIZE SERIAL =====

arduino = serial.Serial(PORT, 115200)
time.sleep(2)


# ===== INITIAL SERVO POSITION =====

servoX = HOME_X
servoY = HOME_Y


def move(x, y):

    global servoX, servoY

    servoX = max(0, min(180, x))
    servoY = max(0, min(180, y))

    command = f"{servoX},{servoY}\n"
    arduino.write(command.encode())

    print(f"ServoX: {servoX}  ServoY: {servoY}")


# move to home initially

move(HOME_X, HOME_Y)


# ===== START CAMERA =====

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Camera not detected!")
    exit()


print("\nJog Control Active")
print("Arrow keys = Move laser")
print("O = Home position")
print("Q = Quit\n")


# ===== MAIN LOOP =====

while True:

    ret, frame = cap.read()

    if not ret:
        continue


    # show live camera

    cv2.putText(frame,
                "Jog Mode Active",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("Laser Jog Camera Monitor", frame)


    key = cv2.waitKeyEx(10)


    # LEFT ARROW

    if key == 2424832:
        move(servoX - STEP, servoY)


    # RIGHT ARROW

    elif key == 2555904:
        move(servoX + STEP, servoY)


    # UP ARROW

    elif key == 2490368:
        move(servoX, servoY + STEP)


    # DOWN ARROW

    elif key == 2621440:
        move(servoX, servoY - STEP)


    # HOME POSITION

    elif key == ord('o') or key == ord('O'):
        move(HOME_X, HOME_Y)


    # EXIT

    elif key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()