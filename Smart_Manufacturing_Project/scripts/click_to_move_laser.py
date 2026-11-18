import cv2
import numpy as np
import serial
import time
from pathlib import Path

# Project folder: this file lives in <project>/scripts/
PROJECT_DIR = Path(__file__).resolve().parents[1]


########################################
# SETTINGS
########################################

PORT = "COM3"
BAUD = 115200
CAMERA_INDEX = 0

MAPPING_FILE = str(PROJECT_DIR / "mapping_model.npy")


# Servo limits (same workspace used in calibration)

SERVO_MIN_X = 60
SERVO_MAX_X = 130

SERVO_MIN_Y = 60
SERVO_MAX_Y = 130


########################################
# CONNECT ARDUINO
########################################

print("Connecting Arduino...")

arduino = serial.Serial(PORT, BAUD)

time.sleep(2)

print("Arduino connected")


########################################
# LOAD CALIBRATION MODEL
########################################

print("Loading calibration model...")

mx, bx, my, by = np.load(MAPPING_FILE)

print("\nMapping loaded successfully")

print("\nMapping equations:")

print(f"servoX = {mx:.6f} * pixelX + {bx:.6f}")
print(f"servoY = {my:.6f} * pixelY + {by:.6f}")


########################################
# HELPER FUNCTIONS
########################################

def clamp(value, minimum, maximum):

    return max(minimum, min(maximum, value))


def move_laser(px, py):

    servoX = mx * px + bx
    servoY = my * py + by

    servoX = clamp(servoX, SERVO_MIN_X, SERVO_MAX_X)
    servoY = clamp(servoY, SERVO_MIN_Y, SERVO_MAX_Y)

    command = f"{round(servoX)},{round(servoY)}\n"

    arduino.write(command.encode())

    print(f"\nClicked pixel: ({px}, {py})")
    print(f"Moving servo: ({round(servoX)}, {round(servoY)})")


########################################
# MOUSE CALLBACK FUNCTION
########################################

def mouse_click(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        move_laser(x, y)


########################################
# START CAMERA
########################################

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():

    print("Camera failed to open ❌")
    exit()


cv2.namedWindow("Click to Move Laser")

cv2.setMouseCallback("Click to Move Laser", mouse_click)


print("\nClick anywhere on the screen to move laser")
print("Press Q to exit\n")


########################################
# MAIN LOOP
########################################

while True:

    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Click to Move Laser", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break


cap.release()
cv2.destroyAllWindows()

print("Program closed")