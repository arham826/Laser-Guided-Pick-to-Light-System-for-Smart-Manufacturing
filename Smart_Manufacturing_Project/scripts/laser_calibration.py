import cv2
import numpy as np
import serial
import time
from pathlib import Path

# Project folder: this file lives in <project>/scripts/
PROJECT_DIR = Path(__file__).resolve().parents[1]


########################################
# SETTINGS (MATCH YOUR ARDUINO)
########################################

COM_PORT = "COM3"
CAMERA_INDEX = 1

CENTER_X = 90
CENTER_Y = 60

ANGLE_STEPS = [5, 10]
REPEAT_COUNT = 3


########################################
# CONNECT ARDUINO
########################################

print("Connecting Arduino...")

arduino = serial.Serial(COM_PORT, 9600)

time.sleep(2)

print("Arduino connected")


########################################
# MOVE SERVO FUNCTION
########################################

def move_servo(x, y):

    command = f"{x},{y}\n"
    arduino.write(command.encode())

    time.sleep(2)


########################################
# LASER DETECTION FUNCTION
########################################

def detect_laser(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0,70,70])
    upper_red1 = np.array([10,255,255])

    lower_red2 = np.array([170,70,70])
    upper_red2 = np.array([180,255,255])

    mask = cv2.inRange(hsv, lower_red1, upper_red1) + \
           cv2.inRange(hsv, lower_red2, upper_red2)

    contours,_ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:

        largest = max(contours,key=cv2.contourArea)

        x,y,w,h = cv2.boundingRect(largest)

        return (x+w//2, y+h//2), (x,y,w,h)

    return None, None


########################################
# GET STABLE LASER POSITION
########################################

def get_average_position(cap):

    points = []

    for _ in range(6):

        ret,frame = cap.read()

        if not ret:
            continue

        point,_ = detect_laser(frame)

        if point:
            points.append(point)

        time.sleep(0.05)

    if len(points) == 0:
        return None

    avg_x = int(np.mean([p[0] for p in points]))
    avg_y = int(np.mean([p[1] for p in points]))

    return (avg_x, avg_y)


########################################
# START CAMERA
########################################

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():

    print("Camera failed")

    exit()


########################################
# LIVE WINDOW FUNCTION
########################################

def show_frame():

    ret,frame = cap.read()

    if not ret:
        return None,None,None

    point,box = detect_laser(frame)

    if point:

        x,y,w,h = box

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.circle(frame,point,4,(0,255,0),-1)

    cv2.imshow("Calibration Window",frame)

    cv2.waitKey(1)

    return frame,point,box


########################################
# MOVE TO CENTER
########################################

print("Moving to center position")

move_servo(CENTER_X, CENTER_Y)

time.sleep(2)

center_pos = get_average_position(cap)

print("Center position:",center_pos)


########################################
# CALIBRATION FUNCTION
########################################

def calibrate_axis(axis):

    shifts = []

    for repeat in range(REPEAT_COUNT):

        print(f"\nRepeat {repeat+1} for {axis}-axis")

        for step in ANGLE_STEPS:

            if axis == "X":

                move_servo(CENTER_X + step, CENTER_Y)

            else:

                move_servo(CENTER_X, CENTER_Y + step)

            pos_plus = get_average_position(cap)


            if axis == "X":

                move_servo(CENTER_X - step, CENTER_Y)

            else:

                move_servo(CENTER_X, CENTER_Y - step)

            pos_minus = get_average_position(cap)


            move_servo(CENTER_X, CENTER_Y)


            if pos_plus and pos_minus:

                if axis == "X":

                    pixel_shift = pos_plus[0] - pos_minus[0]

                else:

                    pixel_shift = pos_plus[1] - pos_minus[1]


                angle_shift = 2 * step

                scale = pixel_shift / angle_shift

                shifts.append(scale)

                print(f"{axis} step {step}: scale =",scale)

    return np.mean(shifts)


########################################
# RUN CALIBRATION
########################################

print("\nCalibrating X axis...")

scaleX = calibrate_axis("X")

print("\nCalibrating Y axis...")

scaleY = calibrate_axis("Y")


########################################
# SAVE RESULTS
########################################

print("\nFinal calibration results:")

print("scaleX =",scaleX)
print("scaleY =",scaleY)

np.save(str(PROJECT_DIR / "laser_calibration.npy"),[scaleX,scaleY])

print("\nCalibration saved successfully")


########################################
# RETURN HOME
########################################

move_servo(CENTER_X, CENTER_Y)

print("Returned to center position")


########################################
# KEEP WINDOW OPEN UNTIL EXIT
########################################

print("\nPress Q to exit")

while True:

    frame,_,_ = show_frame()

    if frame is None:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()