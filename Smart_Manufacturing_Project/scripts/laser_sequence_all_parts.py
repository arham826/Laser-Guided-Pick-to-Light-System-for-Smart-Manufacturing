import cv2
import numpy as np
import serial
import time
from ultralytics import YOLO


########################################
# SETTINGS
########################################

COM_PORT = "COM3"
CAMERA_INDEX = 1

MODEL_PATH = r"C:\Smart_Manufacturing_Project\runs\detect\runs\parts_detection9\weights\best.pt"
CALIBRATION_PATH = r"C:\Smart_Manufacturing_Project\laser_calibration.npy"

CENTER_SERVO_X = 90
CENTER_SERVO_Y = 85

WAIT_TIME = 3

MIN_ANGLE = 55
MAX_ANGLE = 120


########################################
# CONNECT ARDUINO
########################################

print("Connecting Arduino...")

arduino = serial.Serial(COM_PORT, 9600)

time.sleep(2)

print("Arduino connected")


########################################
# LOAD CALIBRATION
########################################

scaleX, scaleY = np.load(CALIBRATION_PATH)

print("Calibration loaded")
print("scaleX =", scaleX)
print("scaleY =", scaleY)


########################################
# LOAD MODEL
########################################

model = YOLO(MODEL_PATH)

print("Model loaded successfully")


########################################
# ROBOFLOW CLASS ORDER FIX
########################################

roboflow_index_to_part = {
0:'p1',1:'p10',2:'p11',3:'p12',4:'p13',
5:'p14',6:'p15',7:'p16',8:'p17',
9:'p2',10:'p3',11:'p4',12:'p5',
13:'p6',14:'p7',15:'p8',16:'p9'
}


########################################
# REQUIRED VISIT SEQUENCE
########################################

visit_sequence = [
'p1','p2','p3','p4','p5','p6','p7','p8','p9',
'p10','p11','p12','p13','p14','p15','p16','p17'
]


########################################
# LASER DETECTION FUNCTION
########################################

def detect_laser(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0,120,120])
    upper_red1 = np.array([10,255,255])

    lower_red2 = np.array([170,120,120])
    upper_red2 = np.array([180,255,255])

    mask = cv2.inRange(hsv, lower_red1, upper_red1) + \
           cv2.inRange(hsv, lower_red2, upper_red2)

    contours,_ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:

        largest = max(contours, key=cv2.contourArea)

        x,y,w,h = cv2.boundingRect(largest)

        return (x+w//2 , y+h//2)

    return None


########################################
# START CAMERA
########################################

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

ret, frame = cap.read()

if not ret:

    print("Camera failed")

    exit()


########################################
# DETECT LASER ORIGIN
########################################

laser_origin = detect_laser(frame)

if laser_origin is None:

    print("Laser not detected!")

    exit()

print("Laser origin =", laser_origin)


########################################
# MAIN LOOP
########################################

print("\nLaser sequencing started")
print("Press Q to exit\n")


while True:

    ret, frame = cap.read()

    if not ret:

        break


    ####################################
    # RUN YOLO DETECTION
    ####################################

    results = model(frame, imgsz=640, conf=0.7)

    detected_parts = {}


    for box in results[0].boxes:

        cls = int(box.cls[0])

        x1,y1,x2,y2 = box.xyxy[0]

        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)

        part_name = roboflow_index_to_part[cls]

        detected_parts[part_name] = (cx,cy)

        cv2.circle(frame,(cx,cy),6,(255,0,0),-1)

        cv2.putText(frame,
                    part_name,
                    (cx+10,cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,0,0),
                    2)


    ####################################
    # DRAW LASER ORIGIN
    ####################################

    cv2.circle(frame, laser_origin, 8, (0,255,0), -1)


    ####################################
    # MOVE LASER SEQUENTIALLY
    ####################################

    if detected_parts:

        print("\nNew cycle started")

        for part in visit_sequence:

            if part in detected_parts:

                obj = detected_parts[part]

                dx = obj[0] - laser_origin[0]
                dy = obj[1] - laser_origin[1]

                servoX = CENTER_SERVO_X + dx/scaleX
                servoY = CENTER_SERVO_Y + dy/scaleY

                servoX = max(MIN_ANGLE, min(MAX_ANGLE, int(servoX)))
                servoY = max(MIN_ANGLE, min(MAX_ANGLE, int(servoY)))

                command = f"{servoX},{servoY}\n"

                arduino.write(command.encode())

                print("Laser moved to", part)

                start_time = time.time()

                while time.time() - start_time < WAIT_TIME:

                    if cv2.waitKey(1) & 0xFF == ord('q'):

                        cap.release()
                        cv2.destroyAllWindows()

                        print("System stopped")

                        exit()


        print("Cycle completed")


    ####################################
    # SHOW CAMERA WINDOW
    ####################################

    cv2.imshow("Laser Pick-To-Light System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break


cap.release()
cv2.destroyAllWindows()