import cv2
import numpy as np
import serial
import time

arduino = serial.Serial('COM3',115200)
time.sleep(2)

cap = cv2.VideoCapture(1)

positions = [
(68,114), (82,114), (95,114), (109,114),

(68,122), (82,122), (95,122), (109,122),

(68,130), (82,130), (95,130), (109,130),

(68,138), (82,138), (95,138), (109,138),

]


def move(x,y):
    arduino.write(f"{int(x)},{int(y)}\n".encode())


def detect_laser(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower1=np.array([0,120,200])
    upper1=np.array([10,255,255])

    lower2=np.array([170,120,200])
    upper2=np.array([180,255,255])

    mask=cv2.inRange(hsv,lower1,upper1)+cv2.inRange(hsv,lower2,upper2)

    contours,_=cv2.findContours(mask,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

    if contours:

        c=max(contours,key=cv2.contourArea)

        if cv2.contourArea(c)>5:

            x,y,w,h=cv2.boundingRect(c)

            return x+w//2,y+h//2,x,y,w,h

    return None


results=[]

print("Starting calibration...")

for sx,sy in positions:

    print("Moving:",sx,sy)

    move(sx,sy)

    time.sleep(3)

    start=time.time()

    detected=None

    while time.time()-start<2:

        ret,frame=cap.read()

        if not ret:
            continue

        r=detect_laser(frame)

        if r:

            cx,cy,x,y,w,h=r

            detected=(cx,cy)

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.imshow("Calibration",frame)

        cv2.waitKey(1)

    if detected:

        print("Detected:",detected)

        results.append([sx,sy,detected[0],detected[1]])

data=np.array(results)

servoX=data[:,0]
servoY=data[:,1]
pixelX=data[:,2]
pixelY=data[:,3]

A=np.vstack([pixelX,np.ones(len(pixelX))]).T
mx,bx=np.linalg.lstsq(A,servoX,rcond=None)[0]

A=np.vstack([pixelY,np.ones(len(pixelY))]).T
my,by=np.linalg.lstsq(A,servoY,rcond=None)[0]


np.save("mapping_model.npy",[mx,bx,my,by])

print("\nCalibration saved successfully ✅")