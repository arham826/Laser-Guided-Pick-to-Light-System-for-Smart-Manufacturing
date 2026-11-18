import serial, time

arduino = serial.Serial("COM3",9600)
time.sleep(2)

arduino.write(b"95,85\n")   # test X axis
time.sleep(2)

arduino.write(b"90,90\n")   # test Y axis