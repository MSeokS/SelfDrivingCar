# LiDAR Lib
import Lib_LiDAR as LiDAR
import serial
import time
import numpy as np

arduino = serial.Serial(port='COM5', baudrate=9600, timeout=.1)
time.sleep(2)  # 아두이노 초기화 시간 대기

if __name__ == "__main__":
    env = LiDAR.libLidar('COM6')
    env.init()
    count = 0
    arduino.write("0\n".encode('utf-8'))

    for scan in env.scanning():
        scan1 = env.getAngleDistanceRange(scan, 90, 100, 500, 2000)

        if len(scan1) > 0:
            print(scan1)
            print('stop1')
            time.sleep(2)
            break
        else:
            print('go')
        if count == 600:
            env.stop()
            break

        count += 1

    
    count = 0
    for scan in env.scanning():
        scan2 = env.getAngleDistanceRange(scan, 90, 100, 500, 2000)

        if len(scan2) > 0:
            print(scan2)
            print('stop2')
            arduino.write("1\n".encode('utf-8'))
            time.sleep(5)
            break
        else:
            print('go')
        if count == 600:
            env.stop()
            break

        count += 1

    count = 0
    for scan in env.scanning():
        scan_l = env.getAngleDistanceRange(scan, 260, 280, 500, 900)
        scan_r = env.getAngleDistanceRange(scan, 90, 110, 500, 900)
        scan_comb = np.concatenate((scan_l, scan_r), axis=0)

        if len(scan_comb) > 0:
            print(scan_comb)
            print('stop3')
            env.stop()
            arduino.write("2\n".encode('utf-8'))
            break
        else:
            print('go')
        if count == 600:
            env.stop()
            break

        count += 1

    time.sleep(3)
    arduino.write("3\n".encode('utf-8'))
