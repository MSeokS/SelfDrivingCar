# LiDAR Lib
import Lib_LiDAR as LiDAR
import serial
import time
import cv2
import numpy as np


class Arduino:
    def __init__(self):
        # 초기 상태 설정
        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.arduino = serial.Serial(port='COM5', baudrate=9600, timeout=.1)
        time.sleep(2.0)
        # LiDAR 초기화
        self.env = LiDAR.libLidar('COM3')
        self.env.init()
        self.computerVision = ComputerVision()

    def __delete__(self):
        self.cap.release()
        self.arduino.release()

    def move(self):
        # 모터 회전 후 이동
        try:
            check = 0
            count = 0   # 모드
            while True:
                # 라이더 코드
                scan = next(self.env.scanning())
                scan1 = self.env.getAngleDistanceRange(scan, 360, 360, 500, 1000)
                scan2 = self.env.getAngleDistanceRange(scan, 0, 10, 500, 1000)
    
                # 두 범위의 데이터를 결합
                scan_comb = np.vstack((scan1, scan2))

                if len(scan_comb) > 0:
                    print(scan_comb)
                    if count == 0:
                        print('turn left')
                        self.arduino.write("100\n".encode('utf-8'))
                        count += 1
                        time.sleep(6)
                    elif count == 1:
                        print('turn2 right')
                        self.arduino.write("200\n".encode('utf-8'))
                        count -= 1
                        time.sleep(6)

                # if 정지선을 발견 or 횡단보도 인식:
                # temp = 999
                # 신호등 루프를 돌고 초록불일때 탈출
                # 주행

                # 원래 주행 코드
                start = time.time()
                self.reward = self.take_picture()
                
                if self.reward is None:
                    print("Camera Disconnected")
                    angle = 999
                elif self.reward == -999:
                    print("No Line")
                    continue
                else:
                    angle = self.reward * angleK + 17
                                
                angle = int(angle)
                print(lineL, lineR, angle)
                
                end = time.time()
                if end - start < 0.1:
                    time.sleep(0.1 - (end - start))
                self.arduino.flush()
                self.arduino.write(f"{angle}\n".encode('utf-8'))
                cv2.waitKey(1)
        except serial.SerialException as e:
            print(f"Serial Error : {e}")
        except KeyboardInterrupt:
            print("stop")
        finally:
            time.sleep(1.0)
            self.arduino.write("999\n".encode('utf-8'))
    
    def take_picture(self):
        reward = self.computerVision.detect(self.cap)
        return reward




if __name__ == "__main__":    
    run = Arduino()
    run.move()
