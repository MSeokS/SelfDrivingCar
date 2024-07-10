import numpy as np
import serial
import cv2
import time
import sys

# 하이퍼파라미터 설정

angleK = 18 # 회전 각도 배율 

lineR = 470 # 기본 라인 위치 설정
lineL = 170

RMax = 520 # 우측 임계값 (좌측 차선) -> 좌측 차선에 민감하게 반응하면 올리고 둔감하면 내리기
RMin = 420 # 좌측 임계값 (우측 차선) -> 우측 차선에 민감하게 반응하면 내리고 둔감하면 올리기

laneMove = 0.25 # 라인에서 벗어났을 때 움직이는 정도
laneC = 20 # 라인 보정치 -> 웬만하면 건들지 말고 라인 보정 이후에 정신 못차리면 수정(lane Move 크게 바꾸면 비례해서 수정 추천)

Svalue = 50 # 색 구분값 초록색 인식되면 내리고 흰색 안보이면 올리기

class ComputerVision(object):
    def grayscale(self, img):
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        """
        imgH, imgS, imgV = cv2.split(hsv)
        cv2.imshow("H", imgH)
        cv2.imshow("S", imgS)
        cv2.imshow("V", imgV)
        """
        lower = np.array([0, 0, 150])
        upper = np.array([255, Svalue, 255])

        mask = cv2.inRange(hsv, lower, upper)

        white = cv2.bitwise_and(img, img, mask=mask)
        return cv2.cvtColor(white, cv2.COLOR_BGR2GRAY)

    def gaussian_blur(self, img, kernel_size=(None, None)):
        return cv2.GaussianBlur(img, kernel_size, 0)

    def canny_edge(self, img, lth, hth):
        return cv2.Canny(img.copy(), lth, hth)

    def histogram_equalization(self, gray):
        return cv2.equalizeHist(gray)

    def hough_transform(self, img, rho=None, theta=None, threshold=None, mll=None, mlg=None, mode="lineP"):
        if mode == "line":
            return cv2.HoughLines(img.copy(), rho, theta, threshold)
        elif mode == "lineP":
            return cv2.HoughLinesP(img.copy(), rho, theta, threshold, lines=np.array([]),
                                   minLineLength=mll, maxLineGap=mlg)
        elif mode == "circle":
            return cv2.HoughCircles(img.copy(), cv2.HOUGH_GRADIENT, dp=1, minDist=80,
                                    param1=200, param2=10, minRadius=40, maxRadius=100)

    def calculation(self, img,  lines):
        total = 0.0
        cnt = 0
            
        if lines is None:
            return None

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                cnt += 1
            else:
                m = (y2 - y1) / (x2 - x1)
                if abs(m) > 1:
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    theta = np.arctan(abs(m)) - np.pi / 2
                    if m > 0:
                        theta = theta * -1

                    total += theta
                    cnt += 1 
       
        
        self.plothistogram(img)
        
        global lineL, lineR

        if lineR > RMax:
            lineL -= laneC
            lineR -= laneC
            return min(-1 * laneMove, result)
        if lineR < Rmin:
            lineL += laneC
            lineR += laneC
            return max(laneMove, result)
        
        if cnt == 0:
            return -999
        
        result = total / cnt

        return result

    def morphology(self, img, kernel_size=(None, None), mode="opening"):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    def region_of_interest(self, img, vertices):
        mask = np.zeros_like(img)   
        if len(img.shape) > 2:
            channel_count = img.shape[2]  
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def plothistogram(self, image):
        global lineL, lineR
        histogram = np.sum(image[7 * image.shape[0]//8:, :], axis=0)
        indices = np.where(histogram >= 7000)[0]
        if indices.size > 0:
            tempR = indices[np.argmin(np.abs(indices - lineR))]
            tempL = indices[np.argmin(np.abs(indices - lineL))]
            if np.abs(tempR - lineR) < 30:
                a = tempR - lineR
                lineR += a
                lineL += a
                return
            if np.abs(tempL - lineL) < 30:
                a = tempL - lineL
                lineR += a
                lineL += a
                return

    def wrapping(self, image):
        points = [[ 87, 357],
    [559, 357],
    [433, 119],
    [226, 119]]

        height, width = image.shape[0], image.shape[1]
        scaled_points = [(int(p[0]), int(p[1])) for p in points]
        
        src_points = np.float32([scaled_points[0], scaled_points[1], scaled_points[3], scaled_points[2]])
        dst_points = np.float32([[160, height], [width - 160, height], [160, 0], [width - 160, 0]])
        
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        bird_eye_view = cv2.warpPerspective(image, matrix, (width, height))
        return bird_eye_view

    def detect(self, cap):
        if not cap.isOpened():
            return None
        
        ret, img = cap.read()
        if not ret:
            return None
        
        bird_eye_view = self.wrapping(img) 

        # 그레이스케일 변환
        gray = self.grayscale(bird_eye_view)
        
        _, binary_img = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        blur_gray = self.gaussian_blur(binary_img, (5, 5))
        
        hist = self.histogram_equalization(blur_gray)
        dst = self.morphology(hist, (2, 2), mode="opening")

        # 가우시안 블러 적용
        cv2.imshow("dst", dst)
        imshape = img.shape
        vertices = np.array([[
            (160, imshape[0] * 1.0),          
            (100, imshape[0] * 0),
            (540, imshape[0] * 0),
            (496, imshape[0] * 1)         
        ]], dtype=np.int32)
       
        masked = self.region_of_interest(dst, vertices)
        canny = self.canny_edge(masked, 150, 200)
        lines = self.hough_transform(canny, 1, np.pi/180, 50, 50, 20, mode="lineP")

        reward = self.calculation(dst, lines)

        return reward

class Arduino:
    def __init__(self):
        # 초기 상태 설정
        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.arduino = serial.Serial(port='COM5', baudrate=9600, timeout=.1)
        time.sleep(2.0)
        self.computerVision = ComputerVision()

    def __delete__(self):
        self.cap.release()
        self.arduino.release()

    def move(self):
        # 모터 회전 후 이동
        try:
            check = 0
            while True:
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

