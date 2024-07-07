import numpy as np
import serial
import cv2
import time
import sys

# 하이퍼파라미터 설정

class ComputerVision(object):
    def grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
       
        cv2.imshow('line', img)

        hist = self.plothistogram(img)
        print(hist)
        """
        if hist < 400 :
            return 0.2
        if hist > 500:
            return -0.2
        """
        if cnt == 0:
            return None

        result = total / cnt

        return result

    def morphology(self, img, kernel_size=(None, None), mode="opening"):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

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
        histogram = np.sum(image[image.shape[0]//2:, :], axis=0)
        rightbase = np.argmax(histogram[:])
    
        return rightbase


    def wrapping(self, image):
        points = [[ 19, 359],
      [598, 358],
      [515, 167],
      [123, 172]]

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
        hist = self.histogram_equalization(gray)
        dst = self.morphology(hist, (2, 2), mode="opening")

        # 가우시안 블러 적용
        blur_gray = self.gaussian_blur(gray, (5, 5))
        _, blur_gray = cv2.threshold(blur_gray, 220, 255, cv2.THRESH_BINARY)
        
        imshape = img.shape
        vertices = np.array([[
            (160, imshape[0] * 1.0),          
            (100, imshape[0] * 0),
            (540, imshape[0] * 0),
            (496, imshape[0] * 1)         
        ]], dtype=np.int32)
       

        #cv2.polylines(bird_eye_view, vertices, isClosed=True, color=(0, 255, 0), thickness=2)

        masked = self.region_of_interest(blur_gray, vertices)
        canny = self.canny_edge(masked, 150, 200)
        
        lines = self.hough_transform(canny, 1, np.pi/180, 50, 10, 20, mode="lineP")
        #lines = self.hough_transform(canny, 1, np.pi/180, 100, 100, 10, mode="lineP")

        
        reward = self.calculation(canny, lines)

        return reward

class Arduino:
    def __init__(self):
        # 초기 상태 설정
        self.cap = cv2.VideoCapture("/dev/video2")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.arduino = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=.1)
        time.sleep(2.0)
        self.computerVision = ComputerVision()

    def __delete__(self):
        self.cap.release()

    def move(self):
        # 모터 회전 후 이동
        try:
            while True:
                self.reward = self.take_picture()
                
                if self.reward is None:
                    continue
                
                k = 50
                angle = self.reward * k + 17
                
                angle = int(angle)
                print(angle)
                
                time.sleep(0.1)
                self.arduino.flush()
                self.arduino.write(f"{angle}\n".encode('utf-8'))
                cv2.waitKey(1)
                
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

