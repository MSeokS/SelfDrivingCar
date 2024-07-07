import numpy as np
import serial
import cv2
import time

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

    def calculation(self, img, lines):
        total = 0.0
        cnt = 0
        img = np.zeros_like(img)

        if lines is None:
            return 0

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs((y2 - y1) / (x2 - x1)) > 0.1:
                cv2.line(img, (x1, y1), (x2, y2), 255, 5)
                total += np.arctan((y2 - y1) / (x2 - x1))
                cnt += 1
        
        cv2.imshow('line', img)
        result = total / cnt
        result -= np.pi / 2

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

    def wrapping(self, image):
        points = [[ 37, 717],
  [709, 714],
  [483, 438],
  [241, 446]]


        height, width = image.shape[0], image.shape[1]
        scaled_points = [(int(p[0]), int(p[1])) for p in points]
        
        src_points = np.float32([scaled_points[0], scaled_points[1], scaled_points[3], scaled_points[2]])
        dst_points = np.float32([[180, height], [width - 180, height], [180, 0], [width - 180, 0]])
        
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        bird_eye_view = cv2.warpPerspective(image, matrix, (width, height))
        return bird_eye_view

    def detect(self, cap):
        if not cap.isOpened():
            return None
        
        ret, img = cap.read()
        if not ret:
            return None
        print(img.shape[0], img.shape[1])
        bird_eye_view = self.wrapping(img) 
        
        cv2.imshow('main', bird_eye_view)

        # 그레이스케일 변환
        gray = self.grayscale(bird_eye_view)
        hist = self.histogram_equalization(gray)
        dst = self.morphology(hist, (2, 2), mode="opening")

        # 가우시안 블러 적용
        blur_gray = self.gaussian_blur(gray, (5, 5))
        _, blur_gray = cv2.threshold(blur_gray, 100, 255, cv2.THRESH_BINARY)
       
        canny = self.canny_edge(blur_gray, 50, 150)
        
        """ 
        imshape = canny.shape
        vertices = np.array([[
            (imshape[1] * 0, imshape[0] * 1.0),          
            (imshape[1] * 0, imshape[0] * 0.4),         
            (imshape[1] * 0.3, imshape[0] * 0.4),
            (imshape[1] * 0.3, imshape[0] * 1.0), 
            (imshape[1] * 0.7, imshape[0] * 1.0),
            (imshape[1] * 0.7, imshape[0] * 0.4),
            (imshape[1] * 1, imshape[0] * 0.4),
            (imshape[1] * 1, imshape[0] * 1)         
        ]], dtype=np.int32)
        
        masked = self.region_of_interest(canny, vertices)
        """
        masked = canny
        
        lines = self.hough_transform(masked, 1, np.pi/180, 50, 10, 20, mode="lineP")
        #lines = self.hough_transform(canny, 1, np.pi/180, 100, 100, 10, mode="lineP")
        
        reward = self.calculation(masked, lines)
        
        cv2.imshow('img', masked)
        cv2.waitKey(1)
        
        print(reward)

        return reward

if __name__ == "__main__":
    cap = cv2.VideoCapture("curv.mp4")
    run = ComputerVision()
    while True:
        reward = run.detect(cap)
        if reward is None:
            break
        time.sleep(0.1)
    cap.release()

