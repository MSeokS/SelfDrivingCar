import math
import cv2
import numpy as np
import time

target_theta = 1.2  # 예상 radian

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def plothistogram(image):
    histogram = np.sum(image[image.shape[0]//2:, :], axis=0)
    midpoint = np.int32(histogram.shape[0]/2)
    leftbase = np.argmax(histogram[:midpoint])
    rightbase = np.argmax(histogram[midpoint:]) + midpoint
    
    return leftbase, rightbase

def slide_window_search(binary_warped, left_current, right_current):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    nwindows = 4
    window_height = np.int32(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()  # 선이 있는 부분의 인덱스만 저장 
    nonzero_y = np.array(nonzero[0])  # 선이 있는 부분 y의 인덱스 값
    nonzero_x = np.array(nonzero[1])  # 선이 있는 부분 x의 인덱스 값 
    margin = 100
    minpix = 50
    left_lane = []
    right_lane = []
    color = [0, 255, 0]
    thickness = 2

    for w in range(nwindows):
        win_y_low = binary_warped.shape[0] - (w + 1) * window_height  # window 윗부분
        win_y_high = binary_warped.shape[0] - w * window_height  # window 아랫 부분
        win_xleft_low = max(0, left_current - margin)  # 왼쪽 window 왼쪽 위
        win_xleft_high = min(left_current + margin, binary_warped.shape[1])  # 왼쪽 window 오른쪽 아래
        win_xright_low = max(0, right_current - margin)  # 오른쪽 window 왼쪽 위 
        win_xright_high = min(right_current + margin, binary_warped.shape[1])  # 오른쪽 window 오른쪽 아래
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), color, thickness)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), color, thickness)
        good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]
        left_lane.append(good_left)
        right_lane.append(good_right)

        if len(good_left) > minpix:
            left_current = np.int32(np.mean(nonzero_x[good_left]))
        if len(good_right) > minpix:
            right_current = np.int32(np.mean(nonzero_x[good_right]))

    left_lane = np.concatenate(left_lane)  # np.concatenate() -> array를 1차원으로 합침
    right_lane = np.concatenate(right_lane)

    leftx = nonzero_x[left_lane]
    lefty = nonzero_y[left_lane]
    rightx = nonzero_x[right_lane]
    righty = nonzero_y[right_lane]

    if len(leftx) == 0 or len(rightx)==0:
        return None, None, None, None

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    ltx = np.trunc(left_fitx)  # np.trunc() -> 소수점 부분을 버림
    rtx = np.trunc(right_fitx)

    out_img[nonzero_y[left_lane], nonzero_x[left_lane]] = [255, 0, 0]
    out_img[nonzero_y[right_lane], nonzero_x[right_lane]] = [0, 0, 255]

    for i in range(len(ploty)):
        cv2.line(out_img, (int(left_fitx[i]), int(ploty[i])), (int(left_fitx[i]), int(ploty[i])), (255, 255, 0), 2)
        cv2.line(out_img, (int(right_fitx[i]), int(ploty[i])), (int(right_fitx[i]), int(ploty[i])), (255, 255, 0), 2)
    
    cv2.imshow('oo', out_img)
    return ltx, rtx, ploty

def take_picture(cap):
    if not cap.isOpened():
        return None, None
    
    ret, img = cap.read()
    if not ret:
        return None, None
    
#    cv2.imshow('main', img)
#    img = wrapping(img)
    # 2. 이미지를 HSV 색 공간으로 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 3. 흰색 범위 설정
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 55, 255])

    # 4. 마스크 생성
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 5. 결과 이미지 생성 (흰색 부분만 추출)
    img = cv2.bitwise_and(img, img, mask=mask)

    # 그레이스케일 변환
    gray = grayscale(img)

    # 가우시안 블러 적용
    blur_gray = gaussian_blur(gray, 5)

    _, binary_img = cv2.threshold(blur_gray, 220, 255, cv2.THRESH_BINARY)

    left, right = plothistogram(binary_img)
    ltx, rtx, ploty = slide_window_search(binary_img, left, right)
    
    binary_img = binary_img / 255.0
    binary_img = np.expand_dims(binary_img, axis=0)
    binary_img = np.expand_dims(binary_img, axis=-1)
    if ltx is None:
        return binary_img, -100
    else:
        weight = 10
        if ltx[len(ltx) - 1] > 200:
            weight -= 10
        if rtx[len(rtx) - 1] < 440:
            weight -= 10
        if weight == -10:
            weight = -100
        return binary_img, weight


if __name__ == '__main__':
    cap = cv2.VideoCapture("/dev/video2")
    while True:
        time.sleep(0.1)
        img, reward = take_picture(cap)
        img = np.squeeze(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;
    cap.release()

