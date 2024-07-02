import numpy as np
import serial
import cv2

# 하이퍼파라미터 설정
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    if len(img.shape) > 2:
        channel_count = img.shape[2]  
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def theta_cal(img, lines):
    cnt = 0.0
    total = 0.0
    ver_lines = []
    if lines is None:
        return 0, None

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                total += 1.0
                cnt += 1
                continue

            m = (y1 - y2) / (x2 - x1)
            if -0.1 < m < 0.1:
                continue

            ver_lines.append((x1, y1, x2, y2))

            theta = math.atan(m)

            total += theta
            cnt += 1

    if cnt == 0:
        return 0, None
    else:
        return (total / cnt), ver_lines

def wrapping(image):
    points = [[ 19, 359],
  [598, 358],
  [515, 167],
  [123, 172]]

    height, width = image.shape[0], image.shape[1]
    scaled_points = [(int(p[0]), int(p[1])) for p in points]
    
    src_points = np.float32([scaled_points[0], scaled_points[1], scaled_points[3], scaled_points[2]])
    dst_points = np.float32([[100, height], [width - 100, height], [100, 0], [width - 100, 0]])
    
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    bird_eye_view = cv2.warpPerspective(image, matrix, (width, height))
    return bird_eye_view

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
        #cv2.imshow("oo", out_img)

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

    if len(leftx) == 0 or len(rightx) == 0:
        return None, None, None

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
    
    cv2.imshow("oo", out_img)

    
    return ltx, rtx, ploty

def picture(cap):
    if not cap.isOpened():
        return None, None
    
    ret, img = cap.read()
    if not ret:
        return None, None

    #cv2.imshow('main', img)
    img = wrapping(img) 
    #cv2.imshow('bev', img)

    # 그레이스케일 변환
    gray = grayscale(img)
    _, gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # 가우시안 블러 적용
    blur_gray = gaussian_blur(gray, 5)
    
    _, binary_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    imshape = binary_img.shape
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

    masked = region_of_interest(binary_img, vertices)

    left, right = plothistogram(masked)
    ltx, rtx, ploty =slide_window_search(masked, left, right)

    if ltx is None:
        return 0

    if ltx[len(ltx) - 1] > 200:
        return 0.1
    if rtx[len(rtx) - 1] < 440:
        return -0.8
    mean_start = (ltx[len(ltx) - 1] + rtx[len(rtx) - 1]) / 2
    mean_end = (ltx[0] + rtx[0]) / 2
    
    k = 1
    reward =  ((mean_end - mean_start) / (ploty[0] - ploty[len(ploty) - 1])) * k
    print(reward)
    return reward

class StateTransition:
    def __init__(self):
        # 초기 상태 설정
        self.cap = cv2.VideoCapture("/dev/video2")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.reward = self.take_picture()
        self.arduino = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=.1)

    def move(self):
        # 모터 회전 후 이동
        while True:
            self.reward = self.take_picture()
            angle = self.reward * 10 + 20
            angle = int(angle)
            print(angle)
            self.arduino.write(f"{angle}\n".encode('utf-8'))
            cv2.waitKey(1)
    
    def take_picture(self):
        reward = picture(self.cap)
        if reward is None:
            print("Camera disconnect.")
            sys.exit()
        else:
            return reward

if __name__ == "__main__":
    run = StateTransition()
    run.move()

