import math
import cv2
import numpy as np

target_theta = 1.4

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

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

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """허프 변환을 사용하여 직선 검출"""
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    theta, ver_lines = theta_cal(img, lines)
    draw_lines(line_img, ver_lines)
    return line_img, theta

def draw_lines(img, lines, color=[255, 255, 255], thickness=2):
    """검출된 직선을 이미지에 그림"""
    if lines is None:
        return
    for line in lines:
        (x1, y1, x2, y2) = line
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def weighted_img(img, initial_img, α=1, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

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

def reward_cal(left, right):
    diff_left = abs(left - target_theta)
    diff_right = abs(-right - target_theta)

    reward = max(0, 1 - diff_left) + max(0, 1 - diff_right)

    return reward

def take_picture(cap):
    """if not cap.isOpened():
        return None, None
    
    ret, img = cap.read()
    if not ret:
        return None, None"""
    img = cv2.imread("pic.jpg")
    img = cv2.resize(img, (640, 360))
    
    # 그레이스케일 변환
    gray = grayscale(img)
    _, gray = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

    # 가우시안 블러 적용
    blur_gray = gaussian_blur(gray, 5)

    # Canny 엣지 검출 적용
    edges = canny(blur_gray, 100, 150)

    # 관심 영역 설정
    imshape = img.shape
    vertices_right = np.array([[
        (imshape[1] * 0.6, imshape[0] * 0.95),          # 아래 왼쪽
        (imshape[1] * 0.6, imshape[0] * 0.2),   # 위쪽 왼쪽
        (imshape[1] * 0.9, imshape[0] * 0.2),   # 위쪽 오른쪽
        (imshape[1] * 1.0, imshape[0] * 0.95)           # 아래 오른쪽
    ]], dtype=np.int32)
    
    vertices_left = np.array([[
        (imshape[1] * 0, imshape[0] * 0.95),          # 아래 왼쪽
        (imshape[1] * 0.1, imshape[0] * 0.2),   # 위쪽 왼쪽
        (imshape[1] * 0.4, imshape[0] * 0.2),   # 위쪽 오른쪽
        (imshape[1] * 0.4, imshape[0] * 0.95)           # 아래 오른쪽
    ]], dtype=np.int32)

    img_copy = blur_gray.copy()
    masked_left = cv2.polylines(img_copy, vertices_left, isClosed=True, color=255, thickness=5)
    masked_right = cv2.polylines(img_copy, vertices_right, isClosed=True, color=255, thickness=5)

    cv2.imshow('range', img_copy)

    masked_edges_left = region_of_interest(edges, vertices_left)
    masked_edges_right = region_of_interest(edges, vertices_right)
  
    # 허프 변환을 사용하여 직선 검출
    line_image_L, theta_L = hough_lines(masked_edges_left, 1, np.pi/180, 15, 40, 20)
    line_image_R, theta_R = hough_lines(masked_edges_right, 1, np.pi/180, 15, 40, 20)

    lines_edges = weighted_img(line_image_R, line_image_L)
    
    cv2.imshow('pic', lines_edges)
    cv2.waitKey(0)

    print(theta_L)
    print(theta_R)

    weight = reward_cal(theta_L, theta_R)
    if weight == 0:
        weight = -100

    print(weight)

    # 결과 출력
    return lines_edges, weight

take_picture(0)
