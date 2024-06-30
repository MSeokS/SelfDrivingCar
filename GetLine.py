import math
import cv2
import numpy as np

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
    draw_lines(line_img, lines)
    weight_line = weight_cal(lines)
    return line_img, weight_line

def draw_lines(img, lines, color=[255, 255, 255], thickness=2):
    """검출된 직선을 이미지에 그림"""
    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

def weight_cal(lines):
    cnt = 0.0
    total = 0.0
    if lines is None:
        print("done")
        return -100
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                total += 1.0
                cnt += 1
                continue

            m = (y2 - y1) / (x2 - x1)
            if -0.1 < m < 0.1:
                continue

            theta = math.atan(m)
            weight = abs(math.sin(theta))

            total += weight
            cnt += 1

    if cnt == 0:
        print("done");
        return -100
    else:
        return (total / cnt)

def take_picture(cap):
    """if not cap.isOpened():
        return None, None
    
    ret, img = cap.read()
    if not ret:
        return None, None"""
    img = cv2.imread("pic.jpg")
    img = cv2.resize(img, (300, 400))
    
    # 그레이스케일 변환
    gray = grayscale(img)
    _, gray = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

    # 가우시안 블러 적용
    blur_gray = gaussian_blur(gray, 5)

    # Canny 엣지 검출 적용
    edges = canny(blur_gray, 100, 150)

    # 관심 영역 설정
    imshape = img.shape
    print(imshape)
    vertices = np.array([[
        (imshape[1] * 0, imshape[0] * 0.95),          # 아래 왼쪽
        (imshape[1] * 0.1, imshape[0] * 0.2),   # 위쪽 왼쪽
        (imshape[1] * 0.9, imshape[0] * 0.2),   # 위쪽 오른쪽
        (imshape[1] * 1.0, imshape[0] * 0.95)           # 아래 오른쪽
    ]], dtype=np.int32)

    masked_edges = region_of_interest(edges, vertices)

    # 허프 변환을 사용하여 직선 검출
    line_image, weight_line = hough_lines(masked_edges, 1, np.pi/180, 15, 40, 20)

    lines_edges = weighted_img(line_image, np.zeros(imshape, dtype=np.uint8))
    
    cv2.imshow('pic', lines_edges)
    cv2.waitKey(0)
    print(weight_line)

    # 결과 출력
    return lines_edges, weight_line

take_picture(0)
